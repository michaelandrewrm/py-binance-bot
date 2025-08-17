"""
Baseline - ATR/vol bands heuristic (baseline)

This module implements baseline trading strategies using traditional
technical indicators like ATR and volatility bands.
"""

import numpy as np

from typing import List, Dict, Optional, Tuple, Union
from decimal import Decimal
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

# Import for backtesting compatibility
try:
    from core.state import Order, OrderSide
except ImportError:
    # Fallback for when not using backtesting engine
    Order = None
    OrderSide = None

from data.schema import KlineData, FeatureData, ModelPrediction
from core.state import Order, OrderSide, OrderType, OrderStatus, Position


class BaselineStrategyType(Enum):
    """Types of baseline strategies available"""

    VOLATILITY_BANDS = "volatility_bands"
    MEAN_REVERSION = "mean_reversion"
    ENSEMBLE = "ensemble"
    AUTO_SELECT = "auto_select"


@dataclass
class SignalContext:
    """Context information for trading signals"""

    symbol: str
    timestamp: datetime
    current_price: Decimal
    volume: Decimal
    indicators: Dict[str, Decimal]
    market_conditions: Dict[str, any]


@dataclass
class TradingSignal:
    """Standardized trading signal"""

    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: Decimal  # 0.0 to 1.0
    price: Decimal
    size: Decimal
    reason: str
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    strategy_source: str = ""
    context: Optional[SignalContext] = None


class TechnicalIndicators:
    """Collection of technical indicators for baseline strategies"""

    @staticmethod
    def sma(prices: List[Decimal], period: int) -> List[Optional[Decimal]]:
        """Simple Moving Average"""
        if len(prices) < period:
            return [None] * len(prices)

        result = [None] * (period - 1)
        for i in range(period - 1, len(prices)):
            avg = sum(prices[i - period + 1 : i + 1]) / period
            result.append(avg)

        return result

    @staticmethod
    def ema(
        prices: List[Decimal], period: int, alpha: Optional[Decimal] = None
    ) -> List[Optional[Decimal]]:
        """Exponential Moving Average"""
        if not prices:
            return []

        if alpha is None:
            alpha = Decimal("2") / (period + 1)

        result = [None] * len(prices)
        result[0] = prices[0]

        for i in range(1, len(prices)):
            result[i] = alpha * prices[i] + (1 - alpha) * result[i - 1]

        return result

    @staticmethod
    def atr(
        highs: List[Decimal],
        lows: List[Decimal],
        closes: List[Decimal],
        period: int = 14,
    ) -> List[Optional[Decimal]]:
        """Average True Range"""
        if len(highs) < 2 or len(highs) != len(lows) or len(highs) != len(closes):
            return [None] * len(highs)

        true_ranges = []

        # First TR is just high - low
        true_ranges.append(highs[0] - lows[0])

        # Calculate True Range for each period
        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]  # High - Low
            tr2 = abs(highs[i] - closes[i - 1])  # High - Previous Close
            tr3 = abs(lows[i] - closes[i - 1])  # Low - Previous Close

            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)

        # Calculate ATR using SMA of True Ranges
        return TechnicalIndicators.sma(true_ranges, period)

    @staticmethod
    def bollinger_bands(
        prices: List[Decimal], period: int = 20, std_dev: Decimal = Decimal("2")
    ) -> Tuple[
        List[Optional[Decimal]],
        List[Optional[Decimal]],
        List[Optional[Decimal]],
    ]:
        """Bollinger Bands"""
        if len(prices) < period:
            return (
                [None] * len(prices),
                [None] * len(prices),
                [None] * len(prices),
            )

        sma_values = TechnicalIndicators.sma(prices, period)
        upper_bands = []
        lower_bands = []

        for i in range(len(prices)):
            if i < period - 1 or sma_values[i] is None:
                upper_bands.append(None)
                lower_bands.append(None)
            else:
                # Calculate standard deviation for the period
                period_prices = prices[i - period + 1 : i + 1]
                mean = sma_values[i]
                variance = sum((price - mean) ** 2 for price in period_prices) / period
                std = variance ** Decimal("0.5")

                upper_bands.append(mean + std_dev * std)
                lower_bands.append(mean - std_dev * std)

        return sma_values, upper_bands, lower_bands

    @staticmethod
    def rsi(prices: List[Decimal], period: int = 14) -> List[Optional[Decimal]]:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return [None] * len(prices)

        gains = []
        losses = []

        # Calculate price changes
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(Decimal("0"))
            else:
                gains.append(Decimal("0"))
                losses.append(abs(change))

        result = [None] * len(prices)

        if len(gains) < period:
            return result

        # Calculate initial average gain and loss
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        # Calculate RSI
        for i in range(period, len(gains)):
            if avg_loss == 0:
                result[i + 1] = Decimal("100")
            else:
                rs = avg_gain / avg_loss
                rsi = Decimal("100") - (Decimal("100") / (Decimal("1") + rs))
                result[i + 1] = rsi

            # Update averages for next iteration
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        return result


class VolatilityBandStrategy:
    """
    Baseline strategy using volatility bands for entry/exit signals
    """

    def __init__(
        self,
        atr_period: int = 14,
        atr_multiplier: Decimal = Decimal("2"),
        sma_period: int = 20,
        min_volatility: Decimal = Decimal("0.01"),
    ):
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.sma_period = sma_period
        self.min_volatility = min_volatility

        self.position = 0  # 1 for long, -1 for short, 0 for flat
        self.entry_price: Optional[Decimal] = None
        self.stop_loss: Optional[Decimal] = None
        self.take_profit: Optional[Decimal] = None

    def generate_signals(self, klines: List[KlineData]) -> List[Dict]:
        """
        Generate trading signals based on volatility bands

        Returns:
            List of signal dictionaries
        """
        if len(klines) < max(self.atr_period, self.sma_period) + 1:
            return []

        # Extract price data
        highs = [k.high_price for k in klines]
        lows = [k.low_price for k in klines]
        closes = [k.close_price for k in klines]

        # Calculate indicators
        sma_values = TechnicalIndicators.sma(closes, self.sma_period)
        atr_values = TechnicalIndicators.atr(highs, lows, closes, self.atr_period)

        signals = []

        for i in range(len(klines)):
            if sma_values[i] is None or atr_values[i] is None:
                continue

            current_price = closes[i]
            sma = sma_values[i]
            atr = atr_values[i]

            # Skip if volatility is too low
            if atr < self.min_volatility:
                continue

            # Calculate bands
            upper_band = sma + self.atr_multiplier * atr
            lower_band = sma - self.atr_multiplier * atr

            signal = self._evaluate_position(
                klines[i], current_price, sma, upper_band, lower_band, atr
            )

            if signal:
                signals.append(signal)

        return signals

    def _evaluate_position(
        self,
        kline: KlineData,
        price: Decimal,
        sma: Decimal,
        upper_band: Decimal,
        lower_band: Decimal,
        atr: Decimal,
    ) -> Optional[Dict]:
        """Evaluate current position and generate signals"""

        signal = None

        if self.position == 0:  # No position
            # Look for entry signals
            if price > upper_band and price > sma:
                # Long entry signal
                signal = {
                    "timestamp": kline.open_time,
                    "symbol": kline.symbol,
                    "action": "BUY",
                    "price": price,
                    "size": Decimal("1.0"),  # Full position
                    "reason": "breakout_above_upper_band",
                    "stop_loss": price - atr * self.atr_multiplier,
                    "take_profit": price + atr * self.atr_multiplier * 2,
                }
                self.position = 1
                self.entry_price = price
                self.stop_loss = signal["stop_loss"]
                self.take_profit = signal["take_profit"]

            elif price < lower_band and price < sma:
                # Short entry signal
                signal = {
                    "timestamp": kline.open_time,
                    "symbol": kline.symbol,
                    "action": "SELL",
                    "price": price,
                    "size": Decimal("1.0"),
                    "reason": "breakdown_below_lower_band",
                    "stop_loss": price + atr * self.atr_multiplier,
                    "take_profit": price - atr * self.atr_multiplier * 2,
                }
                self.position = -1
                self.entry_price = price
                self.stop_loss = signal["stop_loss"]
                self.take_profit = signal["take_profit"]

        elif self.position == 1:  # Long position
            # Check exit conditions
            if price <= self.stop_loss or price >= self.take_profit or price < sma:
                reason = (
                    "stop_loss"
                    if price <= self.stop_loss
                    else (
                        "take_profit" if price >= self.take_profit else "trend_reversal"
                    )
                )

                signal = {
                    "timestamp": kline.open_time,
                    "symbol": kline.symbol,
                    "action": "SELL",
                    "price": price,
                    "size": Decimal("1.0"),
                    "reason": reason,
                    "pnl": price - self.entry_price,
                }
                self._reset_position()

        elif self.position == -1:  # Short position
            # Check exit conditions
            if price >= self.stop_loss or price <= self.take_profit or price > sma:
                reason = (
                    "stop_loss"
                    if price >= self.stop_loss
                    else (
                        "take_profit" if price <= self.take_profit else "trend_reversal"
                    )
                )

                signal = {
                    "timestamp": kline.open_time,
                    "symbol": kline.symbol,
                    "action": "BUY",
                    "price": price,
                    "size": Decimal("1.0"),
                    "reason": reason,
                    "pnl": self.entry_price - price,
                }
                self._reset_position()

        return signal

    def _reset_position(self):
        """Reset position state"""
        self.position = 0
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None


class MeanReversionStrategy:
    """
    Baseline mean reversion strategy using Bollinger Bands
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: Decimal = Decimal("2"),
        rsi_period: int = 14,
        rsi_oversold: Decimal = Decimal("30"),
        rsi_overbought: Decimal = Decimal("70"),
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

        self.position = 0
        self.entry_price: Optional[Decimal] = None

    def generate_signals(self, klines: List[KlineData]) -> List[Dict]:
        """Generate mean reversion signals"""
        if len(klines) < max(self.bb_period, self.rsi_period) + 1:
            return []

        closes = [k.close_price for k in klines]

        # Calculate indicators
        sma, upper_bb, lower_bb = TechnicalIndicators.bollinger_bands(
            closes, self.bb_period, self.bb_std
        )
        rsi_values = TechnicalIndicators.rsi(closes, self.rsi_period)

        signals = []

        for i in range(len(klines)):
            if (
                sma[i] is None
                or upper_bb[i] is None
                or lower_bb[i] is None
                or rsi_values[i] is None
            ):
                continue

            current_price = closes[i]
            signal = self._evaluate_mean_reversion(
                klines[i],
                current_price,
                sma[i],
                upper_bb[i],
                lower_bb[i],
                rsi_values[i],
            )

            if signal:
                signals.append(signal)

        return signals

    def _evaluate_mean_reversion(
        self,
        kline: KlineData,
        price: Decimal,
        sma: Decimal,
        upper_bb: Decimal,
        lower_bb: Decimal,
        rsi: Decimal,
    ) -> Optional[Dict]:
        """Evaluate mean reversion opportunities"""

        signal = None

        if self.position == 0:
            # Look for oversold conditions
            if price <= lower_bb and rsi <= self.rsi_oversold:
                signal = {
                    "timestamp": kline.open_time,
                    "symbol": kline.symbol,
                    "action": "BUY",
                    "price": price,
                    "size": Decimal("1.0"),
                    "reason": "oversold_mean_reversion",
                    "target": sma,
                }
                self.position = 1
                self.entry_price = price

            # Look for overbought conditions
            elif price >= upper_bb and rsi >= self.rsi_overbought:
                signal = {
                    "timestamp": kline.open_time,
                    "symbol": kline.symbol,
                    "action": "SELL",
                    "price": price,
                    "size": Decimal("1.0"),
                    "reason": "overbought_mean_reversion",
                    "target": sma,
                }
                self.position = -1
                self.entry_price = price

        elif self.position == 1:
            # Exit long position when price reaches SMA or RSI becomes overbought
            if price >= sma or rsi >= self.rsi_overbought:
                signal = {
                    "timestamp": kline.open_time,
                    "symbol": kline.symbol,
                    "action": "SELL",
                    "price": price,
                    "size": Decimal("1.0"),
                    "reason": "mean_reversion_target",
                    "pnl": price - self.entry_price,
                }
                self._reset_position()

        elif self.position == -1:
            # Exit short position when price reaches SMA or RSI becomes oversold
            if price <= sma or rsi <= self.rsi_oversold:
                signal = {
                    "timestamp": kline.open_time,
                    "symbol": kline.symbol,
                    "action": "BUY",
                    "price": price,
                    "size": Decimal("1.0"),
                    "reason": "mean_reversion_target",
                    "pnl": self.entry_price - price,
                }
                self._reset_position()

        return signal

    def _reset_position(self):
        """Reset position state"""
        self.position = 0
        self.entry_price = None


class BaselineEvaluator:
    """
    Evaluator for baseline strategies to establish performance benchmarks
    """

    def __init__(self):
        self.volatility_strategy = VolatilityBandStrategy()
        self.mean_reversion_strategy = MeanReversionStrategy()

    def evaluate_all_strategies(self, klines: List[KlineData]) -> Dict:
        """Evaluate all baseline strategies"""
        results = {}

        # Volatility band strategy
        vol_signals = self.volatility_strategy.generate_signals(klines)
        results["volatility_bands"] = {
            "signals": vol_signals,
            "total_signals": len(vol_signals),
            "metrics": self._calculate_signal_metrics(vol_signals),
        }

        # Mean reversion strategy
        mr_signals = self.mean_reversion_strategy.generate_signals(klines)
        results["mean_reversion"] = {
            "signals": mr_signals,
            "total_signals": len(mr_signals),
            "metrics": self._calculate_signal_metrics(mr_signals),
        }

        return results

    def _calculate_signal_metrics(self, signals: List[Dict]) -> Dict:
        """Calculate basic metrics for signals"""
        if not signals:
            return {
                "total_pnl": Decimal("0"),
                "win_rate": Decimal("0"),
                "trades": 0,
            }

        trade_pnls = [
            signal.get("pnl", Decimal("0")) for signal in signals if signal.get("pnl")
        ]

        if not trade_pnls:
            return {
                "total_pnl": Decimal("0"),
                "win_rate": Decimal("0"),
                "trades": 0,
            }

        total_pnl = sum(trade_pnls)
        winning_trades = sum(1 for pnl in trade_pnls if pnl > 0)
        win_rate = (
            Decimal(str(winning_trades / len(trade_pnls)))
            if trade_pnls
            else Decimal("0")
        )

        return {
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "trades": len(trade_pnls),
            "avg_pnl": (total_pnl / len(trade_pnls) if trade_pnls else Decimal("0")),
        }


# Utility functions


def create_baseline_features(klines: List[KlineData]) -> List[FeatureData]:
    """Create baseline technical features from klines"""
    if len(klines) < 50:  # Need minimum data
        return []

    closes = [k.close_price for k in klines]
    highs = [k.high_price for k in klines]
    lows = [k.low_price for k in klines]
    volumes = [k.volume for k in klines]

    # Calculate indicators
    sma_20 = TechnicalIndicators.sma(closes, 20)
    sma_50 = TechnicalIndicators.sma(closes, 50)
    ema_12 = TechnicalIndicators.ema(closes, 12)
    ema_26 = TechnicalIndicators.ema(closes, 26)
    atr_14 = TechnicalIndicators.atr(highs, lows, closes, 14)
    rsi_14 = TechnicalIndicators.rsi(closes, 14)

    features = []

    for i in range(len(klines)):
        if i < 49:  # Skip first 49 for complete indicators
            continue

        feature_dict = {
            "close_price": float(closes[i]),
            "volume": float(volumes[i]),
            "price_change": float((closes[i] - closes[i - 1]) / closes[i - 1]),
            "volume_change": (
                float((volumes[i] - volumes[i - 1]) / volumes[i - 1])
                if volumes[i - 1] > 0
                else 0
            ),
        }

        # Add technical indicators
        if sma_20[i] is not None:
            feature_dict["sma_20"] = float(sma_20[i])
            feature_dict["price_vs_sma20"] = float(closes[i] / sma_20[i] - 1)

        if sma_50[i] is not None:
            feature_dict["sma_50"] = float(sma_50[i])
            feature_dict["sma20_vs_sma50"] = (
                float(sma_20[i] / sma_50[i] - 1) if sma_20[i] and sma_50[i] else 0
            )

        if ema_12[i] is not None and ema_26[i] is not None:
            feature_dict["macd"] = float(ema_12[i] - ema_26[i])
            feature_dict["macd_signal"] = feature_dict["macd"]  # Simplified

        if atr_14[i] is not None:
            feature_dict["atr_14"] = float(atr_14[i])
            feature_dict["volatility"] = float(atr_14[i] / closes[i])

        if rsi_14[i] is not None:
            feature_dict["rsi_14"] = float(rsi_14[i])

        features.append(
            FeatureData(
                symbol=klines[i].symbol,
                timestamp=klines[i].open_time,
                features=feature_dict,
            )
        )

    return features


def run_baseline_benchmark(klines: List[KlineData]) -> Dict:
    """Run all baseline strategies as benchmark"""
    evaluator = BaselineEvaluator()
    return evaluator.evaluate_all_strategies(klines)


class BaselineStrategy:
    """
    Comprehensive baseline trading strategy that integrates multiple baseline approaches.

    This strategy can operate in different modes:
    - Single strategy (volatility bands or mean reversion)
    - Ensemble mode (combines multiple strategies)
    - Auto-select mode (chooses best strategy based on market conditions)

    Compatible with the backtesting engine and implements TradingStrategy interface.
    """

    def __init__(
        self,
        strategy_type: Union[BaselineStrategyType, str] = BaselineStrategyType.ENSEMBLE,
        initial_capital: Decimal = Decimal("10000"),
        commission: Decimal = Decimal("0.001"),
        risk_per_trade: Decimal = Decimal("0.02"),
        max_position_size: Decimal = Decimal("0.5"),
        lookback_period: int = 100,
        strategy_weights: Optional[Dict[str, Decimal]] = None,
    ):
        """
        Initialize BaselineStrategy

        Args:
            strategy_type: Type of baseline strategy to use (enum or string)
            initial_capital: Starting capital for position sizing
            commission: Commission rate for trades
            risk_per_trade: Maximum risk per trade as fraction of capital
            max_position_size: Maximum position size as fraction of capital
            lookback_period: Number of periods to use for indicator calculations
            strategy_weights: Weights for ensemble mode (if None, uses equal weights)
        """
        self.name = "BaselineStrategy"

        # Convert string to enum if needed
        if isinstance(strategy_type, str):
            try:
                self.strategy_type = BaselineStrategyType(strategy_type)
            except ValueError:
                raise ValueError(
                    f"Invalid strategy type: {strategy_type}. Valid options: {[e.value for e in BaselineStrategyType]}"
                )
        else:
            self.strategy_type = strategy_type
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission = commission
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.lookback_period = lookback_period

        # Initialize sub-strategies
        self.volatility_strategy = VolatilityBandStrategy()
        self.mean_reversion_strategy = MeanReversionStrategy()

        # Strategy weights for ensemble mode
        if strategy_weights is None:
            self.strategy_weights = {
                "volatility_bands": Decimal("0.6"),
                "mean_reversion": Decimal("0.4"),
            }
        else:
            self.strategy_weights = strategy_weights

        # State tracking
        self.current_position = 0  # 1 for long, -1 for short, 0 for flat
        self.position_size = Decimal("0")
        self.entry_price: Optional[Decimal] = None
        self.stop_loss: Optional[Decimal] = None
        self.take_profit: Optional[Decimal] = None
        self.last_signal_time: Optional[datetime] = None

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = Decimal("0")
        self.max_drawdown = Decimal("0")
        self.peak_capital = initial_capital

        # Market condition analysis
        self.market_regime = "unknown"  # trending, sideways, volatile
        self.volatility_percentile = Decimal("0.5")

        # Data buffers
        self.price_history: List[Decimal] = []
        self.volume_history: List[Decimal] = []
        self.indicator_buffer: Dict[str, List[Decimal]] = {}

    def generate_signal(self, klines: List[KlineData]) -> Optional[TradingSignal]:
        """
        Generate trading signal from kline data

        Args:
            klines: Historical kline data

        Returns:
            TradingSignal or None if no signal
        """
        if len(klines) < self.lookback_period:
            return None

        # Update internal state
        self._update_market_conditions(klines)
        self._update_price_history(klines)

        current_kline = klines[-1]
        current_price = current_kline.close_price

        # Generate signals from different strategies
        signals = []

        if self.strategy_type == BaselineStrategyType.VOLATILITY_BANDS:
            signal = self._get_volatility_signal(klines)
            if signal:
                signals.append(signal)

        elif self.strategy_type == BaselineStrategyType.MEAN_REVERSION:
            signal = self._get_mean_reversion_signal(klines)
            if signal:
                signals.append(signal)

        elif self.strategy_type == BaselineStrategyType.ENSEMBLE:
            # Get signals from all strategies and combine
            vol_signal = self._get_volatility_signal(klines)
            mr_signal = self._get_mean_reversion_signal(klines)

            combined_signal = self._combine_signals([vol_signal, mr_signal])
            if combined_signal:
                signals.append(combined_signal)

        elif self.strategy_type == BaselineStrategyType.AUTO_SELECT:
            # Choose best strategy based on market conditions
            selected_strategy = self._select_strategy_by_market_conditions()
            if selected_strategy == "volatility_bands":
                signal = self._get_volatility_signal(klines)
            else:
                signal = self._get_mean_reversion_signal(klines)
            if signal:
                signals.append(signal)

        # Return the best signal or None
        return signals[0] if signals else None

    def _get_volatility_signal(
        self, klines: List[KlineData]
    ) -> Optional[TradingSignal]:
        """Get signal from volatility band strategy"""
        vol_signals = self.volatility_strategy.generate_signals(klines)

        if not vol_signals:
            return None

        # Convert last signal to TradingSignal format
        last_signal = vol_signals[-1]

        return TradingSignal(
            action=last_signal["action"],
            confidence=Decimal("0.7"),  # Default confidence for volatility signals
            price=last_signal["price"],
            size=self._calculate_position_size(last_signal["price"]),
            reason=last_signal["reason"],
            stop_loss=last_signal.get("stop_loss"),
            take_profit=last_signal.get("take_profit"),
            strategy_source="volatility_bands",
        )

    def _get_mean_reversion_signal(
        self, klines: List[KlineData]
    ) -> Optional[TradingSignal]:
        """Get signal from mean reversion strategy"""
        mr_signals = self.mean_reversion_strategy.generate_signals(klines)

        if not mr_signals:
            return None

        # Convert last signal to TradingSignal format
        last_signal = mr_signals[-1]

        return TradingSignal(
            action=last_signal["action"],
            confidence=Decimal("0.6"),  # Default confidence for mean reversion signals
            price=last_signal["price"],
            size=self._calculate_position_size(last_signal["price"]),
            reason=last_signal["reason"],
            stop_loss=last_signal.get("stop_loss"),
            take_profit=last_signal.get("take_profit"),
            strategy_source="mean_reversion",
        )

    def _combine_signals(
        self, signals: List[Optional[TradingSignal]]
    ) -> Optional[TradingSignal]:
        """Combine multiple signals using weighted voting"""
        valid_signals = [s for s in signals if s is not None]

        if not valid_signals:
            return None

        if len(valid_signals) == 1:
            return valid_signals[0]

        # Weighted voting for action
        buy_weight = Decimal("0")
        sell_weight = Decimal("0")
        total_confidence = Decimal("0")

        for signal in valid_signals:
            weight = self.strategy_weights.get(signal.strategy_source, Decimal("0.5"))
            weighted_confidence = signal.confidence * weight

            if signal.action == "BUY":
                buy_weight += weighted_confidence
            elif signal.action == "SELL":
                sell_weight += weighted_confidence

            total_confidence += weighted_confidence

        # Determine final action
        if buy_weight > sell_weight and buy_weight > Decimal("0.5"):
            action = "BUY"
            confidence = buy_weight
        elif sell_weight > buy_weight and sell_weight > Decimal("0.5"):
            action = "SELL"
            confidence = sell_weight
        else:
            return None  # No clear signal

        # Use the signal with highest confidence for other parameters
        best_signal = max(valid_signals, key=lambda s: s.confidence)

        return TradingSignal(
            action=action,
            confidence=min(confidence, Decimal("1.0")),
            price=best_signal.price,
            size=self._calculate_position_size(best_signal.price),
            reason=f"ensemble_{len(valid_signals)}_strategies",
            stop_loss=best_signal.stop_loss,
            take_profit=best_signal.take_profit,
            strategy_source="ensemble",
        )

    def _select_strategy_by_market_conditions(self) -> str:
        """Select best strategy based on current market conditions"""
        # Simple heuristic: use volatility bands in trending markets,
        # mean reversion in sideways markets

        if self.market_regime == "trending" or self.volatility_percentile > Decimal(
            "0.7"
        ):
            return "volatility_bands"
        else:
            return "mean_reversion"

    def _calculate_position_size(self, entry_price: Decimal) -> Decimal:
        """Calculate position size based on risk management rules"""
        # Risk-based position sizing
        max_risk_amount = self.current_capital * self.risk_per_trade

        # Estimate risk per unit (simplified - could use ATR or volatility)
        risk_per_unit = entry_price * Decimal("0.02")  # 2% risk per unit

        if risk_per_unit > 0:
            risk_based_size = max_risk_amount / risk_per_unit
        else:
            risk_based_size = Decimal("0")

        # Apply maximum position size constraint
        max_capital_size = self.current_capital * self.max_position_size / entry_price

        # Return the smaller of the two constraints
        return min(risk_based_size, max_capital_size)

    def _update_market_conditions(self, klines: List[KlineData]):
        """Update market regime and condition analysis"""
        if len(klines) < 50:
            return

        # Calculate trend strength and volatility
        closes = [k.close_price for k in klines[-50:]]
        sma_20 = TechnicalIndicators.sma(closes, 20)
        sma_50 = TechnicalIndicators.sma(closes, 50)

        if sma_20[-1] and sma_50[-1]:
            trend_strength = abs(sma_20[-1] - sma_50[-1]) / sma_50[-1]

            if trend_strength > Decimal("0.02"):  # 2% difference
                self.market_regime = "trending"
            else:
                self.market_regime = "sideways"

        # Calculate volatility percentile (simplified)
        if len(closes) >= 20:
            recent_volatility = np.std([float(c) for c in closes[-20:]])
            historical_volatilities = []

            for i in range(20, len(closes)):
                vol = np.std([float(c) for c in closes[i - 20 : i]])
                historical_volatilities.append(vol)

            if historical_volatilities:
                percentile = len(
                    [v for v in historical_volatilities if v < recent_volatility]
                ) / len(historical_volatilities)
                self.volatility_percentile = Decimal(str(percentile))

    def _update_price_history(self, klines: List[KlineData]):
        """Update internal price and volume buffers"""
        if klines:
            current_kline = klines[-1]
            self.price_history.append(current_kline.close_price)
            self.volume_history.append(current_kline.volume)

            # Keep only recent history
            if len(self.price_history) > self.lookback_period * 2:
                self.price_history = self.price_history[-self.lookback_period :]
                self.volume_history = self.volume_history[-self.lookback_period :]

    def update_position(
        self, action: str, price: Decimal, size: Decimal, timestamp: datetime
    ):
        """Update position tracking after trade execution"""
        if action == "BUY":
            if self.current_position <= 0:
                # Opening or flipping to long
                self.current_position = 1
                self.position_size = size
                self.entry_price = price
            else:
                # Adding to long position
                total_size = self.position_size + size
                self.entry_price = (
                    self.entry_price * self.position_size + price * size
                ) / total_size
                self.position_size = total_size

        elif action == "SELL":
            if self.current_position >= 0:
                # Closing long or opening short
                if self.current_position > 0 and self.entry_price:
                    # Calculate P&L for closing long
                    pnl = (price - self.entry_price) * self.position_size
                    self._record_trade(pnl, timestamp)

                self.current_position = -1
                self.position_size = size
                self.entry_price = price
            else:
                # Adding to short position
                total_size = self.position_size + size
                self.entry_price = (
                    self.entry_price * self.position_size + price * size
                ) / total_size
                self.position_size = total_size

        self.last_signal_time = timestamp

    def _record_trade(self, pnl: Decimal, timestamp: datetime):
        """Record completed trade for performance tracking"""
        self.total_trades += 1
        self.total_pnl += pnl

        if pnl > 0:
            self.winning_trades += 1

        # Update capital
        self.current_capital += pnl

        # Track drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        else:
            drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown

    def get_performance_metrics(self) -> Dict[str, Union[Decimal, int, float]]:
        """Get current performance metrics"""
        win_rate = (
            Decimal(self.winning_trades) / Decimal(self.total_trades)
            if self.total_trades > 0
            else Decimal("0")
        )

        total_return = (
            (self.current_capital - self.initial_capital) / self.initial_capital
            if self.initial_capital > 0
            else Decimal("0")
        )

        avg_trade_pnl = (
            self.total_pnl / Decimal(self.total_trades)
            if self.total_trades > 0
            else Decimal("0")
        )

        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": win_rate,
            "total_pnl": self.total_pnl,
            "total_return": total_return,
            "current_capital": self.current_capital,
            "max_drawdown": self.max_drawdown,
            "avg_trade_pnl": avg_trade_pnl,
            "strategy_type": self.strategy_type.value,
            "market_regime": self.market_regime,
            "volatility_percentile": float(self.volatility_percentile),
        }

    def reset(self):
        """Reset strategy state for new backtest"""
        self.current_capital = self.initial_capital
        self.current_position = 0
        self.position_size = Decimal("0")
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.last_signal_time = None

        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = Decimal("0")
        self.max_drawdown = Decimal("0")
        self.peak_capital = self.initial_capital

        self.price_history.clear()
        self.volume_history.clear()
        self.indicator_buffer.clear()

        # Reset sub-strategies
        self.volatility_strategy = VolatilityBandStrategy()
        self.mean_reversion_strategy = MeanReversionStrategy()

    # TradingStrategy interface compatibility methods
    async def on_market_data(self, kline: KlineData) -> List:
        """
        Handle new market data for backtesting engine compatibility

        Args:
            kline: New kline data

        Returns:
            List of orders to place (empty for signal-based strategy)
        """
        # This strategy generates signals rather than direct orders
        # The backtesting engine should call generate_signal separately
        return []

    async def on_order_fill(self, order, fill_price: Decimal) -> List:
        """
        Handle order fill for backtesting engine compatibility

        Args:
            order: Filled order object
            fill_price: Actual fill price

        Returns:
            List of new orders to place (empty for this strategy)
        """
        # Update position tracking based on the filled order
        if Order is not None and hasattr(order, "side"):
            # Full backtesting engine Order object
            action = "BUY" if order.side == OrderSide.BUY else "SELL"
            size = order.quantity
        else:
            # Fallback for simple dict or other order representations
            action = (
                order.get("side", "BUY") if hasattr(order, "get") else str(order.side)
            )
            size = (
                order.get("quantity", Decimal("0"))
                if hasattr(order, "get")
                else order.quantity
            )

        timestamp = datetime.now()
        self.update_position(action, fill_price, size, timestamp)
        return []

    async def on_start(self, state, broker):
        """Called when simulation starts"""
        self.state = state
        self.broker = broker

    async def on_end(self):
        """Called when simulation ends"""
        pass


# Factory function for easy strategy creation
def create_baseline_strategy(
    strategy_type: str = "ensemble", **kwargs
) -> BaselineStrategy:
    """
    Factory function to create BaselineStrategy instances

    Args:
        strategy_type: Type of strategy ("volatility_bands", "mean_reversion", "ensemble", "auto_select")
        **kwargs: Additional parameters for strategy initialization

    Returns:
        Configured BaselineStrategy instance
    """
    type_mapping = {
        "volatility_bands": BaselineStrategyType.VOLATILITY_BANDS,
        "mean_reversion": BaselineStrategyType.MEAN_REVERSION,
        "ensemble": BaselineStrategyType.ENSEMBLE,
        "auto_select": BaselineStrategyType.AUTO_SELECT,
    }

    strategy_enum = type_mapping.get(strategy_type, BaselineStrategyType.ENSEMBLE)
    return BaselineStrategy(strategy_type=strategy_enum, **kwargs)
