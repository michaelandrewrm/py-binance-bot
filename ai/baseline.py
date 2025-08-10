"""
Baseline - ATR/vol bands heuristic (baseline)

This module implements baseline trading strategies using traditional
technical indicators like ATR and volatility bands.
"""

from typing import List, Dict, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timezone
import numpy as np

from ..data.schema import KlineData, FeatureData, ModelPrediction

class TechnicalIndicators:
    """Collection of technical indicators for baseline strategies"""
    
    @staticmethod
    def sma(prices: List[Decimal], period: int) -> List[Optional[Decimal]]:
        """Simple Moving Average"""
        if len(prices) < period:
            return [None] * len(prices)
        
        result = [None] * (period - 1)
        for i in range(period - 1, len(prices)):
            avg = sum(prices[i - period + 1:i + 1]) / period
            result.append(avg)
        
        return result
    
    @staticmethod
    def ema(prices: List[Decimal], period: int, alpha: Optional[Decimal] = None) -> List[Optional[Decimal]]:
        """Exponential Moving Average"""
        if not prices:
            return []
        
        if alpha is None:
            alpha = Decimal('2') / (period + 1)
        
        result = [None] * len(prices)
        result[0] = prices[0]
        
        for i in range(1, len(prices)):
            result[i] = alpha * prices[i] + (1 - alpha) * result[i-1]
        
        return result
    
    @staticmethod
    def atr(highs: List[Decimal], lows: List[Decimal], closes: List[Decimal], 
            period: int = 14) -> List[Optional[Decimal]]:
        """Average True Range"""
        if len(highs) < 2 or len(highs) != len(lows) or len(highs) != len(closes):
            return [None] * len(highs)
        
        true_ranges = []
        
        # First TR is just high - low
        true_ranges.append(highs[0] - lows[0])
        
        # Calculate True Range for each period
        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]  # High - Low
            tr2 = abs(highs[i] - closes[i-1])  # High - Previous Close
            tr3 = abs(lows[i] - closes[i-1])   # Low - Previous Close
            
            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)
        
        # Calculate ATR using SMA of True Ranges
        return TechnicalIndicators.sma(true_ranges, period)
    
    @staticmethod
    def bollinger_bands(prices: List[Decimal], period: int = 20, 
                       std_dev: Decimal = Decimal('2')) -> Tuple[List[Optional[Decimal]], 
                                                                List[Optional[Decimal]], 
                                                                List[Optional[Decimal]]]:
        """Bollinger Bands"""
        if len(prices) < period:
            return ([None] * len(prices), [None] * len(prices), [None] * len(prices))
        
        sma_values = TechnicalIndicators.sma(prices, period)
        upper_bands = []
        lower_bands = []
        
        for i in range(len(prices)):
            if i < period - 1 or sma_values[i] is None:
                upper_bands.append(None)
                lower_bands.append(None)
            else:
                # Calculate standard deviation for the period
                period_prices = prices[i - period + 1:i + 1]
                mean = sma_values[i]
                variance = sum((price - mean) ** 2 for price in period_prices) / period
                std = variance ** Decimal('0.5')
                
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
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(Decimal('0'))
            else:
                gains.append(Decimal('0'))
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
                result[i + 1] = Decimal('100')
            else:
                rs = avg_gain / avg_loss
                rsi = Decimal('100') - (Decimal('100') / (Decimal('1') + rs))
                result[i + 1] = rsi
            
            # Update averages for next iteration
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        return result

class VolatilityBandStrategy:
    """
    Baseline strategy using volatility bands for entry/exit signals
    """
    
    def __init__(self, atr_period: int = 14, atr_multiplier: Decimal = Decimal('2'),
                 sma_period: int = 20, min_volatility: Decimal = Decimal('0.01')):
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
    
    def _evaluate_position(self, kline: KlineData, price: Decimal, sma: Decimal,
                          upper_band: Decimal, lower_band: Decimal, atr: Decimal) -> Optional[Dict]:
        """Evaluate current position and generate signals"""
        
        signal = None
        
        if self.position == 0:  # No position
            # Look for entry signals
            if price > upper_band and price > sma:
                # Long entry signal
                signal = {
                    'timestamp': kline.open_time,
                    'symbol': kline.symbol,
                    'action': 'BUY',
                    'price': price,
                    'size': Decimal('1.0'),  # Full position
                    'reason': 'breakout_above_upper_band',
                    'stop_loss': price - atr * self.atr_multiplier,
                    'take_profit': price + atr * self.atr_multiplier * 2
                }
                self.position = 1
                self.entry_price = price
                self.stop_loss = signal['stop_loss']
                self.take_profit = signal['take_profit']
                
            elif price < lower_band and price < sma:
                # Short entry signal
                signal = {
                    'timestamp': kline.open_time,
                    'symbol': kline.symbol,
                    'action': 'SELL',
                    'price': price,
                    'size': Decimal('1.0'),
                    'reason': 'breakdown_below_lower_band',
                    'stop_loss': price + atr * self.atr_multiplier,
                    'take_profit': price - atr * self.atr_multiplier * 2
                }
                self.position = -1
                self.entry_price = price
                self.stop_loss = signal['stop_loss']
                self.take_profit = signal['take_profit']
        
        elif self.position == 1:  # Long position
            # Check exit conditions
            if price <= self.stop_loss or price >= self.take_profit or price < sma:
                reason = 'stop_loss' if price <= self.stop_loss else \
                        'take_profit' if price >= self.take_profit else \
                        'trend_reversal'
                
                signal = {
                    'timestamp': kline.open_time,
                    'symbol': kline.symbol,
                    'action': 'SELL',
                    'price': price,
                    'size': Decimal('1.0'),
                    'reason': reason,
                    'pnl': price - self.entry_price
                }
                self._reset_position()
        
        elif self.position == -1:  # Short position
            # Check exit conditions
            if price >= self.stop_loss or price <= self.take_profit or price > sma:
                reason = 'stop_loss' if price >= self.stop_loss else \
                        'take_profit' if price <= self.take_profit else \
                        'trend_reversal'
                
                signal = {
                    'timestamp': kline.open_time,
                    'symbol': kline.symbol,
                    'action': 'BUY',
                    'price': price,
                    'size': Decimal('1.0'),
                    'reason': reason,
                    'pnl': self.entry_price - price
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
    
    def __init__(self, bb_period: int = 20, bb_std: Decimal = Decimal('2'),
                 rsi_period: int = 14, rsi_oversold: Decimal = Decimal('30'),
                 rsi_overbought: Decimal = Decimal('70')):
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
            if (sma[i] is None or upper_bb[i] is None or 
                lower_bb[i] is None or rsi_values[i] is None):
                continue
            
            current_price = closes[i]
            signal = self._evaluate_mean_reversion(
                klines[i], current_price, sma[i], upper_bb[i], lower_bb[i], rsi_values[i]
            )
            
            if signal:
                signals.append(signal)
        
        return signals
    
    def _evaluate_mean_reversion(self, kline: KlineData, price: Decimal,
                               sma: Decimal, upper_bb: Decimal, lower_bb: Decimal,
                               rsi: Decimal) -> Optional[Dict]:
        """Evaluate mean reversion opportunities"""
        
        signal = None
        
        if self.position == 0:
            # Look for oversold conditions
            if price <= lower_bb and rsi <= self.rsi_oversold:
                signal = {
                    'timestamp': kline.open_time,
                    'symbol': kline.symbol,
                    'action': 'BUY',
                    'price': price,
                    'size': Decimal('1.0'),
                    'reason': 'oversold_mean_reversion',
                    'target': sma
                }
                self.position = 1
                self.entry_price = price
            
            # Look for overbought conditions
            elif price >= upper_bb and rsi >= self.rsi_overbought:
                signal = {
                    'timestamp': kline.open_time,
                    'symbol': kline.symbol,
                    'action': 'SELL',
                    'price': price,
                    'size': Decimal('1.0'),
                    'reason': 'overbought_mean_reversion',
                    'target': sma
                }
                self.position = -1
                self.entry_price = price
        
        elif self.position == 1:
            # Exit long position when price reaches SMA or RSI becomes overbought
            if price >= sma or rsi >= self.rsi_overbought:
                signal = {
                    'timestamp': kline.open_time,
                    'symbol': kline.symbol,
                    'action': 'SELL',
                    'price': price,
                    'size': Decimal('1.0'),
                    'reason': 'mean_reversion_target',
                    'pnl': price - self.entry_price
                }
                self._reset_position()
        
        elif self.position == -1:
            # Exit short position when price reaches SMA or RSI becomes oversold
            if price <= sma or rsi <= self.rsi_oversold:
                signal = {
                    'timestamp': kline.open_time,
                    'symbol': kline.symbol,
                    'action': 'BUY',
                    'price': price,
                    'size': Decimal('1.0'),
                    'reason': 'mean_reversion_target',
                    'pnl': self.entry_price - price
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
        results['volatility_bands'] = {
            'signals': vol_signals,
            'total_signals': len(vol_signals),
            'metrics': self._calculate_signal_metrics(vol_signals)
        }
        
        # Mean reversion strategy
        mr_signals = self.mean_reversion_strategy.generate_signals(klines)
        results['mean_reversion'] = {
            'signals': mr_signals,
            'total_signals': len(mr_signals),
            'metrics': self._calculate_signal_metrics(mr_signals)
        }
        
        return results
    
    def _calculate_signal_metrics(self, signals: List[Dict]) -> Dict:
        """Calculate basic metrics for signals"""
        if not signals:
            return {'total_pnl': Decimal('0'), 'win_rate': Decimal('0'), 'trades': 0}
        
        trade_pnls = [signal.get('pnl', Decimal('0')) for signal in signals if signal.get('pnl')]
        
        if not trade_pnls:
            return {'total_pnl': Decimal('0'), 'win_rate': Decimal('0'), 'trades': 0}
        
        total_pnl = sum(trade_pnls)
        winning_trades = sum(1 for pnl in trade_pnls if pnl > 0)
        win_rate = Decimal(str(winning_trades / len(trade_pnls))) if trade_pnls else Decimal('0')
        
        return {
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'trades': len(trade_pnls),
            'avg_pnl': total_pnl / len(trade_pnls) if trade_pnls else Decimal('0')
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
            'close_price': float(closes[i]),
            'volume': float(volumes[i]),
            'price_change': float((closes[i] - closes[i-1]) / closes[i-1]),
            'volume_change': float((volumes[i] - volumes[i-1]) / volumes[i-1]) if volumes[i-1] > 0 else 0,
        }
        
        # Add technical indicators
        if sma_20[i] is not None:
            feature_dict['sma_20'] = float(sma_20[i])
            feature_dict['price_vs_sma20'] = float(closes[i] / sma_20[i] - 1)
        
        if sma_50[i] is not None:
            feature_dict['sma_50'] = float(sma_50[i])
            feature_dict['sma20_vs_sma50'] = float(sma_20[i] / sma_50[i] - 1) if sma_20[i] and sma_50[i] else 0
        
        if ema_12[i] is not None and ema_26[i] is not None:
            feature_dict['macd'] = float(ema_12[i] - ema_26[i])
            feature_dict['macd_signal'] = feature_dict['macd']  # Simplified
        
        if atr_14[i] is not None:
            feature_dict['atr_14'] = float(atr_14[i])
            feature_dict['volatility'] = float(atr_14[i] / closes[i])
        
        if rsi_14[i] is not None:
            feature_dict['rsi_14'] = float(rsi_14[i])
        
        features.append(FeatureData(
            symbol=klines[i].symbol,
            timestamp=klines[i].open_time,
            features=feature_dict
        ))
    
    return features

def run_baseline_benchmark(klines: List[KlineData]) -> Dict:
    """Run all baseline strategies as benchmark"""
    evaluator = BaselineEvaluator()
    return evaluator.evaluate_all_strategies(klines)
