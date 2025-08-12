"""
Test cases for ai/baseline.py module

This module provides comprehensive tests for the baseline trading strategies,
technical indicators, and evaluation functionality.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

from ai.baseline import (
    TechnicalIndicators,
    VolatilityBandStrategy,
    MeanReversionStrategy,
    BaselineEvaluator,
    create_baseline_features,
    run_baseline_benchmark
)
from data.schema import KlineData, FeatureData, TimeFrame


def create_test_kline(symbol: str, timestamp: datetime, close_price: float = 100.0, volume: float = 1000.0) -> KlineData:
    """Helper function to create test KlineData with proper schema."""
    close_dec = Decimal(str(close_price))
    volume_dec = Decimal(str(volume))
    return KlineData(
        symbol=symbol,
        open_time=timestamp,
        close_time=timestamp,
        open_price=close_dec,
        high_price=close_dec * Decimal("1.01"),
        low_price=close_dec * Decimal("0.99"),
        close_price=close_dec,
        volume=volume_dec,
        quote_volume=volume_dec * close_dec,
        trades_count=50,
        taker_buy_volume=volume_dec * Decimal("0.5"),
        taker_buy_quote_volume=volume_dec * close_dec * Decimal("0.5"),
        timeframe=TimeFrame.M1
    )


class TestTechnicalIndicators:
    """Test technical indicators functionality"""
    
    def setup_method(self):
        """Set up test data"""
        self.sample_prices = [
            Decimal('100'), Decimal('102'), Decimal('101'), Decimal('105'), Decimal('103'),
            Decimal('107'), Decimal('106'), Decimal('108'), Decimal('110'), Decimal('109'),
            Decimal('111'), Decimal('113'), Decimal('112'), Decimal('115'), Decimal('114')
        ]
        
    def test_sma_calculation(self):
        """Test Simple Moving Average calculation"""
        # Test with period 3
        result = TechnicalIndicators.sma(self.sample_prices[:5], 3)
        
        # First two values should be None
        assert result[0] is None
        assert result[1] is None
        
        # Third value should be average of first 3 prices
        expected = (Decimal('100') + Decimal('102') + Decimal('101')) / 3
        assert result[2] == expected
        
        # Test with insufficient data
        short_prices = [Decimal('100'), Decimal('102')]
        result_short = TechnicalIndicators.sma(short_prices, 5)
        assert all(x is None for x in result_short)
    
    def test_sma_empty_list(self):
        """Test SMA with empty price list"""
        result = TechnicalIndicators.sma([], 5)
        assert result == []
    
    def test_ema_calculation(self):
        """Test Exponential Moving Average calculation"""
        prices = [Decimal('100'), Decimal('102'), Decimal('101'), Decimal('105')]
        result = TechnicalIndicators.ema(prices, 3)
        
        # First value should equal first price
        assert result[0] == prices[0]
        
        # Check that all values are calculated
        assert all(x is not None for x in result)
        
        # Test with custom alpha
        alpha = Decimal('0.5')
        result_custom = TechnicalIndicators.ema(prices, 3, alpha)
        assert result_custom[0] == prices[0]
    
    def test_ema_empty_list(self):
        """Test EMA with empty price list"""
        result = TechnicalIndicators.ema([], 5)
        assert result == []
    
    def test_atr_calculation(self):
        """Test Average True Range calculation"""
        highs = [Decimal('105'), Decimal('110'), Decimal('108'), Decimal('112')]
        lows = [Decimal('95'), Decimal('100'), Decimal('98'), Decimal('102')]
        closes = [Decimal('100'), Decimal('105'), Decimal('103'), Decimal('107')]
        
        result = TechnicalIndicators.atr(highs, lows, closes, 3)
        
        # Should have same length as input
        assert len(result) == len(highs)
        
        # First two values should be None (period - 1)
        assert result[0] is None
        assert result[1] is None
        
        # Third value should be calculated
        assert result[2] is not None
        assert result[2] > 0
    
    def test_atr_insufficient_data(self):
        """Test ATR with insufficient data"""
        highs = [Decimal('105')]
        lows = [Decimal('95')]
        closes = [Decimal('100')]
        
        result = TechnicalIndicators.atr(highs, lows, closes, 3)
        assert result == [None]
    
    def test_atr_mismatched_lengths(self):
        """Test ATR with mismatched array lengths"""
        highs = [Decimal('105'), Decimal('110')]
        lows = [Decimal('95')]  # Different length
        closes = [Decimal('100'), Decimal('105')]
        
        result = TechnicalIndicators.atr(highs, lows, closes, 3)
        assert all(x is None for x in result)
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation"""
        prices = self.sample_prices[:10]
        sma, upper, lower = TechnicalIndicators.bollinger_bands(prices, 5, Decimal('2'))
        
        # Check lengths
        assert len(sma) == len(prices)
        assert len(upper) == len(prices)
        assert len(lower) == len(prices)
        
        # First 4 values should be None
        for i in range(4):
            assert sma[i] is None
            assert upper[i] is None
            assert lower[i] is None
        
        # Fifth value and onwards should be calculated
        assert sma[4] is not None
        assert upper[4] is not None
        assert lower[4] is not None
        
        # Upper should be greater than SMA, lower should be less
        assert upper[4] > sma[4]
        assert lower[4] < sma[4]
    
    def test_bollinger_bands_insufficient_data(self):
        """Test Bollinger Bands with insufficient data"""
        prices = [Decimal('100'), Decimal('102')]
        sma, upper, lower = TechnicalIndicators.bollinger_bands(prices, 5)
        
        assert all(x is None for x in sma)
        assert all(x is None for x in upper)
        assert all(x is None for x in lower)
    
    def test_rsi_calculation(self):
        """Test Relative Strength Index calculation"""
        # Create trending up prices
        trending_prices = [Decimal(str(100 + i)) for i in range(20)]
        result = TechnicalIndicators.rsi(trending_prices, 14)
        
        # Should have same length as prices
        assert len(result) == len(trending_prices)
        
        # First 14 values should be None
        for i in range(14):
            assert result[i] is None
        
        # RSI values should be between 0 and 100
        for i in range(14, len(result)):
            if result[i] is not None:
                assert 0 <= result[i] <= 100
    
    def test_rsi_insufficient_data(self):
        """Test RSI with insufficient data"""
        prices = [Decimal('100'), Decimal('102')]
        result = TechnicalIndicators.rsi(prices, 14)
        assert all(x is None for x in result)
    
    def test_rsi_no_losses(self):
        """Test RSI when there are no losses (all gains)"""
        # All increasing prices
        prices = [Decimal(str(100 + i * 2)) for i in range(16)]
        result = TechnicalIndicators.rsi(prices, 14)
        
        # Should handle division by zero (no losses)
        # The RSI should be 100 when there are only gains
        assert result[15] == Decimal('100')  # Check index 15 instead of 14


class TestVolatilityBandStrategy:
    """Test volatility band strategy"""
    
    def setup_method(self):
        """Set up test data"""
        self.strategy = VolatilityBandStrategy(
            atr_period=5,
            atr_multiplier=Decimal('2'),
            sma_period=5,
            min_volatility=Decimal('0.01')
        )
        
        # Create sample klines with trending data
        self.sample_klines = []
        base_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
        
        for i in range(10):
            price = Decimal(str(100 + i * 2))  # Trending up
            timestamp = base_time + timedelta(minutes=i)
            kline = create_test_kline("BTCUSDC", timestamp, price)
            self.sample_klines.append(kline)
    
    def test_strategy_initialization(self):
        """Test strategy initialization"""
        assert self.strategy.atr_period == 5
        assert self.strategy.atr_multiplier == Decimal('2')
        assert self.strategy.sma_period == 5
        assert self.strategy.position == 0
        assert self.strategy.entry_price is None
    
    def test_generate_signals_insufficient_data(self):
        """Test signal generation with insufficient data"""
        short_klines = self.sample_klines[:3]
        signals = self.strategy.generate_signals(short_klines)
        assert signals == []
    
    def test_generate_signals_sufficient_data(self):
        """Test signal generation with sufficient data"""
        signals = self.strategy.generate_signals(self.sample_klines)
        
        # Should be able to generate signals
        assert isinstance(signals, list)
        
        # Each signal should have required fields
        for signal in signals:
            assert 'timestamp' in signal
            assert 'symbol' in signal
            assert 'action' in signal
            assert 'price' in signal
            assert 'size' in signal
            assert 'reason' in signal
    
    def test_position_reset(self):
        """Test position reset functionality"""
        self.strategy.position = 1
        self.strategy.entry_price = Decimal('100')
        self.strategy.stop_loss = Decimal('95')
        self.strategy.take_profit = Decimal('105')
        
        self.strategy._reset_position()
        
        assert self.strategy.position == 0
        assert self.strategy.entry_price is None
        assert self.strategy.stop_loss is None
        assert self.strategy.take_profit is None
    
    def test_evaluate_position_no_position(self):
        """Test position evaluation when flat"""
        kline = self.sample_klines[0]
        price = Decimal('120')  # High price to trigger buy
        sma = Decimal('100')
        upper_band = Decimal('110')
        lower_band = Decimal('90')
        atr = Decimal('5')
        
        signal = self.strategy._evaluate_position(
            kline, price, sma, upper_band, lower_band, atr
        )
        
        # Should generate buy signal for breakout above upper band
        if signal:
            assert signal['action'] == 'BUY'
            assert signal['reason'] == 'breakout_above_upper_band'
            assert self.strategy.position == 1


class TestMeanReversionStrategy:
    """Test mean reversion strategy"""
    
    def setup_method(self):
        """Set up test data"""
        self.strategy = MeanReversionStrategy(
            bb_period=5,
            bb_std=Decimal('2'),
            rsi_period=5,
            rsi_oversold=Decimal('30'),
            rsi_overbought=Decimal('70')
        )
        
        # Create sample klines with oscillating data
        self.sample_klines = []
        base_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
        
        prices = [100, 95, 105, 90, 110, 85, 115, 80, 120, 75]  # Oscillating
        
        for i, price in enumerate(prices):
            timestamp = base_time + timedelta(minutes=i)
            kline = create_test_kline("BTCUSDC", timestamp, Decimal(str(price)))
            self.sample_klines.append(kline)
    
    def test_strategy_initialization(self):
        """Test strategy initialization"""
        assert self.strategy.bb_period == 5
        assert self.strategy.bb_std == Decimal('2')
        assert self.strategy.rsi_period == 5
        assert self.strategy.position == 0
        assert self.strategy.entry_price is None
    
    def test_generate_signals_insufficient_data(self):
        """Test signal generation with insufficient data"""
        short_klines = self.sample_klines[:3]
        signals = self.strategy.generate_signals(short_klines)
        assert signals == []
    
    def test_generate_signals_sufficient_data(self):
        """Test signal generation with sufficient data"""
        signals = self.strategy.generate_signals(self.sample_klines)
        
        # Should be able to generate signals
        assert isinstance(signals, list)
        
        # Each signal should have required fields
        for signal in signals:
            assert 'timestamp' in signal
            assert 'symbol' in signal
            assert 'action' in signal
            assert 'price' in signal
            assert 'size' in signal
            assert 'reason' in signal
    
    def test_position_reset(self):
        """Test position reset functionality"""
        self.strategy.position = 1
        self.strategy.entry_price = Decimal('100')
        
        self.strategy._reset_position()
        
        assert self.strategy.position == 0
        assert self.strategy.entry_price is None
    
    def test_evaluate_mean_reversion_oversold(self):
        """Test mean reversion evaluation for oversold conditions"""
        kline = self.sample_klines[0]
        price = Decimal('80')  # Low price
        sma = Decimal('100')
        upper_bb = Decimal('110')
        lower_bb = Decimal('90')
        rsi = Decimal('25')  # Oversold
        
        signal = self.strategy._evaluate_mean_reversion(
            kline, price, sma, upper_bb, lower_bb, rsi
        )
        
        # Should generate buy signal for oversold condition
        if signal:
            assert signal['action'] == 'BUY'
            assert signal['reason'] == 'oversold_mean_reversion'
            assert self.strategy.position == 1


class TestBaselineEvaluator:
    """Test baseline evaluator"""
    
    def setup_method(self):
        """Set up test data"""
        self.evaluator = BaselineEvaluator()
        
        # Create comprehensive sample data
        self.sample_klines = []
        base_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
        
        # Create 30 klines with mixed patterns
        for i in range(30):
            # Create some volatility
            base_price = 100 + (i % 10) * 2 - 5
            price = Decimal(str(base_price))
            timestamp = base_time + timedelta(minutes=i)
            kline = create_test_kline("BTCUSDC", timestamp, price)
            self.sample_klines.append(kline)
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization"""
        assert self.evaluator.volatility_strategy is not None
        assert self.evaluator.mean_reversion_strategy is not None
    
    def test_evaluate_all_strategies(self):
        """Test evaluation of all strategies"""
        results = self.evaluator.evaluate_all_strategies(self.sample_klines)
        
        # Should have results for both strategies
        assert 'volatility_bands' in results
        assert 'mean_reversion' in results
        
        # Each strategy result should have required fields
        for strategy_name in ['volatility_bands', 'mean_reversion']:
            strategy_result = results[strategy_name]
            assert 'signals' in strategy_result
            assert 'total_signals' in strategy_result
            assert 'metrics' in strategy_result
            
            # Metrics should have required fields
            metrics = strategy_result['metrics']
            assert 'total_pnl' in metrics
            assert 'win_rate' in metrics
            assert 'trades' in metrics
    
    def test_calculate_signal_metrics_empty(self):
        """Test metrics calculation with empty signals"""
        metrics = self.evaluator._calculate_signal_metrics([])
        
        assert metrics['total_pnl'] == Decimal('0')
        assert metrics['win_rate'] == Decimal('0')
        assert metrics['trades'] == 0
    
    def test_calculate_signal_metrics_with_pnl(self):
        """Test metrics calculation with PnL data"""
        signals = [
            {'pnl': Decimal('10')},
            {'pnl': Decimal('-5')},
            {'pnl': Decimal('15')},
            {'pnl': Decimal('-3')},
        ]
        
        metrics = self.evaluator._calculate_signal_metrics(signals)
        
        assert metrics['total_pnl'] == Decimal('17')
        assert metrics['trades'] == 4
        assert metrics['win_rate'] == Decimal('0.5')  # 2 out of 4 winning
        assert 'avg_pnl' in metrics
    
    def test_calculate_signal_metrics_no_pnl(self):
        """Test metrics calculation with signals without PnL"""
        signals = [
            {'action': 'BUY'},
            {'action': 'SELL'},
        ]
        
        metrics = self.evaluator._calculate_signal_metrics(signals)
        
        assert metrics['total_pnl'] == Decimal('0')
        assert metrics['win_rate'] == Decimal('0')
        assert metrics['trades'] == 0


class TestUtilityFunctions:
    """Test utility functions"""
    
    def setup_method(self):
        """Set up test data"""
        # Create comprehensive sample data for feature extraction
        self.sample_klines = []
        base_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
        
        # Create 60 klines for complete feature calculation
        for i in range(60):
            price = Decimal(str(100 + i * 0.5))  # Gradual trend
            volume = Decimal(str(1000 + i * 10))
            timestamp = base_time + timedelta(minutes=i)
            kline = create_test_kline("BTCUSDC", timestamp, price, volume)
            self.sample_klines.append(kline)
    
    def test_create_baseline_features_insufficient_data(self):
        """Test feature creation with insufficient data"""
        short_klines = self.sample_klines[:40]
        features = create_baseline_features(short_klines)
        assert features == []
    
    def test_create_baseline_features_sufficient_data(self):
        """Test feature creation with sufficient data"""
        features = create_baseline_features(self.sample_klines)
        
        # Should have features for klines after index 49
        expected_count = len(self.sample_klines) - 49
        assert len(features) == expected_count
        
        # Each feature should be FeatureData instance
        for feature in features:
            assert isinstance(feature, FeatureData)
            assert feature.symbol == "BTCUSDC"
            assert feature.timestamp is not None
            assert isinstance(feature.features, dict)
        
        # Check that required features are present
        if features:
            first_feature = features[0].features
            required_features = [
                'close_price', 'volume', 'price_change', 'volume_change'
            ]
            for req_feature in required_features:
                assert req_feature in first_feature
    
    def test_run_baseline_benchmark(self):
        """Test running baseline benchmark"""
        results = run_baseline_benchmark(self.sample_klines)
        
        # Should return dictionary with strategy results
        assert isinstance(results, dict)
        assert 'volatility_bands' in results
        assert 'mean_reversion' in results
        
        # Each strategy should have the expected structure
        for strategy_name in ['volatility_bands', 'mean_reversion']:
            strategy_result = results[strategy_name]
            assert 'signals' in strategy_result
            assert 'total_signals' in strategy_result
            assert 'metrics' in strategy_result
    
    def test_feature_data_structure(self):
        """Test the structure of created features"""
        features = create_baseline_features(self.sample_klines)
        
        if features:
            feature = features[0]
            feature_dict = feature.features
            
            # Check data types
            assert isinstance(feature_dict['close_price'], float)
            assert isinstance(feature_dict['volume'], float)
            assert isinstance(feature_dict['price_change'], float)
            assert isinstance(feature_dict['volume_change'], float)
            
            # Check technical indicators if present
            if 'sma_20' in feature_dict:
                assert isinstance(feature_dict['sma_20'], float)
                assert isinstance(feature_dict['price_vs_sma20'], float)
            
            if 'rsi_14' in feature_dict:
                assert 0 <= feature_dict['rsi_14'] <= 100


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_technical_indicators_with_zero_values(self):
        """Test technical indicators with zero values"""
        prices_with_zeros = [Decimal('0'), Decimal('100'), Decimal('0'), Decimal('200')]
        
        # SMA should handle zeros
        sma_result = TechnicalIndicators.sma(prices_with_zeros, 2)
        assert sma_result[1] == Decimal('50')  # (0 + 100) / 2
        
        # EMA should handle zeros
        ema_result = TechnicalIndicators.ema(prices_with_zeros, 2)
        assert ema_result[0] == Decimal('0')
    
    def test_strategy_with_identical_prices(self):
        """Test strategies with identical prices (no volatility)"""
        # Create klines with identical prices
        identical_klines = []
        base_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
        
        for i in range(20):
            price = Decimal('100')  # All same price
            timestamp = base_time + timedelta(minutes=i)
            kline = create_test_kline("BTCUSDC", timestamp, price)
            identical_klines.append(kline)
        
        # Test volatility strategy
        vol_strategy = VolatilityBandStrategy(min_volatility=Decimal('0.01'))
        vol_signals = vol_strategy.generate_signals(identical_klines)
        
        # Should generate no signals due to low volatility
        assert isinstance(vol_signals, list)
        
        # Test mean reversion strategy
        mr_strategy = MeanReversionStrategy()
        mr_signals = mr_strategy.generate_signals(identical_klines)
        
        # Should handle identical prices gracefully
        assert isinstance(mr_signals, list)
    
    def test_single_kline_input(self):
        """Test strategies with single kline input"""
        timestamp = datetime(2023, 1, 1, tzinfo=timezone.utc)
        single_kline = [create_test_kline("BTCUSDC", timestamp, Decimal('100'))]
        
        vol_strategy = VolatilityBandStrategy()
        vol_signals = vol_strategy.generate_signals(single_kline)
        assert vol_signals == []
        
        mr_strategy = MeanReversionStrategy()
        mr_signals = mr_strategy.generate_signals(single_kline)
        assert mr_signals == []


if __name__ == "__main__":
    pytest.main([__file__])
