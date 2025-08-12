"""
Test cases for data loader module

This module provides comprehensive tests for the data loading, validation,
and caching functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import List

from data.loader import (
    DataLoader, DataValidator, ParquetCache, CacheConfig,
    DataValidationError, create_data_loader
)
from data.schema import KlineData, TradeData, DepthData, TimeFrame, DepthLevel


class TestDataValidator:
    """Test data validation functionality"""
    
    def test_validate_empty_klines(self):
        """Test validation of empty klines list"""
        is_valid, errors = DataValidator.validate_klines([])
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_insufficient_klines(self):
        """Test validation fails with insufficient data"""
        kline = self._create_sample_kline()
        is_valid, errors = DataValidator.validate_klines([kline])
        assert is_valid is False
        assert "Insufficient data points" in errors[0]
    
    def test_validate_valid_klines(self):
        """Test validation passes with valid klines"""
        klines = [
            self._create_sample_kline(datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)),
            self._create_sample_kline(datetime(2023, 1, 1, 0, 5, tzinfo=timezone.utc))
        ]
        is_valid, errors = DataValidator.validate_klines(klines)
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_chronological_order_error(self):
        """Test validation catches chronological order errors"""
        klines = [
            self._create_sample_kline(datetime(2023, 1, 1, 0, 5, tzinfo=timezone.utc)),
            self._create_sample_kline(datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc))
        ]
        is_valid, errors = DataValidator.validate_klines(klines)
        assert is_valid is False
        assert any("chronological order" in error for error in errors)
    
    def test_validate_ohlc_relationships(self):
        """Test validation of OHLC price relationships"""
        # Create invalid OHLC (open > high)
        kline1 = self._create_sample_kline(
            datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc),
            open_price=Decimal('110'), high_price=Decimal('100')
        )
        kline2 = self._create_sample_kline(datetime(2023, 1, 1, 0, 5, tzinfo=timezone.utc))
        
        is_valid, errors = DataValidator.validate_klines([kline1, kline2])
        assert is_valid is False
        assert any("OHLC relationship" in error for error in errors)
    
    def test_validate_negative_values(self):
        """Test validation catches negative prices/volumes"""
        kline1 = self._create_sample_kline(
            datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc),
            volume=Decimal('-100')
        )
        kline2 = self._create_sample_kline(datetime(2023, 1, 1, 0, 5, tzinfo=timezone.utc))
        
        is_valid, errors = DataValidator.validate_klines([kline1, kline2])
        assert is_valid is False
        assert any("Negative price or volume" in error for error in errors)
    
    def test_validate_time_gaps(self):
        """Test validation catches time gaps in data"""
        klines = [
            self._create_sample_kline(datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)),
            self._create_sample_kline(datetime(2023, 1, 1, 0, 15, tzinfo=timezone.utc))  # 15min gap for 5min timeframe
        ]
        is_valid, errors = DataValidator.validate_klines(klines)
        assert is_valid is False
        assert any("Time gap detected" in error for error in errors)
    
    def test_validate_trades(self):
        """Test trade data validation"""
        trades = [
            self._create_sample_trade(datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)),
            self._create_sample_trade(datetime(2023, 1, 1, 0, 1, tzinfo=timezone.utc))
        ]
        is_valid, errors = DataValidator.validate_trades(trades)
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_depth_data(self):
        """Test order book depth validation"""
        depth = DepthData(
            symbol="BTCUSDC",
            last_update_id=123456,
            timestamp=datetime.now(timezone.utc),
            bids=[
                DepthLevel(price=Decimal('50000'), quantity=Decimal('1.0')),
                DepthLevel(price=Decimal('49999'), quantity=Decimal('0.5'))
            ],
            asks=[
                DepthLevel(price=Decimal('50001'), quantity=Decimal('1.0')),
                DepthLevel(price=Decimal('50002'), quantity=Decimal('0.5'))
            ]
        )
        is_valid, errors = DataValidator.validate_depth(depth)
        assert is_valid is True
        assert len(errors) == 0
    
    def _create_sample_kline(self, open_time: datetime = None,
                           open_price: Decimal = Decimal('100'),
                           high_price: Decimal = Decimal('105'),
                           low_price: Decimal = Decimal('95'),
                           close_price: Decimal = Decimal('102'),
                           volume: Decimal = Decimal('1000')) -> KlineData:
        """Create a sample kline for testing"""
        if open_time is None:
            open_time = datetime.now(timezone.utc)
        
        return KlineData(
            symbol="BTCUSDC",
            open_time=open_time,
            close_time=open_time + timedelta(minutes=5),
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            volume=volume,
            quote_volume=volume * close_price,
            trades_count=100,
            taker_buy_volume=volume * Decimal('0.6'),
            taker_buy_quote_volume=volume * close_price * Decimal('0.6'),
            timeframe=TimeFrame.M5
        )
    
    def _create_sample_trade(self, timestamp: datetime) -> TradeData:
        """Create a sample trade for testing"""
        return TradeData(
            symbol="BTCUSDC",
            trade_id=12345,
            price=Decimal('50000'),
            quantity=Decimal('0.01'),
            quote_quantity=Decimal('500'),
            timestamp=timestamp,
            is_buyer_maker=True
        )


class TestParquetCache:
    """Test Parquet caching functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_config = CacheConfig(
            cache_dir=self.temp_dir,
            max_cache_age_hours=1
        )
        self.cache = ParquetCache(self.cache_config)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_klines(self):
        """Test saving and loading klines to/from cache"""
        klines = [
            self._create_sample_kline(datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)),
            self._create_sample_kline(datetime(2023, 1, 1, 0, 5, tzinfo=timezone.utc))
        ]
        
        # Save to cache
        cache_path = self.cache.save_klines(klines, "BTCUSDC", TimeFrame.M5)
        assert cache_path is not None
        assert cache_path.exists()
        
        # Load from cache
        loaded_klines = self.cache.load_klines("BTCUSDC", TimeFrame.M5)
        assert loaded_klines is not None
        assert len(loaded_klines) == 2
        assert loaded_klines[0].symbol == "BTCUSDC"
        assert loaded_klines[0].timeframe == TimeFrame.M5
    
    def test_cache_expiry(self):
        """Test cache expiry functionality"""
        # Create cache config with very short expiry
        short_cache_config = CacheConfig(
            cache_dir=self.temp_dir,
            max_cache_age_hours=0  # Immediate expiry
        )
        cache = ParquetCache(short_cache_config)
        
        klines = [self._create_sample_kline()]
        cache.save_klines(klines, "BTCUSDC", TimeFrame.M5)
        
        # Should return None due to immediate expiry
        loaded_klines = cache.load_klines("BTCUSDC", TimeFrame.M5)
        assert loaded_klines is None
    
    def test_clear_cache(self):
        """Test cache clearing functionality"""
        klines = [self._create_sample_kline()]
        self.cache.save_klines(klines, "BTCUSDC", TimeFrame.M5)
        
        # Verify file exists
        cache_files = list(Path(self.temp_dir).glob("*.parquet"))
        assert len(cache_files) > 0
        
        # Clear cache
        self.cache.clear_cache(older_than_hours=0)
        
        # Verify files are deleted
        cache_files_after = list(Path(self.temp_dir).glob("*.parquet"))
        assert len(cache_files_after) == 0
    
    def _create_sample_kline(self, open_time: datetime = None) -> KlineData:
        """Create a sample kline for testing"""
        if open_time is None:
            open_time = datetime.now(timezone.utc)
        
        return KlineData(
            symbol="BTCUSDC",
            open_time=open_time,
            close_time=open_time + timedelta(minutes=5),
            open_price=Decimal('100'),
            high_price=Decimal('105'),
            low_price=Decimal('95'),
            close_price=Decimal('102'),
            volume=Decimal('1000'),
            quote_volume=Decimal('102000'),
            trades_count=100,
            taker_buy_volume=Decimal('600'),
            taker_buy_quote_volume=Decimal('61200'),
            timeframe=TimeFrame.M5
        )


class TestDataLoader:
    """Test main DataLoader functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_config = CacheConfig(cache_dir=self.temp_dir)
        self.loader = DataLoader(self.cache_config)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_loader_initialization(self):
        """Test data loader initialization"""
        assert self.loader.cache_config is not None
        assert self.loader.cache is not None
        assert self.loader.enable_validation is True
        assert self.loader.validator is not None
    
    def test_get_cache_stats(self):
        """Test cache statistics retrieval"""
        stats = self.loader.get_cache_stats()
        assert "cache_dir" in stats
        assert "file_count" in stats
        assert "total_size_mb" in stats
        assert stats["file_count"] == 0  # No cache files initially
    
    def test_create_data_loader_function(self):
        """Test data loader factory function"""
        loader = create_data_loader(
            cache_dir=self.temp_dir,
            max_cache_age_hours=2,
            use_unified_config=False
        )
        assert loader is not None
        assert loader.cache_config.cache_dir == Path(self.temp_dir)
        assert loader.cache_config.max_cache_age_hours == 2


if __name__ == "__main__":
    pytest.main([__file__])
