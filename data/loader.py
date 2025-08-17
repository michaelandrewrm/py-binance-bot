"""
Loader - Klines/trades/depth; validation & parquet cache

This module handles data loading, validation, and caching using Parquet
format for efficient storage and retrieval.
"""

from typing import List, Dict, Optional, Tuple, Union
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from pathlib import Path
import asyncio
import logging
import hashlib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dataclasses import asdict

from .schema import (
    KlineData,
    TradeData,
    TickerData,
    DepthData,
    TimeFrame,
    klines_from_binance,
    trades_from_binance,
    depth_from_binance,
)

# Import unified configuration
try:
    from config import get_settings

    _unified_config_available = True
except ImportError:
    _unified_config_available = False

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Data validation error"""

    pass


class CacheConfig:
    """Data cache configuration"""

    def __init__(
        self,
        cache_dir: str = "data_cache",
        max_cache_age_hours: int = 24,
        compression: str = "snappy",
        validate_on_load: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.max_cache_age_hours = max_cache_age_hours
        self.compression = compression
        self.validate_on_load = validate_on_load

        # Create cache directory
        self.cache_dir.mkdir(exist_ok=True)

    @classmethod
    def from_unified_config(cls, config=None):
        """Create CacheConfig from unified configuration"""
        if not _unified_config_available:
            raise ImportError("Unified config system not available")

        if config is None:
            config = get_settings()

        return cls(
            cache_dir=str(config.get_cache_dir()),
            max_cache_age_hours=config.cache.max_age_hours,
            compression="snappy",  # Could be configurable
            validate_on_load=True,  # Could be configurable
        )


class DataValidator:
    """Validates market data for consistency and quality"""

    @staticmethod
    def validate_klines(klines: List[KlineData]) -> Tuple[bool, List[str]]:
        """Validate kline data for consistency and quality.

        Args:
            klines: List of KlineData objects to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if not klines:
            return True, []

        # Check for missing data
        if len(klines) < 2:
            errors.append("Insufficient data points")
            return False, errors

        # Vectorized chronological and price validation
        prev_kline = klines[0]
        for i, kline in enumerate(klines[1:], start=1):
            # Check chronological order
            if kline.open_time <= prev_kline.open_time:
                errors.append(f"Klines not in chronological order at index {i}")

            # Check OHLC relationships efficiently
            if not (kline.low_price <= kline.open_price <= kline.high_price):
                errors.append(
                    f"Invalid OHLC relationship at index {i}: open {kline.open_price} not within range [{kline.low_price}, {kline.high_price}]"
                )

            if not (kline.low_price <= kline.close_price <= kline.high_price):
                errors.append(
                    f"Invalid OHLC relationship at index {i}: close {kline.close_price} not within range [{kline.low_price}, {kline.high_price}]"
                )

            # Check for negative values
            if any(
                val < 0
                for val in [
                    kline.open_price,
                    kline.high_price,
                    kline.low_price,
                    kline.close_price,
                    kline.volume,
                ]
            ):
                errors.append(f"Negative price or volume at index {i}")

            # Check for non-positive prices (should be > 0)
            if any(
                val <= 0
                for val in [
                    kline.open_price,
                    kline.high_price,
                    kline.low_price,
                    kline.close_price,
                ]
            ):
                errors.append(f"Non-positive prices at index {i}")

            prev_kline = kline

        # Check for gaps in time series data
        expected_interval = DataValidator._get_timeframe_delta(klines[0].timeframe)
        for i in range(1, len(klines)):
            expected_time = klines[i - 1].open_time + expected_interval
            if (
                abs((klines[i].open_time - expected_time).total_seconds()) > 60
            ):  # 1 minute tolerance
                errors.append(f"Time gap detected between index {i-1} and {i}")

        return len(errors) == 0, errors


    @staticmethod
    def validate_trades(trades: List[TradeData]) -> Tuple[bool, List[str]]:
        """Validate trade data"""
        errors = []

        if not trades:
            return True, []

        # Check chronological order
        for i in range(1, len(trades)):
            if trades[i].timestamp < trades[i - 1].timestamp:
                errors.append(f"Trades not in chronological order at index {i}")

        # Check data validity
        for i, trade in enumerate(trades):
            if trade.price <= 0:
                errors.append(f"Non-positive price at index {i}")

            if trade.quantity <= 0:
                errors.append(f"Non-positive quantity at index {i}")

            # Check quote quantity consistency
            expected_quote = trade.price * trade.quantity
            if (
                abs(trade.quote_quantity - expected_quote) / expected_quote > 0.01
            ):  # 1% tolerance
                errors.append(f"Quote quantity mismatch at index {i}")

        return len(errors) == 0, errors

    @staticmethod
    def validate_depth(depth: DepthData) -> Tuple[bool, List[str]]:
        """Validate order book depth data"""
        errors = []

        # Check bid/ask ordering
        for i in range(1, len(depth.bids)):
            if depth.bids[i].price >= depth.bids[i - 1].price:
                errors.append(f"Bids not in descending order at index {i}")

        for i in range(1, len(depth.asks)):
            if depth.asks[i].price <= depth.asks[i - 1].price:
                errors.append(f"Asks not in ascending order at index {i}")

        # Check spread
        if depth.bids and depth.asks:
            if depth.bids[0].price >= depth.asks[0].price:
                errors.append("Negative spread: best bid >= best ask")

        # Check positive quantities
        for i, bid in enumerate(depth.bids):
            if bid.quantity <= 0:
                errors.append(f"Non-positive bid quantity at index {i}")

        for i, ask in enumerate(depth.asks):
            if ask.quantity <= 0:
                errors.append(f"Non-positive ask quantity at index {i}")

        return len(errors) == 0, errors

    @staticmethod
    def _get_timeframe_delta(timeframe: TimeFrame) -> timedelta:
        """Get timedelta for a timeframe"""
        mapping = {
            TimeFrame.M1: timedelta(minutes=1),
            TimeFrame.M3: timedelta(minutes=3),
            TimeFrame.M5: timedelta(minutes=5),
            TimeFrame.M15: timedelta(minutes=15),
            TimeFrame.M30: timedelta(minutes=30),
            TimeFrame.H1: timedelta(hours=1),
            TimeFrame.H2: timedelta(hours=2),
            TimeFrame.H4: timedelta(hours=4),
            TimeFrame.H6: timedelta(hours=6),
            TimeFrame.H8: timedelta(hours=8),
            TimeFrame.H12: timedelta(hours=12),
            TimeFrame.D1: timedelta(days=1),
            TimeFrame.D3: timedelta(days=3),
            TimeFrame.W1: timedelta(weeks=1),
            TimeFrame.MN1: timedelta(days=30),
        }
        return mapping.get(timeframe, timedelta(minutes=1))


class ParquetCache:
    """Parquet-based data cache for efficient storage"""

    def __init__(self, config: CacheConfig):
        self.config = config

    def _get_cache_path(
        self,
        data_type: str,
        symbol: str,
        timeframe: str = None,
        start_date: str = None,
        end_date: str = None,
    ) -> Path:
        """Generate cache file path"""
        # Create hash from parameters for filename
        params = f"{data_type}_{symbol}_{timeframe}_{start_date}_{end_date}"
        cache_hash = hashlib.md5(params.encode()).hexdigest()[:16]

        filename = f"{data_type}_{symbol}_{cache_hash}.parquet"
        return self.config.cache_dir / filename

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file is valid and not expired"""
        if not cache_path.exists():
            return False

        # Check age
        file_age = datetime.now(timezone.utc) - datetime.fromtimestamp(
            cache_path.stat().st_mtime, tz=timezone.utc
        )
        max_age = timedelta(hours=self.config.max_cache_age_hours)

        return file_age < max_age

    def save_klines(
        self,
        klines: List[KlineData],
        symbol: str,
        timeframe: TimeFrame,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> Path:
        """Save klines to cache"""
        if not klines:
            return None

        # Convert to DataFrame
        df_data = []
        for kline in klines:
            row = asdict(kline)
            row["timeframe"] = kline.timeframe.value
            df_data.append(row)

        df = pd.DataFrame(df_data)

        # Convert datetime columns
        df["open_time"] = pd.to_datetime(df["open_time"])
        df["close_time"] = pd.to_datetime(df["close_time"])

        # Convert Decimal columns to float for Parquet compatibility
        decimal_cols = [
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "volume",
            "quote_volume",
            "taker_buy_volume",
            "taker_buy_quote_volume",
        ]
        for col in decimal_cols:
            df[col] = df[col].astype(float)

        # Generate cache path
        start_str = start_date.strftime("%Y%m%d") if start_date else None
        end_str = end_date.strftime("%Y%m%d") if end_date else None
        cache_path = self._get_cache_path(
            "klines", symbol, timeframe.value, start_str, end_str
        )

        # Save to Parquet
        df.to_parquet(cache_path, compression=self.config.compression, index=False)

        logger.info(f"Saved {len(klines)} klines to cache: {cache_path}")
        return cache_path

    def load_klines(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> Optional[List[KlineData]]:
        """Load klines from cache"""
        start_str = start_date.strftime("%Y%m%d") if start_date else None
        end_str = end_date.strftime("%Y%m%d") if end_date else None
        cache_path = self._get_cache_path(
            "klines", symbol, timeframe.value, start_str, end_str
        )

        if not self._is_cache_valid(cache_path):
            return None

        try:
            df = pd.read_parquet(cache_path)

            # Convert back to KlineData objects
            klines = []
            for _, row in df.iterrows():
                kline = KlineData(
                    symbol=row["symbol"],
                    open_time=row["open_time"].to_pydatetime(),
                    close_time=row["close_time"].to_pydatetime(),
                    open_price=Decimal(str(row["open_price"])),
                    high_price=Decimal(str(row["high_price"])),
                    low_price=Decimal(str(row["low_price"])),
                    close_price=Decimal(str(row["close_price"])),
                    volume=Decimal(str(row["volume"])),
                    quote_volume=Decimal(str(row["quote_volume"])),
                    trades_count=int(row["trades_count"]),
                    taker_buy_volume=Decimal(str(row["taker_buy_volume"])),
                    taker_buy_quote_volume=Decimal(str(row["taker_buy_quote_volume"])),
                    timeframe=TimeFrame(row["timeframe"]),
                )
                klines.append(kline)

            logger.info(f"Loaded {len(klines)} klines from cache: {cache_path}")
            return klines

        except Exception as e:
            logger.error(f"Error loading klines from cache: {e}")
            return None

    def save_trades(self, trades: List[TradeData], symbol: str) -> Path:
        """Save trades to cache"""
        if not trades:
            return None

        # Convert to DataFrame
        df_data = []
        for trade in trades:
            row = asdict(trade)
            df_data.append(row)

        df = pd.DataFrame(df_data)

        # Convert datetime columns
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Convert Decimal columns
        decimal_cols = ["price", "quantity", "quote_quantity"]
        for col in decimal_cols:
            df[col] = df[col].astype(float)

        # Generate cache path
        cache_path = self._get_cache_path("trades", symbol)

        # Save to Parquet
        df.to_parquet(cache_path, compression=self.config.compression, index=False)

        logger.info(f"Saved {len(trades)} trades to cache: {cache_path}")
        return cache_path

    def clear_cache(self, older_than_hours: int = None):
        """Clear cache files"""
        if older_than_hours is None:
            older_than_hours = self.config.max_cache_age_hours

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=older_than_hours)

        deleted_count = 0
        for cache_file in self.config.cache_dir.glob("*.parquet"):
            file_time = datetime.fromtimestamp(
                cache_file.stat().st_mtime, tz=timezone.utc
            )
            if file_time < cutoff_time:
                cache_file.unlink()
                deleted_count += 1

        logger.info(
            f"Cleared {deleted_count} cache files older than {older_than_hours} hours"
        )


class DataLoader:
    """
    Main data loader with caching and validation
    """

    def __init__(
        self,
        cache_config: CacheConfig = None,
        enable_validation: bool = True
    ):
        self.cache_config = cache_config or CacheConfig()
        self.cache = ParquetCache(self.cache_config)
        self.enable_validation = enable_validation
        self.validator = DataValidator()

    async def load_klines(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start_date: datetime,
        end_date: datetime,
        force_refresh: bool = False,
    ) -> List[KlineData]:
        """
        Load kline data with caching

        Args:
            symbol: Trading symbol
            timeframe: Kline timeframe
            start_date: Start date
            end_date: End date
            force_refresh: Force refresh from source

        Returns:
            List of KlineData objects
        """
        # Try cache first (unless force refresh)
        if not force_refresh:
            cached_klines = self.cache.load_klines(
                symbol, timeframe, start_date, end_date
            )
            if cached_klines is not None:
                logger.info(f"Loaded {len(cached_klines)} klines from cache")
                return cached_klines

        # Load from source (this would typically call an API)
        logger.info(
            f"Loading klines from source: {symbol} {timeframe.value} {start_date} to {end_date}"
        )

        # Placeholder for actual API call
        # klines = await self._fetch_klines_from_api(symbol, timeframe, start_date, end_date)
        klines = []  # Replace with actual implementation

        # Validate data
        if self.enable_validation and klines:
            is_valid, errors = self.validator.validate_klines(klines)
            if not is_valid:
                error_msg = f"Kline validation failed: {'; '.join(errors)}"
                logger.error(error_msg)
                raise DataValidationError(error_msg)

        # Cache the results
        if klines:
            self.cache.save_klines(klines, symbol, timeframe, start_date, end_date)

        return klines

    async def load_trades(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        force_refresh: bool = False,
    ) -> List[TradeData]:
        """Load trade data with caching"""
        # Implementation similar to load_klines
        logger.info(f"Loading trades: {symbol} {start_date} to {end_date}")

        # Placeholder for actual implementation
        trades = []

        if self.enable_validation and trades:
            is_valid, errors = self.validator.validate_trades(trades)
            if not is_valid:
                error_msg = f"Trade validation failed: {'; '.join(errors)}"
                logger.error(error_msg)
                raise DataValidationError(error_msg)

        return trades

    async def load_current_depth(self, symbol: str) -> DepthData:
        """Load current order book depth (not cached)"""
        logger.info(f"Loading current depth: {symbol}")

        # Placeholder for actual implementation
        # depth = await self._fetch_depth_from_api(symbol)
        depth = None

        if self.enable_validation and depth:
            is_valid, errors = self.validator.validate_depth(depth)
            if not is_valid:
                error_msg = f"Depth validation failed: {'; '.join(errors)}"
                logger.error(error_msg)
                raise DataValidationError(error_msg)

        return depth

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        cache_files = list(self.cache.config.cache_dir.glob("*.parquet"))

        total_size = sum(f.stat().st_size for f in cache_files)
        oldest_file = min(cache_files, key=lambda f: f.stat().st_mtime, default=None)
        newest_file = max(cache_files, key=lambda f: f.stat().st_mtime, default=None)

        return {
            "cache_dir": str(self.cache.config.cache_dir),
            "file_count": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "oldest_file": oldest_file.name if oldest_file else None,
            "newest_file": newest_file.name if newest_file else None,
            "max_age_hours": self.cache.config.max_cache_age_hours,
        }

    def clear_cache(self, older_than_hours: int = None):
        """Clear cache"""
        self.cache.clear_cache(older_than_hours)


# Utility functions


def create_data_loader(
    cache_dir: str = None,
    max_cache_age_hours: int = None,
    use_unified_config: bool = True,
) -> DataLoader:
    """Create a data loader with configuration

    Args:
        cache_dir: Cache directory (if None and use_unified_config=True, will use config)
        max_cache_age_hours: Max cache age (if None and use_unified_config=True, will use config)
        use_unified_config: Whether to use unified configuration for missing parameters
    """
    if use_unified_config and _unified_config_available:
        # Use unified config for missing parameters
        config = get_settings()
        cache_config = CacheConfig.from_unified_config(config)

        # Override with provided parameters
        if cache_dir is not None:
            cache_config.cache_dir = Path(cache_dir)
            cache_config.cache_dir.mkdir(exist_ok=True)
        if max_cache_age_hours is not None:
            cache_config.max_cache_age_hours = max_cache_age_hours
    else:
        # Traditional initialization with defaults
        cache_config = CacheConfig(
            cache_dir=cache_dir or "data_cache",
            max_cache_age_hours=max_cache_age_hours or 24,
        )

    return DataLoader(cache_config)


async def load_historical_data(
    loader: DataLoader, symbol: str, timeframe: TimeFrame, days_back: int = 30
) -> List[KlineData]:
    """Load historical data for the specified number of days"""
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)

    return await loader.load_klines(symbol, timeframe, start_date, end_date)
