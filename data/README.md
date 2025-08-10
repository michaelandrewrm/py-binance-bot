# Data Directory

## File Overview

This directory contains data handling and loading modules for market data management, validation, and caching.

### Python Files

#### `schema.py`
- **Purpose**: Defines typed data schemas and records for system-wide data exchange
- **Functionality**: Provides data structures for klines, trades, tickers, and other market data types

#### `loader.py`
- **Purpose**: Handles data loading, validation, and Parquet-based caching
- **Functionality**: Loads market data from various sources, validates data quality, and manages efficient caching

## Function Documentation

### schema.py

#### `KlineData.__init__(symbol, open_time, close_time, open, high, low, close, volume, ...)`
- **What**: Initializes a candlestick (kline) data record
- **Why**: Provides structured representation of OHLCV price data
- **What for**: Market data analysis, charting, and strategy calculations
- **Example**:
```python
kline = KlineData(
    symbol="BTCUSDT",
    open_time=datetime.now(timezone.utc),
    close_time=datetime.now(timezone.utc) + timedelta(minutes=1),
    open=Decimal("50000"),
    high=Decimal("50100"),
    low=Decimal("49900"),
    close=Decimal("50050"),
    volume=Decimal("10.5")
)
```

#### `TradeData.__init__(symbol, trade_id, price, quantity, timestamp, is_buyer_maker)`
- **What**: Initializes a trade execution record
- **Why**: Represents individual trade executions for detailed analysis
- **What for**: Trade analysis, volume profiling, and execution tracking
- **Example**:
```python
trade = TradeData(
    symbol="BTCUSDT",
    trade_id="12345",
    price=Decimal("50000"),
    quantity=Decimal("0.1"),
    timestamp=datetime.now(timezone.utc),
    is_buyer_maker=True
)
```

#### `TickerData.__init__(symbol, price_change, price_change_percent, last_price, volume, ...)`
- **What**: Initializes 24-hour ticker statistics record
- **Why**: Provides market statistics for analysis and monitoring
- **What for**: Market condition assessment and symbol screening
- **Example**:
```python
ticker = TickerData(
    symbol="BTCUSDT",
    price_change=Decimal("500"),
    price_change_percent=Decimal("1.0"),
    last_price=Decimal("50500"),
    volume=Decimal("1000"),
    high=Decimal("51000"),
    low=Decimal("49500")
)
```

#### `klines_from_binance(raw_data)`
- **What**: Converts raw Binance kline data to KlineData objects
- **Why**: Standardizes data format from Binance API responses
- **What for**: Data ingestion and normalization from Binance
- **Example**:
```python
raw_klines = [
    [1609459200000, "50000.0", "50100.0", "49900.0", "50050.0", "10.5", ...]
]
klines = klines_from_binance(raw_klines)
# Returns: [KlineData(symbol="BTCUSDT", open=50000, ...)]
```

#### `trades_from_binance(raw_data)`
- **What**: Converts raw Binance trade data to TradeData objects
- **Why**: Standardizes trade data format from Binance API
- **What for**: Trade data ingestion and analysis
- **Example**:
```python
raw_trades = [
    {"id": 12345, "price": "50000.0", "qty": "0.1", "time": 1609459200000, ...}
]
trades = trades_from_binance(raw_trades)
# Returns: [TradeData(trade_id="12345", price=50000, ...)]
```

#### `depth_from_binance(raw_data)`
- **What**: Converts raw Binance order book data to DepthData objects
- **Why**: Standardizes order book data format
- **What for**: Order book analysis and liquidity assessment
- **Example**:
```python
raw_depth = {
    "bids": [["49990", "1.0"], ["49980", "2.0"]],
    "asks": [["50010", "1.5"], ["50020", "2.5"]]
}
depth = depth_from_binance(raw_depth)
# Returns: DepthData with bid/ask levels
```

#### `utc_isoformat(dt)`
- **What**: Converts datetime to ISO 8601 string with UTC timezone
- **Why**: Provides consistent datetime string formatting
- **What for**: Data serialization and API communication
- **Example**:
```python
dt = datetime.now(timezone.utc)
iso_string = utc_isoformat(dt)
# Returns: "2024-01-01T12:00:00Z"
```

### loader.py

#### `DataLoader.__init__(cache_config=None)`
- **What**: Initializes data loader with caching configuration
- **Why**: Provides efficient data loading with caching capabilities
- **What for**: Loading and caching market data for analysis
- **Example**:
```python
cache_config = CacheConfig(cache_dir="market_data", max_cache_age_hours=12)
loader = DataLoader(cache_config)
```

#### `load_klines(symbol, interval, start_time=None, end_time=None, limit=500)`
- **What**: Loads kline data with caching and validation
- **Why**: Provides efficient access to historical price data
- **What for**: Market analysis, backtesting, and strategy development
- **Example**:
```python
klines = await loader.load_klines(
    symbol="BTCUSDT",
    interval=TimeFrame.H1,
    start_time=datetime.now() - timedelta(days=7),
    limit=168  # 7 days of hourly data
)
# Returns: List[KlineData] with validated kline data
```

#### `load_trades(symbol, start_time=None, end_time=None, limit=1000)`
- **What**: Loads trade data with optional time filtering
- **Why**: Provides access to detailed trade execution data
- **What for**: Trade analysis, volume profiling, and market microstructure analysis
- **Example**:
```python
trades = await loader.load_trades(
    symbol="BTCUSDT",
    start_time=datetime.now() - timedelta(hours=1),
    limit=500
)
# Returns: List[TradeData] with recent trade executions
```

#### `load_depth(symbol, limit=100)`
- **What**: Loads current order book depth data
- **Why**: Provides liquidity information for order placement
- **What for**: Liquidity analysis and optimal order sizing
- **Example**:
```python
depth = await loader.load_depth("BTCUSDT", limit=20)
# Returns: DepthData with top 20 bid/ask levels
```

#### `cache_data(data, cache_key, ttl_hours=24)`
- **What**: Caches data using Parquet format with TTL
- **Why**: Improves performance by avoiding repeated API calls
- **What for**: Efficient data storage and retrieval
- **Example**:
```python
await loader.cache_data(
    data=klines_list,
    cache_key="BTCUSDT_1h_20241201",
    ttl_hours=6
)
# Caches data for 6 hours
```

#### `get_cached_data(cache_key)`
- **What**: Retrieves cached data if available and not expired
- **Why**: Provides fast access to previously loaded data
- **What for**: Performance optimization and reduced API usage
- **Example**:
```python
cached_klines = await loader.get_cached_data("BTCUSDT_1h_20241201")
# Returns: List[KlineData] or None if not cached/expired
```

#### `validate_data_quality(data, data_type)`
- **What**: Validates data quality and consistency
- **Why**: Ensures data integrity for analysis and trading
- **What for**: Data quality assurance and error detection
- **Example**:
```python
is_valid, issues = loader.validate_data_quality(klines, "klines")
# Returns: (True, []) if valid, or (False, ["Missing data points"]) if issues
```

#### `DataValidator.validate_klines(klines)`
- **What**: Validates kline data for consistency and completeness
- **Why**: Ensures kline data meets quality standards
- **What for**: Data quality assurance in market analysis
- **Example**:
```python
is_valid, errors = DataValidator.validate_klines(klines_list)
# Returns: (True, []) if valid, or (False, ["OHLC inconsistency at index 5"])
```

#### `DataValidator.validate_trades(trades)`
- **What**: Validates trade data for consistency and logical correctness
- **Why**: Ensures trade data integrity for analysis
- **What for**: Trade data quality assurance
- **Example**:
```python
is_valid, errors = DataValidator.validate_trades(trades_list)
# Returns: (True, []) if valid, or (False, ["Negative quantity at trade 123"])
```

#### `ParquetCache.save(data, file_path)`
- **What**: Saves data to Parquet format for efficient storage
- **Why**: Provides fast, compressed data storage and retrieval
- **What for**: Long-term data caching and archival
- **Example**:
```python
cache = ParquetCache()
cache.save(klines_list, "cache/BTCUSDT_2024_01.parquet")
# Saves data in compressed Parquet format
```

#### `ParquetCache.load(file_path)`
- **What**: Loads data from Parquet format cache
- **Why**: Provides fast data loading from compressed storage
- **What for**: Efficient data retrieval from cache
- **Example**:
```python
klines = cache.load("cache/BTCUSDT_2024_01.parquet")
# Returns: List[KlineData] loaded from Parquet file
```

#### `DataAggregator.resample_klines(klines, target_interval)`
- **What**: Resamples kline data to different time intervals
- **Why**: Enables analysis at different timeframes
- **What for**: Multi-timeframe analysis and strategy development
- **Example**:
```python
hourly_klines = DataAggregator.resample_klines(
    minute_klines,
    target_interval=TimeFrame.H1
)
# Converts 1-minute klines to 1-hour klines
```

#### `DataQualityChecker.check_gaps(klines, expected_interval)`
- **What**: Identifies missing data points or gaps in time series
- **Why**: Ensures data completeness for analysis
- **What for**: Data quality monitoring and gap detection
- **Example**:
```python
gaps = DataQualityChecker.check_gaps(klines, TimeFrame.M5)
# Returns: [datetime(2024, 1, 1, 12, 15), ...] list of missing timestamps
```
