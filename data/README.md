# Data Directory

## Purpose

The data directory provides the foundational data structures and loading utilities that support all trading operations, including market data schemas, intelligent caching systems, and comprehensive data validation.

## Architecture

The data layer serves as the foundation for all trading operations by providing:
- Comprehensive market data type definitions with precise decimal handling
- Intelligent data loading with Parquet-based caching for performance
- Robust data validation and quality assurance systems
- Standardized data interfaces used throughout the application

## Files Overview

### Core Data Components

#### `schema.py` - Data Structures (541 lines)
- **Market Data Types**: Complete OHLCV data structures with precise Decimal arithmetic
- **Trading Data**: Order, trade, and position data structures with serialization
- **Event Data**: Market events, news, and real-time data stream structures
- **Validation Framework**: Built-in data validation with type checking and range validation
- **Serialization Support**: JSON and binary serialization for efficient storage and transmission

#### `loader.py` - Data Loading Engine (504 lines)
- **Intelligent Caching**: Parquet-based caching with automatic cache invalidation
- **Multi-source Loading**: Support for multiple data providers with failover
- **Data Validation**: Comprehensive data quality checks and anomaly detection
- **Performance Optimization**: Lazy loading, batch processing, and memory management
- **Cache Management**: Automatic cache cleanup and storage optimization

## Key Functions and Classes

### Data Structures

#### Core Market Data
- **`KlineData`**: OHLCV candlestick data with volume and trade count
- **`TradeData`**: Individual trade records with price, quantity, and timestamps
- **`TickerData`**: 24-hour ticker statistics with price change metrics
- **`DepthData`**: Order book depth with bid/ask levels and quantities

#### Trading Operations
- **`OrderData`**: Order information with status tracking and execution details
- **`PositionData`**: Position tracking with unrealized P&L and margin requirements
- **`BalanceData`**: Account balance information with available and locked amounts

#### Advanced Data Types
- **`NewsEvent`**: Market news and events with sentiment analysis
- **`MarketEvent`**: Trading halts, system maintenance, and market status updates

### Data Loading Functions

**`DataLoader.load_kline_data(symbol: str, interval: str, limit: int = 1000) -> List[KlineData]`**
- **Purpose**: Load OHLCV candlestick data with intelligent caching
- **Parameters**: Trading symbol, time interval (1m, 5m, 1h, 1d), optional limit
- **Returns**: List of validated KlineData objects
- **Caching**: Automatic Parquet caching with cache invalidation
- **Usage**: Primary interface for historical market data loading

**`DataLoader.get_latest_price(symbol: str) -> Decimal`**
- **Purpose**: Get current market price with caching for performance
- **Parameters**: Trading symbol (e.g., "BTCUSDC")
- **Returns**: Current price as Decimal for precise calculations
- **Caching**: Short-term caching to reduce API calls
- **Usage**: Real-time price monitoring and order calculations

**`DataValidator.validate_kline_data(data: List[KlineData]) -> ValidationResult`**
- **Purpose**: Comprehensive validation of market data quality
- **Parameters**: List of candlestick data to validate
- **Returns**: Validation result with detailed error reporting
- **Checks**: Price consistency, volume validation, timestamp ordering
- **Usage**: Data quality assurance before trading decisions

### Caching System

**`CacheConfig`**
- **Purpose**: Configuration for data caching behavior
- **Features**: TTL settings, storage limits, cleanup policies
- **Storage**: Parquet format for efficient compression and querying

**`DataLoader._get_cache_path(symbol: str, interval: str) -> Path`**
- **Purpose**: Generate cache file paths for data storage
- **Parameters**: Symbol and interval for cache organization
- **Returns**: Path object for cache file location
- **Organization**: Hierarchical cache structure by symbol and timeframe

## Data Validation Framework

### Validation Types
- **Type Validation**: Ensure all fields match expected types
- **Range Validation**: Verify prices and quantities within reasonable bounds
- **Consistency Validation**: Check data consistency across time series
- **Completeness Validation**: Detect missing data and gaps

### Error Handling
- **Graceful Degradation**: Continue operation with partial data when possible
- **Error Reporting**: Detailed validation error reports with corrective suggestions
- **Data Recovery**: Automatic data repair for common issues

## Integration Points

### With Core System
- **core/executor.py**: Real-time market data consumption for trading decisions
- **core/safety.py**: Price data for risk calculations and monitoring
- **core/state.py**: State data structures and persistence

### With AI Layer
- **ai/baseline.py**: Historical data for technical analysis and signal generation
- **ai/bo_suggester.py**: Market data for strategy optimization and backtesting

### With Storage Layer
- **storage/repo.py**: Data structures for trade and state persistence
- **storage/artifacts.py**: Market data for model training and validation

## Usage Examples

### Basic Data Loading
```python
# Load historical candlestick data
loader = DataLoader()
klines = await loader.load_kline_data("BTCUSDC", "1h", limit=100)

# Get current price
current_price = await loader.get_latest_price("BTCUSDC")
```

### Data Validation
```python
# Validate data quality
validator = DataValidator()
result = validator.validate_kline_data(klines)
if not result.is_valid:
    print(f"Validation errors: {result.errors}")
```

### Cache Management
```python
# Configure caching
cache_config = CacheConfig(
    ttl_hours=24,
    max_size_gb=1.0,
    cleanup_interval_hours=6
)
loader = DataLoader(cache_config=cache_config)

# Manual cache operations
await loader.clear_cache("BTCUSDC")
cache_info = loader.get_cache_info()
```

### Working with Data Types
```python
# Create market data
kline = KlineData(
    symbol="BTCUSDC",
    open_price=Decimal("50000.00"),
    high_price=Decimal("51000.00"),
    low_price=Decimal("49500.00"),
    close_price=Decimal("50500.00"),
    volume=Decimal("100.5"),
    timestamp=datetime.now(timezone.utc)
)

# Serialize for storage
json_data = kline.to_json()
binary_data = kline.to_bytes()

# Deserialize from storage
kline_restored = KlineData.from_json(json_data)
```
