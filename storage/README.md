# Storage Directory

## File Overview

This directory contains data persistence and storage modules for managing trading data, models, and artifacts.

### Python Files

#### `repo.py`
- **Purpose**: Main data repository with SQLite database and file storage
- **Functionality**: Provides comprehensive data persistence for trades, klines, backtests, and bot states

#### `artifacts.py`
- **Purpose**: Machine learning model and strategy artifact storage
- **Functionality**: Manages storage and retrieval of trained models, strategies, and other ML artifacts

## Function Documentation

### repo.py

#### `SQLiteRepository.__init__(db_path="trading_bot.db")`
- **What**: Initializes SQLite-based repository with database creation
- **Why**: Provides persistent storage for all trading data and results
- **What for**: Data persistence, analysis, and audit trails
- **Example**:
```python
repo = SQLiteRepository("production_bot.db")
# Creates/opens database with optimized settings
```

#### `save_trade(trade_record)`
- **What**: Saves a trade record to the database with full details
- **Why**: Maintains complete trade history for analysis and reporting
- **What for**: Trade logging, performance analysis, and compliance
- **Example**:
```python
trade = TradeRecord(
    run_id="session_123",
    order_id="binance_456",
    symbol="BTCUSDC",
    side="BUY",
    quantity=Decimal("0.001"),
    price=Decimal("50000"),
    timestamp=datetime.now(timezone.utc)
)
repo.save_trade(trade)
```

#### `load_trades(run_id=None, symbol=None, start_time=None, end_time=None, limit=None)`
- **What**: Loads trade records with flexible filtering options
- **Why**: Enables analysis and reporting on historical trades
- **What for**: Performance analysis, strategy evaluation, and reporting
- **Example**:
```python
trades = repo.load_trades(
    symbol="BTCUSDC",
    start_time=datetime.now() - timedelta(days=7),
    limit=100
)
# Returns: List[TradeRecord] for last 7 days
```

#### `save_klines(klines, symbol, interval)`
- **What**: Saves kline (candlestick) data to database with efficient storage
- **Why**: Maintains historical price data for analysis and backtesting
- **What for**: Market analysis, strategy development, and backtesting
- **Example**:
```python
repo.save_klines(klines_list, "BTCUSDC", "1h")
# Saves hourly kline data with deduplication
```

#### `load_klines(symbol, interval, start_time=None, end_time=None, limit=None)`
- **What**: Loads kline data with time-based filtering
- **Why**: Provides historical price data for analysis
- **What for**: Chart data, technical analysis, and strategy backtesting
- **Example**:
```python
klines = repo.load_klines(
    symbol="BTCUSDC",
    interval="1h",
    start_time=datetime.now() - timedelta(days=30)
)
# Returns: List[KlineData] for last 30 days
```

#### `save_backtest_result(result)`
- **What**: Saves complete backtest results including performance metrics
- **Why**: Preserves backtest results for comparison and analysis
- **What for**: Strategy evaluation, optimization, and reporting
- **Example**:
```python
result = BacktestResult(
    strategy_name="grid_trading",
    symbol="BTCUSDC",
    start_time=start_dt,
    end_time=end_dt,
    initial_capital=Decimal("1000"),
    final_capital=Decimal("1150"),
    total_return=Decimal("0.15"),
    max_drawdown=Decimal("0.05"),
    trades=trades_list
)
repo.save_backtest_result(result)
```

#### `load_backtest_results(strategy_name=None, symbol=None, limit=None)`
- **What**: Loads historical backtest results with filtering
- **Why**: Enables comparison and analysis of strategy performance
- **What for**: Strategy evaluation and optimization
- **Example**:
```python
results = repo.load_backtest_results(
    strategy_name="grid_trading",
    symbol="BTCUSDC",
    limit=10
)
# Returns: List[BacktestResult] for recent backtests
```

#### `save_bot_state(state, run_id)`
- **What**: Saves current bot state for persistence and recovery
- **Why**: Enables bot recovery after restarts or failures
- **What for**: State management and disaster recovery
- **Example**:
```python
repo.save_bot_state(current_bot_state, "session_123")
# Saves complete bot state including positions and orders
```

#### `load_bot_state(run_id)`
- **What**: Loads previously saved bot state for recovery
- **Why**: Enables seamless bot restart and state recovery
- **What for**: Bot recovery and session continuation
- **Example**:
```python
recovered_state = repo.load_bot_state("session_123")
# Returns: BotState object with all previous data
```

#### `get_trading_summary(run_id=None, start_time=None, end_time=None)`
- **What**: Generates comprehensive trading performance summary
- **Why**: Provides quick overview of trading performance
- **What for**: Performance monitoring and reporting
- **Example**:
```python
summary = repo.get_trading_summary(
    start_time=datetime.now() - timedelta(days=30)
)
# Returns: {'total_trades': 150, 'total_pnl': 250.5, 'win_rate': 0.65, ...}
```

#### `export_data(format="csv", output_dir="exports", tables=None)`
- **What**: Exports database data to various formats (CSV, JSON, Excel)
- **Why**: Enables data analysis in external tools and reporting
- **What for**: Data export, external analysis, and compliance reporting
- **Example**:
```python
repo.export_data(
    format="csv",
    output_dir="reports/2024_q1",
    tables=["trades", "klines"]
)
# Exports specified tables to CSV files
```

#### `FileRepository.__init__(base_path="data")`
- **What**: Initializes file-based repository with directory structure
- **Why**: Provides alternative storage using file system
- **What for**: Simple storage, development, and backup scenarios
- **Example**:
```python
file_repo = FileRepository("trading_data")
# Creates directory structure for file-based storage
```

#### `save_klines_parquet(klines, symbol, interval)`
- **What**: Saves kline data in efficient Parquet format
- **Why**: Provides fast, compressed storage for large datasets
- **What for**: High-performance data storage and analytics
- **Example**:
```python
file_repo.save_klines_parquet(klines_list, "BTCUSDC", "1m")
# Saves data in compressed Parquet format
```

#### `load_klines_parquet(symbol, interval)`
- **What**: Loads kline data from Parquet files
- **Why**: Provides fast access to large datasets
- **What for**: Analytics and backtesting with large data volumes
- **Example**:
```python
klines = file_repo.load_klines_parquet("BTCUSDC", "1m")
# Returns: List[KlineData] loaded from Parquet
```

### artifacts.py

#### `ArtifactManager.__init__(base_path="artifacts")`
- **What**: Initializes artifact storage manager with directory structure
- **Why**: Provides organized storage for models and ML artifacts
- **What for**: Model versioning, deployment, and management
- **Example**:
```python
artifact_mgr = ArtifactManager("ml_models")
# Creates directory structure for artifact storage
```

#### `save_model(model, name, model_type, version="1.0", metadata=None)`
- **What**: Saves machine learning model with versioning and metadata
- **Why**: Enables model versioning and deployment management
- **What for**: Model storage, versioning, and production deployment
- **Example**:
```python
metadata = {
    "accuracy": 0.87,
    "training_data": "BTCUSDC_2024_Q1",
    "features": ["SMA", "RSI", "MACD"]
}
artifact_mgr.save_model(
    model=trained_model,
    name="price_predictor",
    model_type="tensorflow",
    version="2.1",
    metadata=metadata
)
```

#### `load_model(name, version="latest")`
- **What**: Loads a previously saved model by name and version
- **Why**: Enables model deployment and inference
- **What for**: Loading trained models for prediction and trading
- **Example**:
```python
model = artifact_mgr.load_model("price_predictor", version="2.1")
# Returns: Loaded model ready for inference
```

#### `list_models(model_type=None)`
- **What**: Lists all available models with optional type filtering
- **Why**: Provides visibility into available models and versions
- **What for**: Model management and selection
- **Example**:
```python
models = artifact_mgr.list_models(model_type="tensorflow")
# Returns: [{'name': 'price_predictor', 'version': '2.1', ...}, ...]
```

#### `save_strategy_config(config, name, version="1.0")`
- **What**: Saves trading strategy configuration with versioning
- **Why**: Enables strategy configuration management and deployment
- **What for**: Strategy versioning and parameter management
- **Example**:
```python
config = {
    "grid_levels": 20,
    "grid_spacing": 0.001,
    "max_position": 1000,
    "risk_limit": 0.02
}
artifact_mgr.save_strategy_config(config, "btc_grid_v1", "1.3")
```

#### `load_strategy_config(name, version="latest")`
- **What**: Loads strategy configuration by name and version
- **Why**: Enables consistent strategy deployment and testing
- **What for**: Strategy configuration loading and execution
- **Example**:
```python
config = artifact_mgr.load_strategy_config("btc_grid_v1", "1.3")
# Returns: Strategy configuration dictionary
```

#### `save_backtest_artifacts(artifacts, strategy_name, run_id)`
- **What**: Saves backtest artifacts including plots, reports, and data
- **Why**: Preserves backtest results for analysis and reporting
- **What for**: Backtest result storage and analysis
- **Example**:
```python
artifacts = {
    "equity_curve": equity_plot,
    "trade_analysis": trade_report,
    "performance_metrics": metrics_dict
}
artifact_mgr.save_backtest_artifacts(artifacts, "grid_trading", "run_123")
```

#### `create_model_package(name, version, include_dependencies=True)`
- **What**: Creates deployable package with model and dependencies
- **Why**: Enables easy model deployment and distribution
- **What for**: Model deployment and production packaging
- **Example**:
```python
package_path = artifact_mgr.create_model_package(
    "price_predictor",
    "2.1",
    include_dependencies=True
)
# Returns: Path to deployment-ready package
```

#### `validate_model_integrity(name, version)`
- **What**: Validates model file integrity using checksums
- **Why**: Ensures model files haven't been corrupted
- **What for**: Model integrity verification and security
- **Example**:
```python
is_valid = artifact_mgr.validate_model_integrity("price_predictor", "2.1")
# Returns: True if model files are intact
```

#### `cleanup_old_versions(days_to_keep=30)`
- **What**: Removes old model versions to manage storage space
- **Why**: Prevents storage bloat from accumulating model versions
- **What for**: Storage management and cleanup automation
- **Example**:
```python
removed_count = artifact_mgr.cleanup_old_versions(days_to_keep=60)
# Returns: Number of old versions removed
```
