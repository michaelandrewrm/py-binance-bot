# Storage Directory

## Purpose

The storage directory provides comprehensive data persistence and artifact management, including SQLite-based trade storage, machine learning model management, and performance analytics. This layer ensures data integrity and efficient retrieval for all system operations.

## Architecture

The storage layer provides:
- **Persistent Data Management**: SQLite database for trade history, state management, and analytics
- **ML Artifact Storage**: Model versioning, metric tracking, and artifact lifecycle management
- **Performance Analytics**: Trade analysis, portfolio tracking, and performance measurement
- **Data Integrity**: ACID compliance, backup systems, and data validation

## Files Overview

### Core Storage Components

#### `repo.py` - Trade Repository
- **SQLite Database Management**: Complete trade and state persistence with ACID compliance
- **Trade History**: Comprehensive trade logging with execution details and performance metrics
- **State Management**: Bot state persistence for reliable restart and recovery
- **Query Interface**: Efficient data retrieval with filtering, sorting, and aggregation
- **Performance Analytics**: Built-in trade analysis and portfolio performance calculation

#### `artifacts.py` - ML Artifact Manager
- **Model Versioning**: Complete ML model lifecycle management with version control
- **Metric Tracking**: Performance metrics storage and historical comparison
- **Artifact Storage**: Efficient storage of models, datasets, and training artifacts
- **Cleanup Management**: Automatic artifact cleanup with retention policies
- **Backup Systems**: Automated backup and restore for critical ML assets

## Key Functions and Classes

### Trade Repository System

#### Database Management
**`TradeRepo.__init__(db_path: str = "trades.db")`**
- **Purpose**: Initialize SQLite database with proper schema and indexing
- **Parameters**: Database file path (defaults to trades.db)
- **Features**: Automatic schema creation, migration support, connection pooling
- **Usage**: Primary interface for all trade data operations

**`TradeRepo.save_trade(trade_data: TradeData) -> int`**
- **Purpose**: Persist trade information with complete execution details
- **Parameters**: TradeData object with order details, prices, and timestamps
- **Returns**: Unique trade ID for future reference
- **Features**: Automatic P&L calculation, duplicate detection, data validation
- **Usage**: Called after every trade execution for permanent record keeping

**`TradeRepo.get_trade_history(symbol: Optional[str] = None, days: int = 30) -> List[TradeData]`**
- **Purpose**: Retrieve historical trades with flexible filtering options
- **Parameters**: Optional symbol filter, time range in days
- **Returns**: List of TradeData objects matching criteria
- **Features**: Efficient indexing, pagination support, sorting options
- **Usage**: Performance analysis, tax reporting, strategy evaluation

#### State Management
**`TradeRepo.save_bot_state(state: BotState) -> None`**
- **Purpose**: Persist complete bot state for reliable restart capability
- **Parameters**: BotState object with positions, balances, and configuration
- **Features**: Atomic updates, versioning, rollback capability
- **Usage**: Called periodically and before shutdown for state preservation

**`TradeRepo.load_bot_state() -> Optional[BotState]`**
- **Purpose**: Restore bot state from database on startup
- **Returns**: BotState object or None if no saved state exists
- **Features**: State validation, corruption detection, recovery options
- **Usage**: Bot initialization to restore previous session state

#### Performance Analytics
**`TradeRepo.calculate_performance_metrics(days: int = 30) -> PerformanceMetrics`**
- **Purpose**: Calculate comprehensive trading performance statistics
- **Parameters**: Analysis period in days
- **Returns**: PerformanceMetrics with returns, Sharpe ratio, drawdown analysis
- **Features**: Risk-adjusted returns, benchmark comparison, drawdown calculation
- **Usage**: Performance evaluation, strategy comparison, reporting

**`TradeRepo.get_portfolio_value_history(days: int = 30) -> List[Tuple[datetime, Decimal]]`**
- **Purpose**: Track portfolio value changes over time
- **Parameters**: Analysis period in days
- **Returns**: Time series of portfolio values
- **Features**: Real-time value calculation, historical tracking, trend analysis
- **Usage**: Portfolio monitoring, performance visualization, risk assessment

### ML Artifact Management

#### Model Storage and Versioning
**`ArtifactManager.__init__(base_path: str = "artifacts/")`**
- **Purpose**: Initialize artifact storage with proper directory structure
- **Parameters**: Base storage path for all ML artifacts
- **Features**: Automatic directory creation, permission management, backup setup
- **Usage**: Central hub for all ML model and data management

**`ArtifactManager.save_model(model: Any, name: str, version: str, metadata: Dict) -> str`**
- **Purpose**: Save ML model with versioning and comprehensive metadata
- **Parameters**: Model object, name identifier, version string, metadata dictionary
- **Returns**: Unique artifact ID for future reference
- **Features**: Model serialization, metadata validation, version conflict detection
- **Usage**: Model persistence after training, deployment preparation

**`ArtifactManager.load_model(name: str, version: Optional[str] = None) -> Tuple[Any, Dict]`**
- **Purpose**: Load ML model with specific version or latest available
- **Parameters**: Model name, optional version (defaults to latest)
- **Returns**: Tuple of model object and metadata
- **Features**: Version resolution, dependency checking, lazy loading
- **Usage**: Model deployment, inference, continued training

#### Metric Tracking
**`ArtifactManager.save_metrics(experiment_id: str, metrics: Dict[str, float]) -> None`**
- **Purpose**: Store training and validation metrics for experiment tracking
- **Parameters**: Experiment identifier, metrics dictionary
- **Features**: Time series storage, metric validation, aggregation support
- **Usage**: Training progress tracking, model comparison, hyperparameter optimization

**`ArtifactManager.get_metric_history(experiment_id: str, metric_name: str) -> List[Tuple[datetime, float]]`**
- **Purpose**: Retrieve historical metric values for analysis
- **Parameters**: Experiment ID, specific metric name
- **Returns**: Time series of metric values
- **Features**: Efficient querying, interpolation, trend analysis
- **Usage**: Training visualization, convergence analysis, performance comparison

#### Artifact Lifecycle
**`ArtifactManager.list_artifacts(name_pattern: Optional[str] = None) -> List[ArtifactInfo]`**
- **Purpose**: List available artifacts with filtering and metadata
- **Parameters**: Optional name pattern for filtering
- **Returns**: List of artifact information objects
- **Features**: Pattern matching, metadata extraction, size calculation
- **Usage**: Artifact discovery, storage management, cleanup planning

**`ArtifactManager.cleanup_old_artifacts(retention_days: int = 30) -> int`**
- **Purpose**: Remove old artifacts based on retention policy
- **Parameters**: Retention period in days
- **Returns**: Number of artifacts removed
- **Features**: Safe deletion, backup verification, usage tracking
- **Usage**: Storage management, cost optimization, maintenance

## Database Schema

### Core Tables

#### trades
- **id** (INTEGER PRIMARY KEY): Unique trade identifier
- **symbol** (TEXT): Trading pair (e.g., BTCUSDC)
- **side** (TEXT): BUY or SELL operation
- **quantity** (DECIMAL): Trade quantity with precision
- **price** (DECIMAL): Execution price
- **timestamp** (DATETIME): Trade execution time
- **pnl** (DECIMAL): Realized profit/loss
- **fees** (DECIMAL): Trading fees paid
- **strategy** (TEXT): Strategy that generated the trade
- **metadata** (JSON): Additional trade context

#### bot_states
- **id** (INTEGER PRIMARY KEY): State version identifier
- **timestamp** (DATETIME): State save time
- **state_data** (JSON): Complete bot state serialization
- **checksum** (TEXT): Data integrity verification

#### portfolio_snapshots
- **id** (INTEGER PRIMARY KEY): Snapshot identifier
- **timestamp** (DATETIME): Snapshot time
- **total_value** (DECIMAL): Portfolio value in USDC
- **positions** (JSON): Active positions data
- **balances** (JSON): Account balances

## Integration Points

### With Core System
- **core/executor.py**: Trade execution logging and state persistence
- **core/safety.py**: Risk metric storage and historical analysis
- **core/state.py**: State management and persistence coordination

### With AI Layer
- **ai/baseline.py**: Model storage and performance metric tracking
- **ai/bo_suggester.py**: Optimization result storage and experiment tracking
- **ai/validation.py**: Validation metric storage and model comparison

### With UI Layer
- **ui/dashboard.py**: Performance data retrieval for visualization
- **reports/**: Historical data for comprehensive report generation

## Usage Examples

### Trade Repository Operations
```python
# Initialize repository
repo = TradeRepo("trading_data.db")

# Save trade execution
trade = TradeData(
    symbol="BTCUSDC",
    side="BUY",
    quantity=Decimal("0.1"),
    price=Decimal("50000.00"),
    timestamp=datetime.now(timezone.utc)
)
trade_id = await repo.save_trade(trade)

# Retrieve trade history
recent_trades = await repo.get_trade_history(days=7)
btc_trades = await repo.get_trade_history(symbol="BTCUSDC", days=30)

# Performance analysis
metrics = await repo.calculate_performance_metrics(days=30)
print(f"Total Return: {metrics.total_return:.2%}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
```

### State Management
```python
# Save bot state
current_state = BotState(
    positions={"BTCUSDC": position_data},
    balances={"USDC": balance_data},
    configuration=config_dict
)
await repo.save_bot_state(current_state)

# Restore on startup
saved_state = await repo.load_bot_state()
if saved_state:
    bot.restore_state(saved_state)
```

### ML Artifact Management
```python
# Initialize artifact manager
artifacts = ArtifactManager("ml_artifacts/")

# Save trained model
model_id = await artifacts.save_model(
    model=trained_model,
    name="price_predictor",
    version="1.2.0",
    metadata={
        "accuracy": 0.87,
        "training_samples": 10000,
        "features": ["price", "volume", "rsi"]
    }
)

# Load model for inference
model, metadata = await artifacts.load_model("price_predictor", "1.2.0")
prediction = model.predict(input_data)

# Track training metrics
await artifacts.save_metrics("exp_001", {
    "loss": 0.023,
    "accuracy": 0.87,
    "val_loss": 0.031
})

# Cleanup old artifacts
removed_count = await artifacts.cleanup_old_artifacts(retention_days=30)
print(f"Removed {removed_count} old artifacts")
```

### Performance Monitoring
```python
# Portfolio value tracking
value_history = await repo.get_portfolio_value_history(days=30)
current_value = value_history[-1][1]
initial_value = value_history[0][1]
return_pct = (current_value - initial_value) / initial_value

# Detailed performance analysis
metrics = await repo.calculate_performance_metrics(days=90)
print(f"Win Rate: {metrics.win_rate:.1%}")
print(f"Avg Win: ${metrics.avg_win:.2f}")
print(f"Avg Loss: ${metrics.avg_loss:.2f}")
print(f"Profit Factor: {metrics.profit_factor:.2f}")
```
