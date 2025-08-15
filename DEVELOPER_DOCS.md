# Developer Documentation

This document provides detailed API documentation for all functions and classes in the py-binance-bot codebase, organized by directory and module.

## Storage Layer

### storage/repo.py - Data Repository

#### `SQLiteRepository` Class

**`__init__(self, db_path: str = "trading_bot.db")`**
- **Purpose**: Initialize SQLite repository with connection pooling and schema setup
- **Parameters**: Database file path (defaults to "trading_bot.db")
- **Features**: Automatic schema creation, connection pooling, migration support

**`save_trade(self, trade: TradeData) -> bool`**
- **Purpose**: Persist trade data with validation and duplicate detection
- **Parameters**: `trade` - TradeData object with complete trade information
- **Returns**: Boolean indicating successful save operation
- **Features**: Automatic P&L calculation, duplicate prevention, referential integrity
- **Usage**: Called after every trade execution to maintain complete trade history

**`load_trades(self, start_time: datetime, end_time: datetime = None, symbol: str = None) -> List[TradeData]`**
- **Purpose**: Load historical trades with flexible filtering options
- **Parameters**: 
  - `start_time` - Start datetime for trade range
  - `end_time` - Optional end datetime (defaults to now)
  - `symbol` - Optional symbol filter
- **Returns**: List of TradeData objects sorted by timestamp
- **Optimization**: Indexed queries with pagination support for large datasets

**`save_state(self, state: TradingState, backup: bool = True) -> bool`**
- **Purpose**: Atomic persistence of complete bot state with backup support
- **Parameters**: 
  - `state` - TradingState object containing complete bot state
  - `backup` - Whether to create backup before saving
- **Returns**: Boolean indicating successful state persistence
- **Features**: Atomic writes, automatic backups, state validation

**`load_state(self, state_id: str = "default") -> Optional[TradingState]`**
- **Purpose**: Load saved bot state with validation and error recovery
- **Parameters**: `state_id` - State identifier (defaults to "default")
- **Returns**: TradingState object or None if not found
- **Features**: State validation, migration support, error recovery

### storage/artifacts.py - ML Artifact Management

#### `ArtifactManager` Class

**`__init__(self, base_path: str = "artifacts")`**
- **Purpose**: Initialize artifact manager with configurable storage location
- **Parameters**: `base_path` - Base directory for artifact storage
- **Features**: Directory creation, metadata indexing, version management

**`save_model(self, model: Any, name: str, metadata: Dict = None) -> str`**
- **Purpose**: Save ML models with automatic versioning and metadata tracking
- **Parameters**: 
  - `model` - Model object to serialize and store
  - `name` - Descriptive name for the model
  - `metadata` - Optional metadata dictionary
- **Returns**: Unique model identifier for future retrieval
- **Features**: Automatic serialization, version management, metadata validation

**`load_model(self, name: str, version: str = "latest") -> Tuple[Any, Dict]`**
- **Purpose**: Load saved models with metadata by name and version
- **Parameters**: 
  - `name` - Model name to load
  - `version` - Model version (defaults to "latest")
- **Returns**: Tuple of (model_object, metadata_dict)
- **Features**: Version resolution, metadata loading, automatic deserialization

**`get_model_performance(self, name: str) -> List[Dict]`**
- **Purpose**: Retrieve performance metrics across all versions of a model
- **Parameters**: `name` - Model name for performance lookup
- **Returns**: List of performance dictionaries sorted by version
- **Usage**: Model selection, performance monitoring, optimization tracking

**`cleanup_old_versions(self, name: str, keep_versions: int = 5) -> int`**
- **Purpose**: Remove old model versions while preserving recent ones
- **Parameters**: 
  - `name` - Model name for cleanup
  - `keep_versions` - Number of recent versions to preserve
- **Returns**: Number of versions removed
- **Features**: Safe cleanup with backup verification

## AI Layer

### ai/baseline.py - Strategy Engine

#### `BaselineStrategy` Class

**`__init__(self, mode: TradingMode = TradingMode.BALANCED, config: Dict = None)`**
- **Purpose**: Initialize baseline strategy with configurable trading mode
- **Parameters**: 
  - `mode` - Trading mode (CONSERVATIVE, BALANCED, AGGRESSIVE)
  - `config` - Optional configuration overrides
- **Features**: Mode-specific parameter sets, technical indicator configuration

**`generate_signal(self, market_data: List[KlineData]) -> TradingSignal`**
- **Purpose**: Generate trading signals based on technical analysis
- **Parameters**: `market_data` - Recent candlestick data for analysis
- **Returns**: TradingSignal with action, confidence, and reasoning
- **Analysis**: RSI, MACD, Bollinger Bands, volatility, trend detection

**`calculate_position_size(self, signal: TradingSignal, account_balance: Decimal, current_price: Decimal) -> Decimal`**
- **Purpose**: Calculate optimal position size based on risk parameters
- **Parameters**: 
  - `signal` - Trading signal with confidence score
  - `account_balance` - Available account balance
  - `current_price` - Current market price
- **Returns**: Position size in base currency
- **Features**: Volatility-adjusted sizing, confidence-based scaling

**`detect_market_regime(self, market_data: List[KlineData]) -> MarketRegime`**
- **Purpose**: Classify current market conditions for strategy adaptation
- **Parameters**: `market_data` - Historical price data for analysis
- **Returns**: MarketRegime (TRENDING, RANGING, VOLATILE)
- **Analysis**: Trend strength, volatility analysis, regime persistence

### ai/bo_suggester.py - Bayesian Optimization

#### `BayesianOptimizer` Class

**`__init__(self, acquisition_function: str = "ei", n_initial_points: int = 10)`**
- **Purpose**: Initialize Bayesian optimizer for hyperparameter tuning
- **Parameters**: 
  - `acquisition_function` - Acquisition function ("ei" or "ucb")
  - `n_initial_points` - Number of random initial evaluations
- **Features**: Gaussian Process modeling, acquisition function optimization

**`suggest_parameters(self, symbol: str, n_suggestions: int = 3) -> List[Dict]`**
- **Purpose**: Generate AI-powered parameter suggestions with confidence scores
- **Parameters**: 
  - `symbol` - Trading symbol for optimization
  - `n_suggestions` - Number of parameter sets to suggest
- **Returns**: List of parameter dictionaries with confidence scores
- **Process**: Market analysis, historical backtesting, Gaussian Process optimization

**`optimize_strategy(self, strategy_config: Dict, symbol: str, optimization_budget: int = 50) -> OptimizationResult`**
- **Purpose**: Comprehensive strategy optimization using Bayesian methods
- **Parameters**: 
  - `strategy_config` - Base strategy configuration
  - `symbol` - Trading symbol for optimization
  - `optimization_budget` - Number of evaluations to perform
- **Returns**: OptimizationResult with best parameters and performance metrics
- **Features**: Multi-objective optimization, convergence monitoring

### ai/validation.py - Model Validation

#### `WalkForwardValidator` Class

**`__init__(self, train_size: int = 30, test_size: int = 7, step_size: int = 1)`**
- **Purpose**: Initialize walk-forward validation for time series strategies
- **Parameters**: 
  - `train_size` - Training window size in days
  - `test_size` - Testing window size in days
  - `step_size` - Step size for window advancement
- **Features**: Time-series aware validation, overlapping windows

**`validate_strategy(self, strategy_config: Dict, symbol: str, total_days: int = 90) -> ValidationResult`**
- **Purpose**: Comprehensive strategy validation using walk-forward analysis
- **Parameters**: 
  - `strategy_config` - Strategy configuration to validate
  - `symbol` - Trading symbol for validation
  - `total_days` - Total historical period for validation
- **Returns**: ValidationResult with performance metrics and statistics
- **Metrics**: Sharpe ratio, maximum drawdown, win rate, profit factor

## Core Layer

### core/executor.py - Trading Executor

#### `TradingExecutor` Class

**`__init__(self, client, state: TradingState, strategy_callback, risk_limits: RiskLimits = None)`**
- **Purpose**: Initialize main trading executor with strategy integration
- **Parameters**: 
  - `client` - Exchange client for order execution
  - `state` - Current trading bot state
  - `strategy_callback` - Strategy function for signal generation
  - `risk_limits` - Optional risk management configuration
- **Features**: Strategy integration, risk monitoring, state management

**`run(self, state_save_path: str = None, save_interval_minutes: int = 5) -> None`**
- **Purpose**: Main trading loop with continuous market monitoring and execution
- **Parameters**: 
  - `state_save_path` - Path for periodic state saves
  - `save_interval_minutes` - Frequency of automatic state saves
- **Features**: Asynchronous execution, error recovery, graceful shutdown

**`process_strategy_signal(self, signal: TradingSignal) -> bool`**
- **Purpose**: Process trading signals and execute orders with risk checks
- **Parameters**: `signal` - Trading signal from strategy
- **Returns**: Boolean indicating successful signal processing
- **Features**: Risk validation, order execution, state updates

### core/safety.py - Safety Manager

#### `SafetyManager` Class

**`__init__(self, risk_limits: RiskLimits)`**
- **Purpose**: Initialize safety manager with configurable risk limits
- **Parameters**: `risk_limits` - Risk configuration object
- **Features**: Multi-layer protection, circuit breakers, real-time monitoring

**`check_risk_limits(self, state: TradingState, current_price: Decimal) -> RiskStatus`**
- **Purpose**: Comprehensive risk assessment with multiple safety checks
- **Parameters**: 
  - `state` - Current trading state
  - `current_price` - Current market price
- **Returns**: RiskStatus indicating safety level and any violations
- **Checks**: Position limits, daily loss limits, portfolio risk, correlation limits

**`execute_emergency_halt(self, reason: str) -> bool`**
- **Purpose**: Emergency trading halt with immediate position protection
- **Parameters**: `reason` - Reason for emergency halt
- **Returns**: Boolean indicating successful halt execution
- **Actions**: Stop new orders, flag for manual review, log emergency event

### core/state.py - State Management

#### `TradingState` Class

**`__init__(self, symbol: str, initial_balance: Decimal = Decimal('10000'))`**
- **Purpose**: Initialize trading state with symbol and balance
- **Parameters**: 
  - `symbol` - Primary trading symbol
  - `initial_balance` - Starting account balance
- **Features**: Position tracking, balance management, order history

**`update_position(self, side: str, quantity: Decimal, price: Decimal) -> None`**
- **Purpose**: Update position tracking after trade execution
- **Parameters**: 
  - `side` - Trade side ("BUY" or "SELL")
  - `quantity` - Trade quantity
  - `price` - Execution price
- **Features**: Position accumulation, average price calculation, P&L tracking

**`calculate_unrealized_pnl(self, current_price: Decimal) -> Decimal`**
- **Purpose**: Calculate unrealized profit/loss for current positions
- **Parameters**: `current_price` - Current market price
- **Returns**: Unrealized P&L as Decimal
- **Features**: Mark-to-market calculation, multi-position aggregation

#### State Management Functions

**`load_state(state_file: str) -> TradingState`**
- **Purpose**: Load trading state from file with validation and migration
- **Parameters**: `state_file` - Path to state file
- **Returns**: Loaded and validated TradingState object
- **Features**: Error recovery, schema migration, backup restoration

**`save_state(state: TradingState, state_file: str, backup: bool = True) -> bool`**
- **Purpose**: Atomic state persistence with backup support
- **Parameters**: 
  - `state` - TradingState object to save
  - `state_file` - Target file path
  - `backup` - Whether to create backup
- **Returns**: Boolean indicating successful save
- **Features**: Atomic writes, backup creation, validation

### core/grid_engine.py - Grid Trading Engine

#### `GridEngine` Class

**`__init__(self, config: GridConfig)`**
- **Purpose**: Initialize grid trading engine with configuration
- **Parameters**: `config` - Grid configuration with levels and spacing
- **Features**: Dynamic grid management, order coordination, profit tracking

**`execute_strategy(self, client, state: TradingState) -> None`**
- **Purpose**: Execute grid trading strategy with continuous management
- **Parameters**: 
  - `client` - Exchange client for order execution
  - `state` - Current trading state
- **Features**: Order placement, execution monitoring, grid rebalancing

**`calculate_grid_levels(self) -> List[GridLevel]`**
- **Purpose**: Calculate optimal grid levels based on configuration and market conditions
- **Returns**: List of GridLevel objects with prices and quantities
- **Features**: Dynamic spacing, volatility adjustment, risk-based sizing

**`rebalance_grid(self, current_price: Decimal) -> bool`**
- **Purpose**: Adjust grid levels based on market movement
- **Parameters**: `current_price` - Current market price
- **Returns**: Boolean indicating successful rebalancing
- **Features**: Level adjustment, order migration, profit preservation

## Data Layer

### data/schema.py - Data Structures

#### `KlineData` Class

**`__init__(self, symbol: str, open_price: Decimal, high_price: Decimal, low_price: Decimal, close_price: Decimal, volume: Decimal, timestamp: datetime)`**
- **Purpose**: Initialize OHLCV candlestick data with precise decimal arithmetic
- **Parameters**: Symbol, OHLCV prices, volume, and timestamp
- **Features**: Decimal precision, validation, serialization support

**`to_json(self) -> str`**
- **Purpose**: Serialize kline data to JSON format for storage or transmission
- **Returns**: JSON string representation
- **Features**: Decimal handling, timezone preservation, compact format

**`from_json(cls, json_str: str) -> 'KlineData'`**
- **Purpose**: Deserialize kline data from JSON format
- **Parameters**: `json_str` - JSON string to deserialize
- **Returns**: KlineData object
- **Features**: Type validation, error handling, decimal precision

### data/loader.py - Data Loading Engine

#### `DataLoader` Class

**`__init__(self, cache_config: CacheConfig = None)`**
- **Purpose**: Initialize data loader with intelligent caching configuration
- **Parameters**: `cache_config` - Optional cache configuration
- **Features**: Parquet caching, automatic invalidation, performance optimization

**`load_kline_data(self, symbol: str, interval: str, limit: int = 1000) -> List[KlineData]`**
- **Purpose**: Load OHLCV data with intelligent caching and validation
- **Parameters**: 
  - `symbol` - Trading symbol (e.g., "BTCUSDC")
  - `interval` - Time interval ("1m", "5m", "1h", "1d")
  - `limit` - Maximum number of candles to load
- **Returns**: List of validated KlineData objects
- **Features**: Cache checking, data validation, automatic retries

**`get_latest_price(self, symbol: str) -> Decimal`**
- **Purpose**: Get current market price with short-term caching for performance
- **Parameters**: `symbol` - Trading symbol
- **Returns**: Current price as Decimal for precise calculations
- **Features**: Real-time pricing, cache optimization, error handling

**`validate_data(self, data: List[KlineData]) -> ValidationResult`**
- **Purpose**: Comprehensive data quality validation
- **Parameters**: `data` - List of candlestick data to validate
- **Returns**: ValidationResult with detailed error reporting
- **Checks**: Price consistency, volume validation, timestamp ordering, gap detection

## Simulation & Reporting Layer

### sim/engine.py - Backtesting Engine

#### `BacktestEngine` Class

**`__init__(self, config: BacktestConfig)`**
- **Purpose**: Initialize event-driven backtesting engine with comprehensive simulation
- **Parameters**: `config` - Backtest configuration with symbol, dates, capital
- **Features**: Event processing, order simulation, realistic execution

**`run_backtest(self, strategy: TradingStrategy, market_data: List[KlineData]) -> Dict`**
- **Purpose**: Execute complete backtest with strategy and historical data
- **Parameters**: 
  - `strategy` - Trading strategy implementation
  - `market_data` - Historical market data for simulation
- **Returns**: Comprehensive backtest results with performance metrics
- **Features**: Event-driven processing, order execution, portfolio tracking

#### `TradingStrategy` (Abstract Base Class)

**`on_market_data(self, kline: KlineData) -> List[Order]`**
- **Purpose**: Process new market data and generate trading signals
- **Parameters**: `kline` - New candlestick data
- **Returns**: List of orders to place
- **Usage**: Core strategy logic implementation

**`on_order_fill(self, order: Order, fill_price: Decimal) -> List[Order]`**
- **Purpose**: Handle order execution and generate follow-up orders
- **Parameters**: Filled order and actual execution price
- **Returns**: New orders based on fill event
- **Usage**: Order management and grid rebalancing

### sim/evaluation.py - Performance Evaluation

#### `PerformanceEvaluator` Class

**`evaluate(self, equity_curve: List[EquityCurvePoint], trades: List[TradeRecord], initial_capital: Decimal) -> PerformanceMetrics`**
- **Purpose**: Calculate comprehensive performance analysis with all metrics
- **Parameters**: Portfolio history, trade records, starting capital
- **Returns**: PerformanceMetrics object with comprehensive analysis
- **Metrics**: Returns, risk, risk-adjusted, trade statistics, efficiency measures

**`_calculate_sharpe_ratio(self, returns: List[Decimal], volatility: Decimal) -> Decimal`**
- **Purpose**: Risk-adjusted return calculation using Sharpe ratio
- **Formula**: (Average Return - Risk Free Rate) / Volatility
- **Usage**: Primary risk-adjusted performance metric

**`_calculate_drawdowns(self, equity_curve: List[EquityCurvePoint]) -> Tuple`**
- **Purpose**: Comprehensive drawdown analysis
- **Returns**: Drawdown series, maximum drawdown, maximum duration
- **Features**: Peak-to-trough calculation, recovery time analysis

### sim/slippage.py - Slippage Models

#### `SlippageModel` (Abstract Base Class)

**`calculate_slippage(self, context: SlippageContext) -> Decimal`**
- **Purpose**: Calculate slippage as percentage of price for realistic execution
- **Parameters**: `context` - Complete market context for slippage calculation
- **Returns**: Slippage as decimal (e.g., 0.001 for 0.1% slippage)
- **Usage**: Called during order execution in backtesting

#### `LinearSlippageModel`

**`__init__(self, base_slippage_bps: Decimal = Decimal('2'), size_impact_factor: Decimal = Decimal('0.1'))`**
- **Purpose**: Order size-dependent slippage modeling
- **Parameters**: Base slippage and size impact factor
- **Formula**: `slippage = base + (order_value / 1000) * impact_factor`
- **Usage**: Realistic large order impact simulation

### reports/end_of_run.py - Report Generation

#### `ReportGenerator` Class

**`generate_backtest_report(self, result: BacktestResult, comparison_results: List[BacktestResult] = None) -> str`**
- **Purpose**: Generate comprehensive backtest analysis with performance charts
- **Parameters**: Primary backtest result and optional comparison results
- **Returns**: File path of generated HTML report
- **Features**: Equity curve charts, trade analysis, monthly returns

**`generate_optimization_report(self, optimization_results: Dict, strategy_name: str, symbol: str) -> str`**
- **Purpose**: Create hyperparameter optimization analysis with parameter importance
- **Features**: Parameter space visualization, convergence analysis, performance heatmaps
- **Usage**: Called after Bayesian optimization completion

**`_calculate_session_metrics(self, trades: List[TradeRecord], session_start: datetime, session_end: datetime) -> Dict`**
- **Purpose**: Calculate comprehensive trading session performance metrics
- **Returns**: Dictionary with session statistics including P&L, win rate, trading frequency
- **Metrics**: Total P&L, profit factor, largest win/loss, trades per hour

### reports/hodl_benchmark.py - Benchmark Analysis

#### `HODLBenchmark` Class

**`calculate_hodl_performance(self, klines: List[KlineData], initial_capital: Decimal) -> Dict`**
- **Purpose**: Calculate buy-and-hold strategy performance with realistic costs
- **Parameters**: Historical price data and initial investment
- **Returns**: Complete HODL performance metrics including fees
- **Features**: Entry/exit fee calculation, annualized returns, volatility analysis

**`compare_with_strategy(self, strategy_metrics: Dict, klines: List[KlineData], initial_capital: Decimal) -> Dict`**
- **Purpose**: Direct strategy vs HODL performance comparison
- **Features**: Risk-adjusted comparison, outperformance analysis, statistical significance
- **Usage**: Strategy validation and performance attribution

## UI Layer

### ui/cli.py - Command Line Interface

#### CLI Command Functions

**`collect_manual_parameters(symbol: str) -> GridParameters`**
- **Purpose**: Interactive collection of grid trading parameters with market context
- **Parameters**: `symbol` - Trading symbol for parameter collection
- **Returns**: Validated GridParameters dataclass with trading configuration
- **Features**: Input validation, market context display, confirmation workflow

**`get_market_context(symbol: str) -> dict`**
- **Purpose**: Fetch current market data to provide context for parameter selection
- **Parameters**: `symbol` - Trading symbol
- **Returns**: Dictionary with current price, volatility, and range data
- **Usage**: Called before parameter collection to show market conditions

### ui/session_manager.py - Session Management

#### `ThreadSafeSessionManager` Class

**`create_session(self, symbol: str, parameters: GridParameters) -> str`**
- **Purpose**: Create new trading session with thread-safe management
- **Parameters**: 
  - `symbol` - Trading symbol
  - `parameters` - Grid trading configuration
- **Returns**: Unique session identifier string
- **Features**: Thread safety, resource limits, automatic cleanup

**`update_session_status(self, session_id: str, status: TradingSessionStatus) -> bool`**
- **Purpose**: Update session status with proper timestamp tracking
- **Parameters**: 
  - `session_id` - Session identifier
  - `status` - New session status
- **Returns**: Success boolean
- **Features**: Status validation, timestamp management, statistics tracking

### ui/security.py - Security Layer

#### Security Functions

**`encrypt_credentials(api_key: str, api_secret: str) -> Dict[str, str]`**
- **Purpose**: Encrypt API credentials using Fernet symmetric encryption
- **Parameters**: 
  - `api_key` - Plain text API key
  - `api_secret` - Plain text API secret
- **Returns**: Dictionary with encrypted credentials or fallback data
- **Features**: PBKDF2 key derivation, secure encryption, fallback handling

**`validate_trading_symbol(symbol: str) -> str`**
- **Purpose**: Validate and normalize trading symbol format
- **Parameters**: `symbol` - Raw symbol input from user
- **Returns**: Standardized symbol format
- **Features**: Format validation, normalization, error handling

**`validate_decimal_input(value: str, min_value: Decimal = None, max_value: Decimal = None) -> Decimal`**
- **Purpose**: Safe parsing of decimal inputs for financial calculations
- **Parameters**: 
  - `value` - String input to parse
  - `min_value` - Optional minimum bound
  - `max_value` - Optional maximum bound
- **Returns**: Validated Decimal object
- **Features**: Range validation, precision handling, error recovery

### ui/error_handling.py - Error Management

#### Error Handling Functions

**`@handle_cli_errors` (decorator)**
- **Purpose**: Comprehensive error handling decorator for CLI functions
- **Features**: Exception categorization, user-friendly messages, detailed logging
- **Usage**: Applied to all major CLI command functions

**`parse_duration_safe(duration_str: str) -> timedelta`**
- **Purpose**: Parse human-readable duration strings with validation
- **Parameters**: `duration_str` - Duration string (e.g., "1h", "30m", "2d")
- **Returns**: Python timedelta object
- **Features**: Multiple format support, validation, error handling

### ui/output.py - Unified Output

#### `OutputHandler` Class

**`message(self, text: str, level: MessageLevel, console_style: str = None)`**
- **Purpose**: Unified interface for Rich console output and structured logging
- **Parameters**: 
  - `text` - Message content
  - `level` - Severity level
  - `console_style` - Optional Rich formatting
- **Features**: Level-based routing, icon mapping, dual output modes
