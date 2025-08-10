# UI Directory

## File Overview

This directory contains user interface modules for interacting with the trading bot through various interfaces.

### Python Files

#### `cli.py`
- **Purpose**: Comprehensive command-line interface using Typer framework
- **Functionality**: Provides interactive CLI with enhanced features including mode switching, parameter collection, runtime controls, and advanced safety monitoring

#### `dashboard.py`
- **Purpose**: Web-based monitoring dashboard (placeholder implementation)
- **Functionality**: Provides portfolio summaries, trade monitoring, and performance visualization capabilities

## Function Documentation

### cli.py

#### `app = typer.Typer()`
- **What**: Main CLI application with comprehensive command structure
- **Why**: Provides user-friendly interface for all bot operations
- **What for**: Trading bot control, monitoring, and management
- **Example**:
```python
# Run CLI commands
python -m ui.cli trade --mode manual --symbol BTCUSDT
python -m ui.cli monitor sessions
python -m ui.cli safety status
```

#### `TradingMode(Enum)`
- **What**: Enumeration defining available trading modes
- **Why**: Provides structured mode selection for different trading approaches
- **What for**: Mode-specific parameter collection and operation
- **Example**:
```python
class TradingMode(Enum):
    MANUAL = "manual"  # Manual parameter input
    AUTO = "auto"      # AI-assisted parameters
    PAPER = "paper"    # Paper trading mode
```

#### `GridParameters.__init__(symbol, lower_bound, upper_bound, grid_count, investment_per_grid)`
- **What**: Data class for structured grid trading parameter storage
- **Why**: Provides type-safe parameter management with validation
- **What for**: Grid strategy configuration and validation
- **Example**:
```python
params = GridParameters(
    symbol="BTCUSDT",
    lower_bound=Decimal("48000"),
    upper_bound=Decimal("52000"),
    grid_count=20,
    investment_per_grid=Decimal("50")
)
```

#### `collect_grid_parameters(mode: TradingMode, symbol: str) -> GridParameters`
- **What**: Interactive parameter collection with mode-specific handling
- **Why**: Guides users through parameter setup with validation
- **What for**: User-friendly parameter configuration
- **Example**:
```python
params = collect_grid_parameters(TradingMode.MANUAL, "BTCUSDT")
# Interactive prompts for price bounds, grid count, investment amounts
# Returns: Validated GridParameters object
```

#### `trade_manual(symbol, config_file=None)`
- **What**: Executes manual trading mode with user-specified parameters
- **Why**: Provides hands-on trading control with custom parameters
- **What for**: Manual trading strategy execution
- **Example**:
```python
@app.command()
def trade_manual(symbol: str = "BTCUSDT", config_file: Optional[str] = None):
    # Collects parameters interactively
    # Executes grid trading with manual configuration
```

#### `trade_auto(symbol, use_ai=True)`
- **What**: Executes automated trading mode with AI-suggested parameters
- **Why**: Provides AI-optimized trading with minimal user input
- **What for**: Automated trading strategy execution
- **Example**:
```python
@app.command()
def trade_auto(symbol: str = "BTCUSDT", use_ai: bool = True):
    # Uses AI to suggest optimal parameters
    # Executes trading with automated configuration
```

#### `SessionManager.__init__()`
- **What**: Manages active trading sessions with lifecycle control
- **Why**: Provides session tracking and management capabilities
- **What for**: Multi-session trading and monitoring
- **Example**:
```python
session_mgr = SessionManager()
session_id = session_mgr.create_session(grid_params)
status = session_mgr.get_session_status(session_id)
```

#### `monitor_sessions()`
- **What**: Displays real-time status of all active trading sessions
- **Why**: Provides visibility into running trading operations
- **What for**: Session monitoring and performance tracking
- **Example**:
```python
@monitor_app.command("sessions")
def monitor_sessions():
    # Displays table of active sessions with:
    # - Session ID, Symbol, Status, P&L, Duration
    # - Real-time updates every 5 seconds
```

#### `adjust_parameters(session_id: str)`
- **What**: Allows real-time adjustment of trading session parameters
- **Why**: Enables dynamic strategy optimization during execution
- **What for**: Live parameter tuning and risk management
- **Example**:
```python
@monitor_app.command("adjust-parameters")
def adjust_parameters(session_id: str):
    # Interactive parameter adjustment
    # Updates grid spacing, position limits, etc.
```

#### `safety_status()`
- **What**: Displays comprehensive safety system status and alerts
- **Why**: Provides visibility into risk management systems
- **What for**: Risk monitoring and safety verification
- **Example**:
```python
@safety_app.command("status")
def safety_status():
    # Displays:
    # - Overall safety status
    # - Active alerts and warnings
    # - Risk metrics and thresholds
```

#### `emergency_stop(session_id: str)`
- **What**: Executes immediate emergency stop for a trading session
- **Why**: Provides emergency risk control capability
- **What for**: Emergency position closure and risk mitigation
- **Example**:
```python
@safety_app.command("emergency-stop")
def emergency_stop(session_id: str):
    # Immediately stops trading
    # Cancels all orders
    # Closes positions if configured
```

#### `monitor_market_conditions()`
- **What**: Displays real-time market condition analysis and regime detection
- **Why**: Provides market insight for trading decisions
- **What for**: Market analysis and strategy adaptation
- **Example**:
```python
@monitor_app.command("market-conditions")
def monitor_market_conditions():
    # Shows:
    # - Market regime (CALM, VOLATILE, EXTREME)
    # - Trend analysis and strength
    # - Volatility metrics
```

#### `flatten(symbol, reason="Manual")`
- **What**: Executes position flattening operation for emergency closure
- **Why**: Provides immediate position closure capability
- **What for**: Emergency position management and risk control
- **Example**:
```python
@app.command()
def flatten(symbol: str, reason: str = "Manual"):
    # Cancels all orders for symbol
    # Closes all positions
    # Provides execution summary
```

#### `backtest_grid(symbol, start_date, end_date, config_file=None)`
- **What**: Executes grid trading strategy backtesting
- **Why**: Validates strategy performance on historical data
- **What for**: Strategy testing and optimization
- **Example**:
```python
@backtest_app.command("grid")
def backtest_grid(
    symbol: str,
    start_date: str,
    end_date: str,
    config_file: Optional[str] = None
):
    # Runs grid strategy backtest
    # Generates performance report
    # Saves results to database
```

#### `export_data(format, output_dir, tables=None)`
- **What**: Exports trading data to various formats (CSV, JSON, Excel)
- **Why**: Enables external analysis and reporting
- **What for**: Data export and compliance reporting
- **Example**:
```python
@data_app.command("export")
def export_data(
    format: str = "csv",
    output_dir: str = "exports",
    tables: Optional[List[str]] = None
):
    # Exports specified data to chosen format
    # Creates organized output structure
```

### dashboard.py

#### `DashboardData.__init__(repository=None)`
- **What**: Initializes dashboard data provider with repository connection
- **Why**: Provides data aggregation for dashboard display
- **What for**: Dashboard data management and presentation
- **Example**:
```python
dashboard_data = DashboardData(repository=custom_repo)
```

#### `get_portfolio_summary()`
- **What**: Retrieves current portfolio summary with key metrics
- **Why**: Provides high-level portfolio overview
- **What for**: Portfolio monitoring and performance tracking
- **Example**:
```python
summary = await dashboard_data.get_portfolio_summary()
# Returns: {
#   'total_value': 10500.0,
#   'pnl_24h': 250.0,
#   'pnl_7d': 500.0,
#   'total_trades': 150,
#   'win_rate': 0.65
# }
```

#### `get_recent_trades(limit=50)`
- **What**: Retrieves recent trading activity with filtering
- **Why**: Provides visibility into recent trading operations
- **What for**: Trade monitoring and analysis
- **Example**:
```python
trades = await dashboard_data.get_recent_trades(limit=20)
# Returns: List of recent trades with timestamps, symbols, P&L
```

#### `get_performance_chart_data(days=30)`
- **What**: Generates data for portfolio performance charts
- **Why**: Provides visual performance tracking over time
- **What for**: Performance visualization and trend analysis
- **Example**:
```python
chart_data = await dashboard_data.get_performance_chart_data(days=7)
# Returns: {'dates': [...], 'values': [...], 'pnl': [...]}
```

#### `get_strategy_performance()`
- **What**: Retrieves performance breakdown by trading strategy
- **Why**: Enables strategy comparison and optimization
- **What for**: Strategy analysis and selection
- **Example**:
```python
performance = await dashboard_data.get_strategy_performance()
# Returns: List of strategies with performance metrics
```

#### `SimpleDashboard.display_summary()`
- **What**: Displays text-based portfolio summary in console
- **Why**: Provides quick portfolio overview without web interface
- **What for**: Console-based monitoring and reporting
- **Example**:
```python
dashboard = SimpleDashboard()
await dashboard.display_summary()
# Outputs formatted portfolio summary to console
```

#### `display_recent_trades(limit=10)`
- **What**: Displays recent trades in formatted table
- **Why**: Provides quick trade history overview
- **What for**: Recent activity monitoring
- **Example**:
```python
await dashboard.display_recent_trades(limit=5)
# Displays formatted table of 5 most recent trades
```

#### `run_full_dashboard()`
- **What**: Runs complete dashboard display with all sections
- **Why**: Provides comprehensive trading overview
- **What for**: Complete trading status review
- **Example**:
```python
await dashboard.run_full_dashboard()
# Displays:
# - Portfolio summary
# - Recent trades
# - Strategy performance
# - Market data status
```

#### `WebDashboard.start_server()`
- **What**: Starts web-based dashboard server (placeholder)
- **Why**: Provides browser-based interface for monitoring
- **What for**: Web-based trading dashboard (future implementation)
- **Example**:
```python
web_dashboard = WebDashboard(host="localhost", port=8080)
await web_dashboard.start_server()
# Would start web server for browser-based dashboard
```

#### `create_streamlit_dashboard()`
- **What**: Creates Streamlit-based interactive dashboard
- **Why**: Provides rich, interactive web interface for monitoring
- **What for**: Advanced dashboard with charts and real-time updates
- **Example**:
```python
if create_streamlit_dashboard():
    # Run with: streamlit run dashboard.py
    # Provides interactive web dashboard with:
    # - Real-time portfolio metrics
    # - Interactive performance charts
    # - Trade history tables
    # - Strategy performance analysis
```

#### `quick_status_check()`
- **What**: Performs quick status check and displays summary
- **Why**: Provides rapid status overview for monitoring
- **What for**: Quick health checks and status verification
- **Example**:
```python
await quick_status_check()
# Quickly displays current portfolio status and key metrics
```
