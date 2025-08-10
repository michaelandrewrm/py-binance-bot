# Core Directory

## File Overview

This directory contains the core trading engine components, including the main executor, grid trading engine, safety systems, and advanced features.

### Python Files

#### `executor.py`
- **Purpose**: Main trading execution loop that integrates safety checks and strategy execution
- **Functionality**: Coordinates all trading operations with comprehensive safety integration

#### `grid_engine.py`
- **Purpose**: Core grid trading strategy engine
- **Functionality**: Manages grid price levels, order planning, and state transitions

#### `safety.py`
- **Purpose**: Risk management and safety control systems
- **Functionality**: Implements circuit breakers, risk checks, and emergency controls

#### `flatten.py`
- **Purpose**: Emergency position flattening operations
- **Functionality**: Handles emergency position closure and order cancellation

#### `sizing.py`
- **Purpose**: Position sizing calculations and fee management
- **Functionality**: Calculates order sizes, manages fees, and handles position limits

#### `state.py`
- **Purpose**: Bot state management and persistence
- **Functionality**: Manages trading state, position tracking, and state persistence

#### `advanced_safety.py`
- **Purpose**: Advanced risk monitoring and market analysis (Phase 3)
- **Functionality**: Real-time risk assessment, market regime detection, and safety alerts

#### `enhanced_error_handling.py`
- **Purpose**: Intelligent error handling and automatic recovery (Phase 3)
- **Functionality**: Error classification, recovery strategies, and health monitoring

#### `baseline_heuristic.py`
- **Purpose**: Baseline parameter calculation and heuristic fallbacks
- **Functionality**: Provides parameter estimation when AI components are unavailable

## Function Documentation

### executor.py

#### `TradingExecutor.__init__(client, state, safety_manager, flatten_config, state_save_path)`
- **What**: Initializes the main trading executor with all required components
- **Why**: Sets up the coordinated trading system with safety integration
- **What for**: Creating a complete trading execution environment
- **Example**:
```python
executor = TradingExecutor(
    client=binance_client,
    state=bot_state,
    safety_manager=safety_mgr,
    flatten_config=flatten_cfg
)
```

#### `start_trading_loop(strategy_callback=None)`
- **What**: Starts the main trading loop with optional strategy callback
- **Why**: Begins automated trading operations with safety monitoring
- **What for**: Running the bot in production or testing environments
- **Example**:
```python
async def my_strategy(client, state):
    # Custom trading logic here
    pass

await executor.start_trading_loop(strategy_callback=my_strategy)
```

#### `stop_trading_loop()`
- **What**: Gracefully stops the trading loop and saves state
- **Why**: Ensures clean shutdown and state preservation
- **What for**: Controlled bot termination
- **Example**:
```python
await executor.stop_trading_loop()
# Output: Trading loop stopped gracefully, state saved
```

#### `run_safety_checks()`
- **What**: Executes comprehensive safety and risk checks
- **Why**: Ensures trading operations remain within safe parameters
- **What for**: Continuous risk monitoring during trading
- **Example**:
```python
action = await executor.run_safety_checks()
# Returns: SafetyAction.CONTINUE or SafetyAction.FLATTEN, etc.
```

### grid_engine.py

#### `GridEngine.__init__(config)`
- **What**: Initializes the grid trading engine with configuration
- **Why**: Sets up the grid strategy with defined price levels and parameters
- **What for**: Creating a grid trading strategy instance
- **Example**:
```python
config = GridConfig(
    center_price=Decimal('50000'),
    grid_spacing=Decimal('100'),
    num_levels_up=10,
    num_levels_down=10,
    order_amount=Decimal('100')
)
engine = GridEngine(config)
```

#### `generate_grid_levels()`
- **What**: Generates all price levels for the trading grid
- **Why**: Creates the price map for buy and sell orders
- **What for**: Setting up the complete grid structure
- **Example**:
```python
levels = engine.generate_grid_levels()
# Returns: {-10: GridLevel(49000, -10), ..., 10: GridLevel(51000, 10)}
```

#### `plan_next_orders(current_price)`
- **What**: Plans the next orders based on current market price
- **Why**: Determines which orders need to be placed or cancelled
- **What for**: Dynamic order management in grid trading
- **Example**:
```python
buy_orders, sell_orders = engine.plan_next_orders(Decimal('49500'))
# Returns: ([GridLevel(49400, -1)], [GridLevel(49600, 1)])
```

#### `update_position(filled_level, quantity)`
- **What**: Updates position tracking when an order is filled
- **Why**: Maintains accurate position and grid state
- **What for**: Position management and grid state synchronization
- **Example**:
```python
engine.update_position(
    filled_level=GridLevel(49500, 0),
    quantity=Decimal('0.002')
)
# Updates internal position tracking
```

### safety.py

#### `SafetyManager.__init__(risk_limits, callbacks=None)`
- **What**: Initializes the safety management system with risk limits
- **Why**: Sets up comprehensive risk monitoring and control
- **What for**: Protecting trading operations from adverse conditions
- **Example**:
```python
limits = RiskLimits(
    max_daily_loss=Decimal('500'),
    max_drawdown=Decimal('0.15')
)
safety_mgr = SafetyManager(limits)
```

#### `check_risk_limits(state, market_data)`
- **What**: Performs comprehensive risk assessment on current state
- **Why**: Identifies risk violations and determines appropriate actions
- **What for**: Continuous risk monitoring during trading
- **Example**:
```python
action, events = safety_mgr.check_risk_limits(bot_state, market_data)
# Returns: (SafetyAction.PAUSE, [SafetyEvent(...)])
```

#### `register_risk_callback(callback)`
- **What**: Registers a callback function for risk events
- **Why**: Enables custom responses to risk situations
- **What for**: Implementing custom risk management logic
- **Example**:
```python
def risk_handler(event: SafetyEvent):
    print(f"Risk event: {event.message}")

safety_mgr.register_risk_callback(risk_handler)
```

#### `get_safety_report()`
- **What**: Generates a comprehensive safety status report
- **Why**: Provides visibility into current risk levels and safety status
- **What for**: Monitoring and reporting on safety systems
- **Example**:
```python
report = safety_mgr.get_safety_report()
# Returns: {'status': 'OK', 'events': [...], 'risk_level': 'LOW'}
```

### flatten.py

#### `FlattenExecutor.__init__(client, config)`
- **What**: Initializes the position flattening executor
- **Why**: Prepares emergency position closure capabilities
- **What for**: Emergency risk management and position closure
- **Example**:
```python
flatten_config = FlattenConfig(
    cancel_orders_first=True,
    use_market_orders=True
)
flattener = FlattenExecutor(client, flatten_config)
```

#### `execute_flatten_operation(symbol, reason="Manual")`
- **What**: Executes complete position flattening for a symbol
- **Why**: Provides emergency position closure capability
- **What for**: Risk mitigation and emergency situations
- **Example**:
```python
result = await flattener.execute_flatten_operation(
    symbol="BTCUSDT",
    reason="Risk limit exceeded"
)
# Returns: FlattenResult with execution details
```

#### `cancel_all_orders(symbol)`
- **What**: Cancels all open orders for a specific symbol
- **Why**: Quickly removes market exposure
- **What for**: Emergency order management
- **Example**:
```python
cancelled_orders = await flattener.cancel_all_orders("BTCUSDT")
# Returns: List of cancelled order IDs
```

### sizing.py

#### `PositionSizer.__init__(fee_config, risk_config)`
- **What**: Initializes position sizing calculator with fee and risk configuration
- **Why**: Ensures proper position sizing considering fees and risk
- **What for**: Calculating appropriate order sizes for trading
- **Example**:
```python
fee_config = FeeConfig(maker_fee=0.001, taker_fee=0.001)
sizer = PositionSizer(fee_config, risk_config)
```

#### `calculate_order_size(price, budget, risk_percentage)`
- **What**: Calculates appropriate order size based on budget and risk
- **Why**: Ensures proper risk management in position sizing
- **What for**: Determining order quantities for trades
- **Example**:
```python
order_size = sizer.calculate_order_size(
    price=Decimal('50000'),
    budget=Decimal('1000'),
    risk_percentage=Decimal('0.02')
)
# Returns: Decimal('0.0004') (BTC quantity)
```

#### `calculate_grid_quantities(grid_config, total_budget)`
- **What**: Calculates quantities for all levels in a grid trading setup
- **Why**: Distributes budget appropriately across grid levels
- **What for**: Setting up grid trading with proper position sizing
- **Example**:
```python
quantities = sizer.calculate_grid_quantities(grid_config, Decimal('5000'))
# Returns: {level: quantity} mapping for each grid level
```

### state.py

#### `BotState.__init__(symbol, initial_balance, config)`
- **What**: Initializes bot state tracking for a trading session
- **Why**: Maintains comprehensive state information for the bot
- **What for**: State management and persistence during trading
- **Example**:
```python
state = BotState(
    symbol="BTCUSDT",
    initial_balance=Decimal('1000'),
    config={'grid_levels': 20}
)
```

#### `update_position(side, quantity, price)`
- **What**: Updates position tracking with new trade information
- **Why**: Maintains accurate position and P&L calculations
- **What for**: Position management and performance tracking
- **Example**:
```python
state.update_position(
    side="buy",
    quantity=Decimal('0.002'),
    price=Decimal('50000')
)
# Updates position size and calculates P&L
```

#### `save_state(state, file_path)`
- **What**: Saves current bot state to a file
- **Why**: Enables state persistence across bot restarts
- **What for**: State recovery and audit trails
- **Example**:
```python
save_state(bot_state, "trading_session_20241201.json")
# Saves complete state to JSON file
```

#### `load_state(file_path)`
- **What**: Loads previously saved bot state from a file
- **Why**: Enables bot recovery and continuation of previous sessions
- **What for**: State restoration and session recovery
- **Example**:
```python
recovered_state = load_state("trading_session_20241201.json")
# Returns: BotState object with all previous data
```

### advanced_safety.py

#### `AdvancedSafetyManager.__init__()`
- **What**: Initializes advanced safety monitoring with real-time risk analysis
- **Why**: Provides sophisticated risk management beyond basic safety checks
- **What for**: Advanced risk monitoring and market condition analysis
- **Example**:
```python
advanced_safety = AdvancedSafetyManager()
advanced_safety.start_monitoring("session_123", "BTCUSDT")
```

#### `check_session_safety(session_id, pnl, position_size, trade_count)`
- **What**: Performs comprehensive safety analysis for a trading session
- **Why**: Identifies complex risk patterns and market conditions
- **What for**: Advanced risk assessment and alert generation
- **Example**:
```python
is_safe, alerts = advanced_safety.check_session_safety(
    "session_123",
    pnl=Decimal('-150'),
    position_size=Decimal('2000'),
    trade_count=25
)
# Returns: (False, [SafetyAlert(...)])
```

#### `get_risk_report(session_id)`
- **What**: Generates detailed risk analysis report for a session
- **Why**: Provides comprehensive risk metrics and analysis
- **What for**: Risk monitoring and decision support
- **Example**:
```python
report = advanced_safety.get_risk_report("session_123")
# Returns: {'risk_level': 'HIGH', 'metrics': {...}, 'recommendations': [...]}
```

### enhanced_error_handling.py

#### `EnhancedErrorHandler.__init__()`
- **What**: Initializes intelligent error handling with automatic recovery
- **Why**: Provides robust error management and system resilience
- **What for**: Automatic error recovery and system stability
- **Example**:
```python
error_handler = EnhancedErrorHandler()
error_handler.start_monitoring()
```

#### `handle_error(error, session_id=None, category=None, context=None)`
- **What**: Processes errors with intelligent classification and recovery
- **Why**: Ensures system resilience through appropriate error responses
- **What for**: Automatic error handling and recovery in trading operations
- **Example**:
```python
recovery_action = await error_handler.handle_error(
    ConnectionError("API timeout"),
    session_id="session_123",
    category=ErrorCategory.NETWORK
)
# Returns: RecoveryAction.RETRY
```

#### `get_error_summary(hours=24)`
- **What**: Generates comprehensive error analysis and statistics
- **Why**: Provides visibility into system health and error patterns
- **What for**: System monitoring and troubleshooting
- **Example**:
```python
summary = error_handler.get_error_summary(hours=24)
# Returns: {'total_errors': 5, 'by_category': {...}, 'connection_health': {...}}
```

### baseline_heuristic.py

#### `BaselineHeuristic.__init__()`
- **What**: Initializes baseline parameter calculation system
- **Why**: Provides fallback when AI components are unavailable
- **What for**: Parameter estimation and strategy fallbacks
- **Example**:
```python
heuristic = BaselineHeuristic()
```

#### `calculate_grid_parameters(symbol, market_data, budget)`
- **What**: Calculates grid trading parameters using heuristic methods
- **Why**: Provides parameter estimation when AI optimization isn't available
- **What for**: Fallback parameter calculation for grid trading
- **Example**:
```python
params = heuristic.calculate_grid_parameters(
    symbol="BTCUSDT",
    market_data=market_data,
    budget=Decimal('1000')
)
# Returns: {'grid_spacing': 100, 'num_levels': 20, ...}
```

#### `estimate_volatility(price_history)`
- **What**: Estimates price volatility from historical data
- **Why**: Provides volatility metrics for risk and parameter calculations
- **What for**: Risk assessment and parameter optimization
- **Example**:
```python
volatility = heuristic.estimate_volatility(price_history)
# Returns: Decimal('0.025') (2.5% daily volatility)
```
