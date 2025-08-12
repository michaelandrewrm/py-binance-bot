# Sim Directory

## File Overview

This directory contains simulation and backtesting components for strategy testing and performance evaluation.

### Python Files

#### `engine.py`
- **Purpose**: Event-driven backtesting engine for strategy simulation
- **Functionality**: Provides comprehensive simulation environment with realistic market conditions and execution

#### `evaluation.py`
- **Purpose**: Performance evaluation and metrics calculation
- **Functionality**: Calculates risk-adjusted returns, drawdowns, Sharpe ratios, and other performance metrics

#### `slippage.py`
- **Purpose**: Pluggable slippage models for realistic trade execution simulation
- **Functionality**: Implements various slippage models to simulate real-world execution costs

## Function Documentation

### engine.py

#### `BacktestEngine.__init__(config, strategy, data_source)`
- **What**: Initializes the backtesting engine with configuration and strategy
- **Why**: Provides a comprehensive simulation environment for strategy testing
- **What for**: Strategy backtesting, optimization, and validation
- **Example**:
```python
config = BacktestConfig(
    symbol="BTCUSDC",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 6, 30),
    initial_capital=Decimal("10000"),
    commission_rate=Decimal("0.001")
)
engine = BacktestEngine(config, my_strategy, data_loader)
```

#### `run_backtest()`
- **What**: Executes the complete backtesting simulation
- **Why**: Tests strategy performance on historical data
- **What for**: Strategy validation and performance assessment
- **Example**:
```python
result = await engine.run_backtest()
# Returns: BacktestResult with performance metrics and trade history
```

#### `add_event(event)`
- **What**: Adds an event to the simulation event queue
- **Why**: Enables event-driven simulation with proper timing
- **What for**: Market data events, order fills, and custom signals
- **Example**:
```python
market_event = SimulationEvent(
    timestamp=datetime.now(),
    event_type=EventType.MARKET_DATA,
    data={"price": 50000, "volume": 100}
)
engine.add_event(market_event)
```

#### `process_events()`
- **What**: Processes all queued events in chronological order
- **Why**: Maintains proper event ordering in simulation
- **What for**: Realistic simulation timing and order execution
- **Example**:
```python
await engine.process_events()
# Processes all pending events in correct sequence
```

#### `SimpleGridStrategy.__init__(grid_config)`
- **What**: Initializes a simple grid trading strategy for backtesting
- **Why**: Provides a reference strategy implementation
- **What for**: Grid trading simulation and testing
- **Example**:
```python
grid_config = {
    "num_grids": 20,
    "grid_spacing": Decimal("100"),
    "order_size": Decimal("100")
}
strategy = SimpleGridStrategy(grid_config)
```

#### `on_market_data(market_data)`
- **What**: Handles market data events in the strategy
- **Why**: Enables strategy reactions to price movements
- **What for**: Order placement and strategy decisions
- **Example**:
```python
# In strategy implementation
async def on_market_data(self, market_data):
    if self.should_place_order(market_data.price):
        await self.place_grid_orders()
```

#### `on_order_fill(order, fill_price)`
- **What**: Handles order fill events in the strategy
- **Why**: Enables strategy reactions to order executions
- **What for**: Position management and grid updates
- **Example**:
```python
# In strategy implementation
async def on_order_fill(self, order, fill_price):
    await self.update_grid_after_fill(order, fill_price)
    await self.place_opposite_order()
```

### evaluation.py

#### `PerformanceEvaluator.__init__()`
- **What**: Initializes the performance evaluation system
- **Why**: Provides comprehensive performance analysis capabilities
- **What for**: Strategy performance assessment and comparison
- **Example**:
```python
evaluator = PerformanceEvaluator()
```

#### `calculate_metrics(equity_curve, trades, benchmark=None)`
- **What**: Calculates comprehensive performance metrics from results
- **Why**: Provides detailed analysis of strategy performance
- **What for**: Performance assessment, optimization, and reporting
- **Example**:
```python
metrics = evaluator.calculate_metrics(
    equity_curve=equity_points,
    trades=trade_list,
    benchmark=benchmark_returns
)
# Returns: PerformanceMetrics with all calculated metrics
```

#### `calculate_sharpe_ratio(returns, risk_free_rate=0.02)`
- **What**: Calculates risk-adjusted return metric (Sharpe ratio)
- **Why**: Measures return per unit of risk taken
- **What for**: Risk-adjusted performance comparison
- **Example**:
```python
sharpe = evaluator.calculate_sharpe_ratio(
    returns=daily_returns,
    risk_free_rate=0.02
)
# Returns: Decimal("1.45") (Sharpe ratio)
```

#### `calculate_max_drawdown(equity_curve)`
- **What**: Calculates maximum drawdown from peak to trough
- **Why**: Measures worst-case loss scenario
- **What for**: Risk assessment and capital requirements
- **Example**:
```python
max_dd, duration = evaluator.calculate_max_drawdown(equity_curve)
# Returns: (Decimal("0.15"), 45) - 15% drawdown for 45 days
```

#### `calculate_calmar_ratio(annualized_return, max_drawdown)`
- **What**: Calculates annualized return divided by maximum drawdown
- **Why**: Measures return relative to worst-case scenario
- **What for**: Risk-adjusted performance evaluation
- **Example**:
```python
calmar = evaluator.calculate_calmar_ratio(
    annualized_return=Decimal("0.25"),
    max_drawdown=Decimal("0.10")
)
# Returns: Decimal("2.5") (25% return / 10% max drawdown)
```

#### `calculate_profit_factor(trades)`
- **What**: Calculates ratio of gross profit to gross loss
- **Why**: Measures trading system profitability
- **What for**: Strategy profitability assessment
- **Example**:
```python
profit_factor = evaluator.calculate_profit_factor(trade_list)
# Returns: Decimal("1.75") (profits are 1.75x losses)
```

#### `calculate_win_rate(trades)`
- **What**: Calculates percentage of profitable trades
- **Why**: Measures strategy consistency
- **What for**: Strategy reliability assessment
- **Example**:
```python
win_rate = evaluator.calculate_win_rate(trade_list)
# Returns: Decimal("0.65") (65% win rate)
```

#### `calculate_turnover(trades, average_capital)`
- **What**: Calculates portfolio turnover rate
- **Why**: Measures trading frequency and transaction costs
- **What for**: Cost analysis and strategy efficiency
- **Example**:
```python
turnover = evaluator.calculate_turnover(trades, Decimal("10000"))
# Returns: Decimal("5.2") (5.2x annual turnover)
```

#### `generate_performance_report(metrics, trades)`
- **What**: Generates comprehensive performance report
- **Why**: Provides detailed analysis summary
- **What for**: Strategy evaluation and presentation
- **Example**:
```python
report = evaluator.generate_performance_report(metrics, trades)
# Returns: Formatted performance report with all metrics
```

### slippage.py

#### `LinearSlippageModel.__init__(base_slippage=0.0005, volume_impact=0.00001)`
- **What**: Initializes linear slippage model with configurable parameters
- **Why**: Provides simple, predictable slippage calculation
- **What for**: Basic slippage simulation in backtesting
- **Example**:
```python
slippage_model = LinearSlippageModel(
    base_slippage=0.0005,  # 0.05% base slippage
    volume_impact=0.00001  # Volume impact factor
)
```

#### `calculate_slippage(context)`
- **What**: Calculates slippage based on order size and market conditions
- **Why**: Provides realistic execution cost simulation
- **What for**: Accurate backtesting and strategy evaluation
- **Example**:
```python
context = SlippageContext(
    order=buy_order,
    market_price=Decimal("50000"),
    volume=Decimal("100")
)
slippage = model.calculate_slippage(context)
# Returns: Decimal("0.001") (0.1% slippage)
```

#### `VolumeImpactModel.__init__(impact_coefficient=0.1, market_depth=10000)`
- **What**: Initializes volume impact slippage model
- **Why**: Simulates market impact based on order size
- **What for**: Realistic large order simulation
- **Example**:
```python
volume_model = VolumeImpactModel(
    impact_coefficient=0.1,
    market_depth=Decimal("10000")
)
```

#### `VolatilityBasedModel.__init__(base_slippage=0.0003, volatility_multiplier=2.0)`
- **What**: Initializes volatility-based slippage model
- **Why**: Adjusts slippage based on market volatility
- **What for**: Dynamic slippage simulation based on market conditions
- **Example**:
```python
vol_model = VolatilityBasedModel(
    base_slippage=0.0003,
    volatility_multiplier=2.0
)
```

#### `OrderBookModel.__init__(depth_levels=5, liquidity_threshold=1000)`
- **What**: Initializes order book-based slippage model
- **Why**: Uses order book depth for realistic slippage calculation
- **What for**: High-fidelity execution simulation
- **Example**:
```python
ob_model = OrderBookModel(
    depth_levels=5,
    liquidity_threshold=Decimal("1000")
)
```

#### `RandomSlippageModel.__init__(min_slippage=0.0001, max_slippage=0.002)`
- **What**: Initializes random slippage model with bounds
- **Why**: Introduces execution uncertainty in simulation
- **What for**: Stress testing and worst-case scenarios
- **Example**:
```python
random_model = RandomSlippageModel(
    min_slippage=0.0001,
    max_slippage=0.002
)
```

#### `get_execution_price(context)`
- **What**: Calculates final execution price including slippage
- **Why**: Provides realistic execution prices for simulation
- **What for**: Order execution simulation in backtesting
- **Example**:
```python
execution_price = model.get_execution_price(context)
# Returns: Decimal("50050") (market price + slippage)
```

#### `apply_bid_ask_spread(market_price, spread_pct=0.0002)`
- **What**: Applies bid-ask spread to market orders
- **Why**: Simulates market maker spread costs
- **What for**: Realistic market order execution
- **Example**:
```python
bid, ask = model.apply_bid_ask_spread(
    market_price=Decimal("50000"),
    spread_pct=0.0002
)
# Returns: (Decimal("49995"), Decimal("50005"))
```

#### `simulate_partial_fills(order, market_conditions)`
- **What**: Simulates partial order fills based on market liquidity
- **Why**: Models realistic order execution in illiquid markets
- **What for**: Liquidity-constrained execution simulation
- **Example**:
```python
fills = model.simulate_partial_fills(large_order, market_conditions)
# Returns: [{'quantity': 0.5, 'price': 50000}, {'quantity': 0.3, 'price': 50010}]
```
