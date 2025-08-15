# Simulation Directory

## Purpose

The simulation directory provides comprehensive backtesting and simulation capabilities for trading strategies, including realistic market simulation, slippage modeling, and performance evaluation. This layer enables thorough strategy testing before live deployment.

## Architecture

The simulation layer provides:
- **Realistic Market Simulation**: Complete market environment simulation with order book dynamics
- **Advanced Slippage Modeling**: Sophisticated slippage calculation based on market conditions
- **Strategy Evaluation**: Comprehensive strategy testing with multiple performance metrics
- **Risk Assessment**: Simulation-based risk analysis and stress testing
- **Scenario Analysis**: What-if analysis and sensitivity testing

## Files Overview

### Core Simulation Components

#### `engine.py` - Simulation Engine (590 lines)
- **Market Simulation**: Complete trading environment simulation with realistic market dynamics
- **Order Execution**: Realistic order execution with market impact and timing considerations
- **Portfolio Management**: Real-time portfolio tracking during simulation runs
- **Event Processing**: Market event simulation including news, volatility, and liquidity changes
- **Performance Tracking**: Comprehensive performance measurement throughout simulation

#### `evaluation.py` - Strategy Evaluation (529 lines)
- **Strategy Testing**: Comprehensive strategy evaluation framework
- **Performance Metrics**: Multiple performance metrics calculation and analysis
- **Risk Analysis**: Simulation-based risk assessment and measurement
- **Comparison Framework**: Strategy comparison and ranking system
- **Statistical Analysis**: Statistical significance testing and confidence intervals

#### `slippage.py` - Market Impact Modeling (381 lines)
- **Slippage Calculation**: Sophisticated slippage modeling based on market conditions
- **Market Impact**: Order size impact on execution prices
- **Liquidity Modeling**: Dynamic liquidity calculation and impact assessment
- **Execution Cost**: Complete transaction cost analysis including fees and slippage
- **Market Microstructure**: Realistic market microstructure simulation

## Key Functions and Classes

### Simulation Engine

#### Core Simulation Framework
**`SimulationEngine.__init__(config: Dict, data_loader: DataLoader)`**
- **Purpose**: Initialize simulation engine with configuration and data source
- **Parameters**: Simulation configuration, market data loader
- **Features**: Environment setup, market state initialization, performance tracking
- **Usage**: Primary interface for strategy backtesting and simulation

**`SimulationEngine.run_simulation(strategy: Strategy, start_date: datetime, end_date: datetime) -> SimulationResult`**
- **Purpose**: Execute complete strategy simulation over specified period
- **Parameters**: Strategy to test, simulation start date, simulation end date
- **Returns**: Comprehensive simulation results with performance metrics
- **Features**: Multi-timeframe simulation, realistic execution, performance tracking
- **Usage**: Primary function for strategy backtesting and evaluation

**`SimulationEngine.process_market_event(event: MarketEvent) -> None`**
- **Purpose**: Process individual market events during simulation
- **Parameters**: Market event object (price update, trade, news, etc.)
- **Features**: Event routing, state updates, strategy notifications
- **Usage**: Core event processing for realistic market simulation

#### Portfolio Management
**`SimulationEngine.execute_order(order: OrderRequest) -> OrderResult`**
- **Purpose**: Execute trading orders within simulation environment
- **Parameters**: Order request with symbol, side, quantity, and type
- **Returns**: Order execution result with actual fill prices and quantities
- **Features**: Realistic execution, slippage calculation, market impact modeling
- **Usage**: Order execution during strategy backtesting

**`SimulationEngine.update_portfolio_state(trade: TradeData) -> None`**
- **Purpose**: Update portfolio state after trade execution
- **Parameters**: Executed trade data
- **Features**: Position tracking, P&L calculation, margin management
- **Usage**: Portfolio state management during simulation

**`SimulationEngine.calculate_unrealized_pnl() -> Dict[str, Decimal]`**
- **Purpose**: Calculate unrealized profit/loss for all positions
- **Returns**: Dictionary mapping symbols to unrealized P&L values
- **Features**: Mark-to-market calculation, real-time position valuation
- **Usage**: Risk monitoring and portfolio evaluation during simulation

#### Market State Simulation
**`SimulationEngine.simulate_market_conditions(timestamp: datetime) -> MarketState`**
- **Purpose**: Simulate realistic market conditions for given timestamp
- **Parameters**: Current simulation timestamp
- **Returns**: Complete market state including prices, spreads, and liquidity
- **Features**: Volatility modeling, spread simulation, liquidity dynamics
- **Usage**: Realistic market environment creation for strategy testing

**`SimulationEngine.generate_market_noise(base_price: Decimal, volatility: float) -> Decimal`**
- **Purpose**: Generate realistic market price noise for simulation
- **Parameters**: Base price level, market volatility parameter
- **Returns**: Price with realistic noise added
- **Features**: Statistical noise modeling, volatility clustering
- **Usage**: Realistic price movement simulation

### Strategy Evaluation Framework

#### Performance Analysis
**`StrategyEvaluator.__init__(simulation_engine: SimulationEngine)`**
- **Purpose**: Initialize strategy evaluator with simulation engine
- **Parameters**: Configured simulation engine instance
- **Features**: Metric calculation setup, comparison framework initialization
- **Usage**: Strategy performance evaluation and comparison

**`StrategyEvaluator.evaluate_strategy(strategy: Strategy, test_periods: List[Tuple[datetime, datetime]]) -> EvaluationResult`**
- **Purpose**: Comprehensive strategy evaluation across multiple time periods
- **Parameters**: Strategy to evaluate, list of test period tuples
- **Returns**: Complete evaluation result with performance metrics
- **Features**: Multi-period testing, statistical analysis, risk assessment
- **Usage**: Comprehensive strategy evaluation and validation

**`StrategyEvaluator.calculate_performance_metrics(returns: List[float]) -> PerformanceMetrics`**
- **Purpose**: Calculate comprehensive performance metrics from return series
- **Parameters**: List of strategy returns
- **Returns**: Complete performance metrics object
- **Features**: Risk-adjusted returns, drawdown analysis, statistical measures
- **Usage**: Performance measurement and strategy comparison

#### Risk Analysis
**`StrategyEvaluator.analyze_risk_metrics(returns: List[float], positions: List[Dict]) -> RiskAnalysis`**
- **Purpose**: Comprehensive risk analysis for strategy performance
- **Parameters**: Return series and position history
- **Returns**: Complete risk analysis with multiple risk measures
- **Features**: VaR calculation, stress testing, correlation analysis
- **Usage**: Risk assessment and strategy validation

**`StrategyEvaluator.perform_stress_testing(strategy: Strategy, scenarios: List[MarketScenario]) -> StressTestResult`**
- **Purpose**: Stress test strategy under extreme market conditions
- **Parameters**: Strategy to test, list of stress test scenarios
- **Returns**: Stress test results with scenario-specific performance
- **Features**: Multiple scenario testing, tail risk analysis
- **Usage**: Strategy robustness evaluation and risk management

**`StrategyEvaluator.calculate_value_at_risk(returns: List[float], confidence: float = 0.95) -> float`**
- **Purpose**: Calculate Value at Risk for strategy returns
- **Parameters**: Return series, confidence level (default 95%)
- **Returns**: VaR value at specified confidence level
- **Features**: Multiple VaR methodologies, historical and parametric approaches
- **Usage**: Risk measurement and capital allocation

#### Statistical Analysis
**`StrategyEvaluator.test_statistical_significance(strategy_a: EvaluationResult, strategy_b: EvaluationResult) -> StatisticalTest`**
- **Purpose**: Test statistical significance between two strategies
- **Parameters**: Evaluation results for two strategies
- **Returns**: Statistical test results with p-values and confidence intervals
- **Features**: Multiple statistical tests, bootstrap analysis
- **Usage**: Strategy comparison and selection validation

**`StrategyEvaluator.calculate_confidence_intervals(metrics: PerformanceMetrics, confidence: float = 0.95) -> Dict[str, Tuple[float, float]]`**
- **Purpose**: Calculate confidence intervals for performance metrics
- **Parameters**: Performance metrics, confidence level
- **Returns**: Dictionary mapping metric names to confidence intervals
- **Features**: Bootstrap methodology, robust interval estimation
- **Usage**: Performance metric reliability assessment

### Slippage and Market Impact Modeling

#### Slippage Calculation
**`SlippageModel.__init__(market_data: MarketData, config: Dict)`**
- **Purpose**: Initialize slippage model with market data and configuration
- **Parameters**: Market data source, slippage model configuration
- **Features**: Market impact modeling, liquidity analysis setup
- **Usage**: Realistic execution cost modeling for simulation

**`SlippageModel.calculate_slippage(order: OrderRequest, market_state: MarketState) -> SlippageResult`**
- **Purpose**: Calculate realistic slippage for order execution
- **Parameters**: Order request, current market state
- **Returns**: Slippage calculation with execution price adjustment
- **Features**: Size-dependent slippage, market condition impact, timing effects
- **Usage**: Realistic order execution modeling in simulation

**`SlippageModel.estimate_market_impact(quantity: Decimal, symbol: str, market_depth: Dict) -> Decimal`**
- **Purpose**: Estimate market impact of large orders
- **Parameters**: Order quantity, trading symbol, current market depth
- **Returns**: Estimated price impact from order execution
- **Features**: Non-linear impact modeling, liquidity consideration
- **Usage**: Large order execution cost estimation

#### Liquidity Modeling
**`SlippageModel.calculate_effective_spread(symbol: str, timestamp: datetime) -> Decimal`**
- **Purpose**: Calculate effective bid-ask spread for trading
- **Parameters**: Trading symbol, timestamp for market conditions
- **Returns**: Effective spread considering market conditions
- **Features**: Time-of-day effects, volatility impact, volume consideration
- **Usage**: Execution cost calculation and strategy evaluation

**`SlippageModel.model_liquidity_dynamics(symbol: str, time_period: Tuple[datetime, datetime]) -> LiquidityProfile`**
- **Purpose**: Model liquidity changes over time for realistic simulation
- **Parameters**: Trading symbol, time period for analysis
- **Returns**: Liquidity profile with time-varying characteristics
- **Features**: Intraday patterns, volatility impact, event-driven changes
- **Usage**: Realistic market condition simulation

#### Transaction Cost Analysis
**`SlippageModel.calculate_total_execution_cost(order: OrderRequest, execution_result: OrderResult) -> ExecutionCost`**
- **Purpose**: Calculate complete execution cost including all components
- **Parameters**: Original order request, actual execution result
- **Returns**: Complete cost breakdown with fees, slippage, and impact
- **Features**: Comprehensive cost analysis, benchmark comparison
- **Usage**: Strategy evaluation and execution quality assessment

## Integration Points

### With Core System
- **core/executor.py**: Execution strategy testing and validation
- **core/safety.py**: Risk management system testing and calibration
- **core/state.py**: State management simulation and testing

### With AI Layer
- **ai/baseline.py**: Strategy backtesting and performance evaluation
- **ai/bo_suggester.py**: Optimization parameter testing and validation
- **ai/validation.py**: Strategy validation and robustness testing

### With Data Layer
- **data/loader.py**: Historical market data for simulation
- **data/schema.py**: Data structures for simulation results and analysis

## Usage Examples

### Basic Strategy Simulation
```python
# Initialize simulation engine
engine = SimulationEngine(
    config=simulation_config,
    data_loader=data_loader
)

# Define strategy to test
strategy = GridTradingStrategy(
    grid_spacing=0.01,
    position_size=0.1,
    max_positions=10
)

# Run simulation
result = await engine.run_simulation(
    strategy=strategy,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31)
)

# Analyze results
print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")
print(f"Win Rate: {result.win_rate:.1%}")
```

### Comprehensive Strategy Evaluation
```python
# Initialize evaluator
evaluator = StrategyEvaluator(engine)

# Define test periods
test_periods = [
    (datetime(2024, 1, 1), datetime(2024, 3, 31)),   # Q1
    (datetime(2024, 4, 1), datetime(2024, 6, 30)),   # Q2
    (datetime(2024, 7, 1), datetime(2024, 9, 30)),   # Q3
    (datetime(2024, 10, 1), datetime(2024, 12, 31))  # Q4
]

# Evaluate strategy
evaluation = await evaluator.evaluate_strategy(strategy, test_periods)

# Performance metrics
metrics = evaluation.performance_metrics
print(f"Annualized Return: {metrics.annualized_return:.2%}")
print(f"Volatility: {metrics.volatility:.2%}")
print(f"Information Ratio: {metrics.information_ratio:.2f}")

# Risk analysis
risk_analysis = evaluation.risk_analysis
print(f"VaR (95%): {risk_analysis.var_95:.2%}")
print(f"CVaR (95%): {risk_analysis.cvar_95:.2%}")
print(f"Maximum Drawdown: {risk_analysis.max_drawdown:.2%}")
```

### Slippage and Market Impact Analysis
```python
# Initialize slippage model
slippage_model = SlippageModel(
    market_data=market_data,
    config=slippage_config
)

# Calculate slippage for order
order = OrderRequest(
    symbol="BTCUSDC",
    side="BUY",
    quantity=Decimal("1.0"),
    order_type="MARKET"
)

market_state = await engine.get_current_market_state("BTCUSDC")
slippage_result = slippage_model.calculate_slippage(order, market_state)

print(f"Expected Slippage: {slippage_result.slippage_bps:.1f} bps")
print(f"Execution Price: ${slippage_result.execution_price:.2f}")
print(f"Market Impact: ${slippage_result.market_impact:.2f}")

# Total execution cost analysis
execution_cost = slippage_model.calculate_total_execution_cost(
    order, slippage_result
)
print(f"Total Cost: {execution_cost.total_cost_bps:.1f} bps")
print(f"  - Exchange Fees: {execution_cost.fees_bps:.1f} bps")
print(f"  - Slippage: {execution_cost.slippage_bps:.1f} bps")
print(f"  - Market Impact: {execution_cost.impact_bps:.1f} bps")
```

### Strategy Comparison and Validation
```python
# Compare multiple strategies
strategies = [
    GridTradingStrategy(grid_spacing=0.01),
    GridTradingStrategy(grid_spacing=0.02),
    MomentumStrategy(lookback=20),
    MeanReversionStrategy(threshold=2.0)
]

# Evaluate all strategies
results = []
for strategy in strategies:
    result = await evaluator.evaluate_strategy(strategy, test_periods)
    results.append((strategy.name, result))

# Rank strategies by Sharpe ratio
ranked_strategies = sorted(
    results, 
    key=lambda x: x[1].performance_metrics.sharpe_ratio, 
    reverse=True
)

for rank, (name, result) in enumerate(ranked_strategies, 1):
    print(f"{rank}. {name}: Sharpe {result.performance_metrics.sharpe_ratio:.2f}")

# Statistical significance testing
best_strategy = ranked_strategies[0][1]
second_strategy = ranked_strategies[1][1]

stat_test = evaluator.test_statistical_significance(
    best_strategy, second_strategy
)
print(f"P-value: {stat_test.p_value:.4f}")
print(f"Significant: {stat_test.is_significant}")
```

### Stress Testing and Scenario Analysis
```python
# Define stress test scenarios
scenarios = [
    MarketScenario(name="Market Crash", volatility_multiplier=3.0, return_shift=-0.20),
    MarketScenario(name="High Volatility", volatility_multiplier=2.0, return_shift=0.0),
    MarketScenario(name="Bull Market", volatility_multiplier=0.5, return_shift=0.15),
    MarketScenario(name="Bear Market", volatility_multiplier=1.5, return_shift=-0.10)
]

# Perform stress testing
stress_results = await evaluator.perform_stress_testing(strategy, scenarios)

for scenario_name, result in stress_results.results.items():
    print(f"{scenario_name}:")
    print(f"  Return: {result.total_return:.2%}")
    print(f"  Max Drawdown: {result.max_drawdown:.2%}")
    print(f"  VaR (95%): {result.var_95:.2%}")

# Overall stress test summary
print(f"Worst Case Return: {stress_results.worst_case_return:.2%}")
print(f"Best Case Return: {stress_results.best_case_return:.2%}")
print(f"Average Stress Return: {stress_results.average_return:.2%}")
```

### Advanced Analysis and Reporting
```python
# Generate comprehensive simulation report
report_data = {
    "simulation_results": result,
    "performance_metrics": metrics,
    "risk_analysis": risk_analysis,
    "slippage_analysis": slippage_result,
    "stress_test_results": stress_results,
    "statistical_tests": stat_test
}

# Export detailed report
report_path = await export_simulation_report(
    report_data, 
    format="pdf",
    include_charts=True
)
print(f"Simulation report saved to: {report_path}")

# Save simulation results for future analysis
await save_simulation_results(result, "grid_strategy_2024_backtest.json")
```
