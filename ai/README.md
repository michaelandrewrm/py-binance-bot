# AI Directory

## Purpose

The AI directory provides sophisticated machine learning and algorithmic trading capabilities, including baseline trading strategies, Bayesian optimization for parameter tuning, and comprehensive validation frameworks. This layer enables intelligent trading decisions and automated strategy optimization.

## Architecture

The AI layer provides:
- **Baseline Trading Strategies**: Technical analysis-based trading algorithms with configurable parameters
- **Bayesian Optimization**: Advanced parameter optimization using Gaussian processes and acquisition functions
- **Strategy Validation**: Comprehensive validation framework with statistical testing and cross-validation
- **Machine Learning Integration**: Support for custom ML models and feature engineering
- **Performance Analytics**: AI-driven performance analysis and strategy comparison

## Files Overview

### Core AI Components

#### `baseline.py` - Technical Analysis Trading Strategies
- **Technical Indicators**: Comprehensive technical analysis indicators (RSI, MACD, Bollinger Bands, SMA/EMA)
- **Trading Signals**: Multi-timeframe signal generation with confidence scoring
- **Strategy Framework**: Configurable trading strategies with parameter optimization
- **Risk Management**: Built-in risk controls and position sizing algorithms
- **Performance Tracking**: Real-time strategy performance monitoring and analytics

#### `bo_suggester.py` - Bayesian Optimization Engine
- **Gaussian Process Models**: Advanced probabilistic models for parameter space exploration
- **Acquisition Functions**: Multiple acquisition strategies (EI, UCB, PI) for optimal exploration
- **Hyperparameter Optimization**: Automated optimization of trading strategy parameters
- **Multi-objective Optimization**: Support for optimizing multiple conflicting objectives
- **Convergence Analysis**: Optimization progress tracking and convergence detection

#### `validation.py` - Strategy Validation Framework
- **Cross-Validation**: Time series cross-validation for strategy robustness testing
- **Statistical Testing**: Hypothesis testing and significance analysis for strategy performance
- **Overfitting Detection**: Methods to detect and prevent strategy overfitting
- **Walk-Forward Analysis**: Forward-looking validation with expanding/rolling windows
- **Performance Metrics**: Comprehensive validation metrics and statistical measures

## Key Functions and Classes

### Baseline Trading Strategies

#### Technical Indicator Framework
**`TechnicalAnalysis.__init__()`**
- **Purpose**: Initialize technical analysis framework with indicator calculations
- **Features**: Configurable indicators, multi-timeframe support, data validation
- **Indicators**: RSI, MACD, Bollinger Bands, SMA, EMA, Stochastic, Williams %R
- **Usage**: Foundation for all technical analysis-based trading strategies

**`TechnicalAnalysis.calculate_rsi(prices: List[float], period: int = 14) -> List[float]`**
- **Purpose**: Calculate Relative Strength Index for momentum analysis
- **Parameters**: Price series, RSI period (default 14)
- **Returns**: RSI values ranging from 0 to 100
- **Features**: Smoothed calculation, configurable period, overbought/oversold detection
- **Usage**: Momentum analysis and reversal signal generation

**`TechnicalAnalysis.calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float], List[float]]`**
- **Purpose**: Calculate MACD (Moving Average Convergence Divergence) indicator
- **Parameters**: Price series, fast EMA period, slow EMA period, signal line period
- **Returns**: Tuple of (MACD line, Signal line, Histogram)
- **Features**: Configurable periods, trend following, momentum analysis
- **Usage**: Trend identification and momentum change detection

**`TechnicalAnalysis.calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Tuple[List[float], List[float], List[float]]`**
- **Purpose**: Calculate Bollinger Bands for volatility and mean reversion analysis
- **Parameters**: Price series, moving average period, standard deviation multiplier
- **Returns**: Tuple of (Upper band, Middle band/SMA, Lower band)
- **Features**: Adaptive volatility bands, mean reversion signals, breakout detection
- **Usage**: Volatility analysis and mean reversion strategy signals

#### Trading Strategy Implementation
**`BaselineStrategy.__init__(config: Dict)`**
- **Purpose**: Initialize baseline trading strategy with configuration parameters
- **Parameters**: Strategy configuration dictionary with parameters and risk settings
- **Features**: Parameter validation, risk management setup, performance tracking
- **Usage**: Base class for all technical analysis-based trading strategies

**`BaselineStrategy.generate_signals(market_data: MarketData) -> List[TradingSignal]`**
- **Purpose**: Generate trading signals based on technical analysis
- **Parameters**: Current market data including OHLCV and indicators
- **Returns**: List of trading signals with confidence scores and metadata
- **Features**: Multi-indicator consensus, signal filtering, confidence weighting
- **Usage**: Primary signal generation for automated trading

**`BaselineStrategy.calculate_position_size(signal: TradingSignal, portfolio: Portfolio) -> Decimal`**
- **Purpose**: Calculate optimal position size based on signal strength and risk management
- **Parameters**: Trading signal with confidence, current portfolio state
- **Returns**: Position size as decimal amount
- **Features**: Kelly criterion, fixed fractional, volatility-based sizing
- **Usage**: Risk-adjusted position sizing for trade execution

#### Strategy Variants
**`MomentumStrategy.execute_strategy(data: MarketData) -> StrategyResult`**
- **Purpose**: Execute momentum-based trading strategy
- **Parameters**: Market data with price and volume information
- **Returns**: Strategy execution result with trades and performance metrics
- **Features**: Trend following, momentum confirmation, breakout trading
- **Usage**: Trend-following strategies in trending markets

**`MeanReversionStrategy.execute_strategy(data: MarketData) -> StrategyResult`**
- **Purpose**: Execute mean reversion trading strategy
- **Parameters**: Market data with statistical analysis
- **Returns**: Strategy execution result with reversal trades
- **Features**: Statistical mean reversion, oversold/overbought conditions
- **Usage**: Contrarian strategies in ranging markets

**`GridTradingStrategy.execute_strategy(data: MarketData) -> StrategyResult`**
- **Purpose**: Execute grid trading strategy with dynamic grid adjustment
- **Parameters**: Market data for grid level calculation
- **Returns**: Strategy execution result with grid-based trades
- **Features**: Dynamic grid sizing, profit taking, loss management
- **Usage**: Market-making strategies in sideways markets

### Bayesian Optimization System

#### Gaussian Process Modeling
**`BayesianOptimizer.__init__(objective_function: Callable, parameter_space: Dict, acquisition_function: str = "ei")`**
- **Purpose**: Initialize Bayesian optimization for strategy parameter tuning
- **Parameters**: Objective function to optimize, parameter search space, acquisition function type
- **Features**: Gaussian process regression, uncertainty quantification, exploration/exploitation balance
- **Usage**: Automated hyperparameter optimization for trading strategies

**`BayesianOptimizer.suggest_parameters() -> Dict[str, float]`**
- **Purpose**: Suggest next parameter configuration to evaluate
- **Returns**: Dictionary of parameter names and suggested values
- **Features**: Acquisition function optimization, uncertainty-aware suggestions
- **Usage**: Iterative parameter optimization process

**`BayesianOptimizer.update_model(parameters: Dict, performance: float) -> None`**
- **Purpose**: Update Gaussian process model with new evaluation result
- **Parameters**: Parameter configuration used, observed performance metric
- **Features**: Online learning, model uncertainty updates, convergence tracking
- **Usage**: Continuous model improvement during optimization

#### Acquisition Functions
**`BayesianOptimizer.expected_improvement(x: np.ndarray) -> float`**
- **Purpose**: Calculate Expected Improvement acquisition function
- **Parameters**: Parameter vector to evaluate
- **Returns**: Expected improvement value for exploration prioritization
- **Features**: Exploitation of promising regions, exploration of uncertain areas
- **Usage**: Balanced exploration-exploitation in parameter optimization

**`BayesianOptimizer.upper_confidence_bound(x: np.ndarray, kappa: float = 2.576) -> float`**
- **Purpose**: Calculate Upper Confidence Bound acquisition function
- **Parameters**: Parameter vector, exploration parameter kappa
- **Returns**: UCB value for optimistic parameter selection
- **Features**: Confidence interval-based exploration, tunable exploration level
- **Usage**: Optimistic parameter exploration with uncertainty quantification

**`BayesianOptimizer.probability_of_improvement(x: np.ndarray) -> float`**
- **Purpose**: Calculate Probability of Improvement acquisition function
- **Parameters**: Parameter vector to evaluate
- **Returns**: Probability of improvement over current best
- **Features**: Conservative exploration, improvement probability assessment
- **Usage**: Risk-averse parameter optimization

#### Multi-objective Optimization
**`MultiObjectiveOptimizer.optimize(objectives: List[str], weights: List[float]) -> OptimizationResult`**
- **Purpose**: Optimize multiple conflicting objectives simultaneously
- **Parameters**: List of objective names, importance weights for each objective
- **Returns**: Pareto-optimal solutions with trade-off analysis
- **Features**: Pareto frontier calculation, weighted objective combination
- **Usage**: Balancing return, risk, and other strategy objectives

**`MultiObjectiveOptimizer.pareto_frontier_analysis() -> ParetoAnalysis`**
- **Purpose**: Analyze Pareto frontier of multi-objective optimization
- **Returns**: Analysis of trade-offs between competing objectives
- **Features**: Domination analysis, efficient frontier identification
- **Usage**: Understanding objective trade-offs and solution selection

### Strategy Validation Framework

#### Cross-Validation Methods
**`StrategyValidator.__init__(validation_config: Dict)`**
- **Purpose**: Initialize validation framework with configuration
- **Parameters**: Validation configuration with methods and parameters
- **Features**: Multiple validation methods, statistical testing, robustness analysis
- **Usage**: Comprehensive strategy validation before deployment

**`StrategyValidator.time_series_cross_validation(strategy: Strategy, data: MarketData, folds: int = 5) -> ValidationResult`**
- **Purpose**: Perform time series cross-validation for strategy robustness
- **Parameters**: Strategy to validate, historical market data, number of folds
- **Returns**: Validation results with performance statistics across folds
- **Features**: Temporal order preservation, expanding/rolling windows, statistical analysis
- **Usage**: Robust strategy performance estimation and overfitting detection

**`StrategyValidator.walk_forward_analysis(strategy: Strategy, data: MarketData, train_period: int, test_period: int) -> WalkForwardResult`**
- **Purpose**: Perform walk-forward analysis for out-of-sample validation
- **Parameters**: Strategy, data, training period length, testing period length
- **Returns**: Walk-forward analysis results with performance evolution
- **Features**: Forward-looking validation, parameter stability analysis
- **Usage**: Real-world performance estimation and strategy degradation detection

#### Statistical Testing
**`StrategyValidator.statistical_significance_test(results_a: ValidationResult, results_b: ValidationResult) -> StatisticalTest`**
- **Purpose**: Test statistical significance between two strategy performances
- **Parameters**: Validation results for two strategies
- **Returns**: Statistical test results with p-values and confidence intervals
- **Features**: Multiple statistical tests, effect size calculation, power analysis
- **Usage**: Strategy comparison and selection validation

**`StrategyValidator.overfitting_detection(in_sample_results: ValidationResult, out_sample_results: ValidationResult) -> OverfittingAnalysis`**
- **Purpose**: Detect potential overfitting in strategy development
- **Parameters**: In-sample and out-of-sample validation results
- **Returns**: Overfitting analysis with severity assessment
- **Features**: Performance degradation analysis, complexity penalties
- **Usage**: Strategy robustness assessment and model selection

**`StrategyValidator.bootstrap_confidence_intervals(performance_series: List[float], confidence: float = 0.95) -> ConfidenceInterval`**
- **Purpose**: Calculate bootstrap confidence intervals for performance metrics
- **Parameters**: Performance time series, confidence level
- **Returns**: Confidence intervals for robust performance estimation
- **Features**: Non-parametric bootstrap, bias correction, multiple metrics
- **Usage**: Performance uncertainty quantification and risk assessment

## Integration Points

### With Core System
- **core/executor.py**: Strategy signal execution and trade management
- **core/safety.py**: AI-driven risk management and position sizing
- **core/state.py**: Strategy state persistence and recovery

### With Data Layer
- **data/loader.py**: Market data retrieval for technical analysis and model training
- **data/schema.py**: Data structures for signals, strategies, and optimization results

### With Storage Layer
- **storage/repo.py**: Strategy performance tracking and historical analysis
- **storage/artifacts.py**: Model storage, optimization results, and validation metrics

### With Simulation Layer
- **sim/engine.py**: Strategy backtesting and performance evaluation
- **sim/evaluation.py**: Validation integration with simulation framework

## Usage Examples

### Technical Analysis Strategy Implementation
```python
# Initialize technical analysis framework
ta = TechnicalAnalysis()

# Calculate technical indicators
prices = [100, 102, 101, 103, 105, 104, 106, 108]
rsi = ta.calculate_rsi(prices, period=14)
macd_line, signal_line, histogram = ta.calculate_macd(prices)
upper_bb, middle_bb, lower_bb = ta.calculate_bollinger_bands(prices)

# Create baseline strategy
strategy_config = {
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "macd_fast": 12,
    "macd_slow": 26,
    "position_sizing": "kelly",
    "max_position_size": 0.1
}

strategy = BaselineStrategy(strategy_config)

# Generate trading signals
market_data = MarketData(
    symbol="BTCUSDC",
    prices=prices,
    volumes=[1000, 1200, 800, 1500, 2000, 1100, 1800, 2200],
    timestamp=datetime.now()
)

signals = strategy.generate_signals(market_data)
for signal in signals:
    print(f"Signal: {signal.action} - Confidence: {signal.confidence:.2f}")
```

### Bayesian Optimization for Strategy Tuning
```python
# Define objective function for optimization
def strategy_objective(params):
    # Configure strategy with parameters
    strategy = BaselineStrategy({
        "rsi_period": int(params["rsi_period"]),
        "rsi_oversold": params["rsi_oversold"],
        "rsi_overbought": params["rsi_overbought"],
        "position_size": params["position_size"]
    })
    
    # Backtest strategy
    result = backtest_strategy(strategy, historical_data)
    
    # Return negative Sharpe ratio (for minimization)
    return -result.sharpe_ratio

# Define parameter space
parameter_space = {
    "rsi_period": {"type": "int", "bounds": [10, 20]},
    "rsi_oversold": {"type": "float", "bounds": [20, 40]},
    "rsi_overbought": {"type": "float", "bounds": [60, 80]},
    "position_size": {"type": "float", "bounds": [0.01, 0.1]}
}

# Initialize Bayesian optimizer
optimizer = BayesianOptimizer(
    objective_function=strategy_objective,
    parameter_space=parameter_space,
    acquisition_function="ei"
)

# Optimization loop
best_performance = float('inf')
best_params = None

for iteration in range(50):
    # Get suggested parameters
    suggested_params = optimizer.suggest_parameters()
    
    # Evaluate performance
    performance = strategy_objective(suggested_params)
    
    # Update model
    optimizer.update_model(suggested_params, performance)
    
    # Track best result
    if performance < best_performance:
        best_performance = performance
        best_params = suggested_params
        
    print(f"Iteration {iteration}: Performance = {-performance:.4f}")

print(f"Best Sharpe ratio: {-best_performance:.4f}")
print(f"Best parameters: {best_params}")
```

### Strategy Validation Workflow
```python
# Initialize validator
validator = StrategyValidator({
    "validation_method": "time_series_cv",
    "n_folds": 5,
    "test_size": 0.2,
    "statistical_tests": ["t_test", "mann_whitney"]
})

# Perform time series cross-validation
cv_results = validator.time_series_cross_validation(
    strategy=optimized_strategy,
    data=historical_data,
    folds=5
)

print(f"Mean CV Sharpe: {cv_results.mean_sharpe:.4f} ± {cv_results.std_sharpe:.4f}")
print(f"Mean CV Return: {cv_results.mean_return:.2%} ± {cv_results.std_return:.2%}")

# Walk-forward analysis
wf_results = validator.walk_forward_analysis(
    strategy=optimized_strategy,
    data=historical_data,
    train_period=252,  # 1 year training
    test_period=63     # 3 months testing
)

print(f"Walk-forward periods: {len(wf_results.periods)}")
print(f"Average out-of-sample Sharpe: {wf_results.avg_oos_sharpe:.4f}")

# Overfitting detection
overfitting_analysis = validator.overfitting_detection(
    in_sample_results=optimization_results,
    out_sample_results=wf_results
)

if overfitting_analysis.is_overfit:
    print(f"Warning: Potential overfitting detected!")
    print(f"Performance degradation: {overfitting_analysis.degradation:.2%}")
```

### Multi-objective Strategy Optimization
```python
# Define multiple objectives
def multi_objective_function(params):
    strategy = BaselineStrategy(params)
    result = backtest_strategy(strategy, historical_data)
    
    return {
        "return": -result.total_return,      # Maximize return (minimize negative)
        "risk": result.max_drawdown,         # Minimize risk
        "trades": -result.num_trades         # Maximize trade frequency
    }

# Multi-objective optimizer
mo_optimizer = MultiObjectiveOptimizer(
    objective_function=multi_objective_function,
    parameter_space=parameter_space
)

# Optimize with different objective weights
weight_combinations = [
    {"return": 0.6, "risk": 0.3, "trades": 0.1},
    {"return": 0.4, "risk": 0.5, "trades": 0.1},
    {"return": 0.5, "risk": 0.2, "trades": 0.3}
]

pareto_solutions = []
for weights in weight_combinations:
    solution = mo_optimizer.optimize(
        objectives=["return", "risk", "trades"],
        weights=[weights["return"], weights["risk"], weights["trades"]]
    )
    pareto_solutions.append(solution)

# Analyze Pareto frontier
pareto_analysis = mo_optimizer.pareto_frontier_analysis()
print(f"Efficient solutions found: {len(pareto_analysis.efficient_points)}")

# Select solution based on preferences
selected_solution = pareto_analysis.select_solution(
    risk_tolerance=0.15,  # Max 15% drawdown
    min_return=0.20       # Min 20% annual return
)
```

### Complete AI-Driven Strategy Development Pipeline
```python
async def develop_ai_strategy():
    # 1. Technical Analysis Setup
    ta = TechnicalAnalysis()
    
    # 2. Initial Strategy Configuration
    base_config = {
        "indicators": ["rsi", "macd", "bollinger"],
        "timeframes": ["1h", "4h", "1d"],
        "risk_management": True
    }
    
    # 3. Bayesian Optimization
    print("Starting parameter optimization...")
    optimizer = BayesianOptimizer(strategy_objective, parameter_space)
    
    optimized_params = await optimizer.optimize(iterations=100)
    print(f"Optimization complete. Best Sharpe: {optimized_params.best_score:.4f}")
    
    # 4. Strategy Validation
    print("Validating optimized strategy...")
    validator = StrategyValidator()
    
    validation_results = await validator.comprehensive_validation(
        strategy=optimized_strategy,
        data=historical_data
    )
    
    # 5. Performance Analysis
    if validation_results.is_robust:
        print("Strategy validation passed!")
        print(f"Expected Sharpe ratio: {validation_results.expected_sharpe:.4f}")
        print(f"95% confidence interval: [{validation_results.ci_lower:.4f}, {validation_results.ci_upper:.4f}]")
        
        # 6. Deploy for paper trading
        await deploy_strategy_for_testing(optimized_strategy)
    else:
        print("Strategy validation failed. Needs further development.")
        print(f"Issues found: {validation_results.issues}")
    
    return optimized_strategy, validation_results

# Run the complete pipeline
strategy, validation = await develop_ai_strategy()
```
