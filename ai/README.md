# AI Directory

## File Overview

This directory contains artificial intelligence and machine learning modules for trading strategy optimization and analysis.

### Python Files

#### `baseline.py`
- **Purpose**: Baseline trading strategies using traditional technical indicators
- **Functionality**: Implements ATR/volatility bands, moving averages, and other technical analysis tools

#### `bo_suggester.py`
- **Purpose**: Bayesian optimization for strategy hyperparameter tuning
- **Functionality**: Optimizes trading strategy parameters using Bayesian optimization over simulation results

#### `validation.py`
- **Purpose**: Strategy validation using walk-forward analysis and rolling cross-validation
- **Functionality**: Prevents overfitting and validates strategy robustness using time-series cross-validation

## Function Documentation

### baseline.py

#### `TechnicalIndicators.sma(prices, period)`
- **What**: Calculates Simple Moving Average over specified period
- **Why**: Provides trend identification and smoothing for price analysis
- **What for**: Trend following strategies and signal generation
- **Example**:
```python
prices = [Decimal("50000"), Decimal("50100"), Decimal("49900"), Decimal("50200")]
sma_values = TechnicalIndicators.sma(prices, period=3)
# Returns: [None, None, Decimal("50000"), Decimal("50067")]
```

#### `TechnicalIndicators.ema(prices, period, alpha=None)`
- **What**: Calculates Exponential Moving Average with configurable smoothing factor
- **Why**: Provides more responsive trend following compared to SMA
- **What for**: Responsive trend detection and signal generation
- **Example**:
```python
prices = [Decimal("100"), Decimal("102"), Decimal("101"), Decimal("103")]
ema_values = TechnicalIndicators.ema(prices, period=3)
# Returns: [100, 101, 101, 102] (more responsive to recent prices)
```

#### `TechnicalIndicators.atr(highs, lows, closes, period=14)`
- **What**: Calculates Average True Range for volatility measurement
- **Why**: Measures market volatility for position sizing and stop-loss placement
- **What for**: Volatility-based position sizing and risk management
- **Example**:
```python
highs = [Decimal("50100"), Decimal("50200"), Decimal("50150")]
lows = [Decimal("49900"), Decimal("49950"), Decimal("49800")]
closes = [Decimal("50000"), Decimal("50100"), Decimal("50000")]
atr_values = TechnicalIndicators.atr(highs, lows, closes, period=2)
# Returns: [None, Decimal("175"), Decimal("162.5")]
```

#### `TechnicalIndicators.bollinger_bands(prices, period=20, std_dev=2)`
- **What**: Calculates Bollinger Bands for volatility-based trading ranges
- **Why**: Identifies overbought/oversold conditions and volatility changes
- **What for**: Mean reversion strategies and volatility breakout detection
- **Example**:
```python
prices = [Decimal("50000")] * 20 + [Decimal("50500"), Decimal("49500")]
upper, middle, lower = TechnicalIndicators.bollinger_bands(prices, period=20)
# Returns: (upper_band, sma, lower_band) arrays
```

#### `TechnicalIndicators.rsi(prices, period=14)`
- **What**: Calculates Relative Strength Index for momentum analysis
- **Why**: Identifies overbought/oversold conditions in price momentum
- **What for**: Mean reversion and momentum strategies
- **Example**:
```python
prices = [Decimal("50000"), Decimal("50100"), Decimal("50200"), Decimal("50150")]
rsi_values = TechnicalIndicators.rsi(prices, period=3)
# Returns: [None, None, Decimal("100"), Decimal("66.67")]
```

#### `BaselineStrategy.__init__(config)`
- **What**: Initializes baseline strategy with technical indicator configuration
- **Why**: Provides a simple, robust trading strategy for comparison
- **What for**: Baseline performance comparison and fallback strategy
- **Example**:
```python
config = {
    "atr_period": 14,
    "atr_multiplier": 2.0,
    "sma_period": 20,
    "position_size": Decimal("100")
}
strategy = BaselineStrategy(config)
```

#### `generate_signals(klines)`
- **What**: Generates buy/sell signals based on technical indicators
- **Why**: Provides systematic trading decisions based on market analysis
- **What for**: Automated trading signal generation
- **Example**:
```python
signals = strategy.generate_signals(klines_data)
# Returns: [{'timestamp': dt, 'signal': 'BUY', 'price': 50000, 'confidence': 0.8}]
```

#### `calculate_position_size(current_price, atr_value, risk_amount)`
- **What**: Calculates position size based on volatility and risk tolerance
- **Why**: Ensures consistent risk management across different market conditions
- **What for**: Risk-adjusted position sizing in volatile markets
- **Example**:
```python
position_size = strategy.calculate_position_size(
    current_price=Decimal("50000"),
    atr_value=Decimal("500"),
    risk_amount=Decimal("100")
)
# Returns: Decimal("0.004") (position size in base currency)
```

### bo_suggester.py

#### `BayesianOptimizer.__init__(hyperparameters, objective_function, n_initial_points=10)`
- **What**: Initializes Bayesian optimizer with parameter space and objective function
- **Why**: Enables efficient hyperparameter optimization using probabilistic models
- **What for**: Strategy optimization and parameter tuning
- **Example**:
```python
hyperparams = [
    HyperParameter("grid_spacing", "float", bounds=(50, 500)),
    HyperParameter("num_grids", "int", bounds=(10, 50)),
    HyperParameter("risk_limit", "float", bounds=(0.01, 0.05))
]
optimizer = BayesianOptimizer(hyperparams, objective_function)
```

#### `optimize(n_trials=50, progress_callback=None)`
- **What**: Runs Bayesian optimization to find optimal hyperparameters
- **Why**: Finds optimal parameters more efficiently than grid search
- **What for**: Strategy optimization and performance improvement
- **Example**:
```python
def objective(params):
    # Run backtest with parameters
    result = run_backtest(params)
    return result.sharpe_ratio

best_result = await optimizer.optimize(n_trials=100)
# Returns: OptimizationResult with best parameters and performance
```

#### `suggest_next_parameters()`
- **What**: Suggests next parameter combination to evaluate using acquisition function
- **Why**: Balances exploration and exploitation in parameter space
- **What for**: Efficient parameter space exploration
- **Example**:
```python
next_params = optimizer.suggest_next_parameters()
# Returns: {'grid_spacing': 125.5, 'num_grids': 25, 'risk_limit': 0.023}
```

#### `update_with_result(parameters, objective_value, metrics)`
- **What**: Updates optimizer with new trial results
- **Why**: Improves model understanding of parameter space
- **What for**: Continuous learning and optimization refinement
- **Example**:
```python
optimizer.update_with_result(
    parameters={"grid_spacing": 100, "num_grids": 20},
    objective_value=1.45,
    metrics={"return": 0.25, "drawdown": 0.08}
)
```

#### `get_optimization_history()`
- **What**: Retrieves complete optimization history and convergence data
- **Why**: Provides visibility into optimization process and convergence
- **What for**: Optimization analysis and debugging
- **Example**:
```python
history = optimizer.get_optimization_history()
# Returns: List[OptimizationResult] with all trials and their results
```

#### `plot_convergence()`
- **What**: Generates convergence plot showing optimization progress
- **Why**: Visualizes optimization effectiveness and convergence behavior
- **What for**: Optimization analysis and presentation
- **Example**:
```python
fig = optimizer.plot_convergence()
# Returns: Matplotlib figure showing objective value over trials
```

### validation.py

#### `WalkForwardValidator.__init__(train_window_days=252, test_window_days=63, step_days=21)`
- **What**: Initializes walk-forward validation with time window configuration
- **Why**: Prevents data leakage and validates strategy robustness over time
- **What for**: Realistic strategy validation and overfitting detection
- **Example**:
```python
validator = WalkForwardValidator(
    train_window_days=252,  # 1 year training
    test_window_days=63,    # 3 months testing
    step_days=21           # Monthly steps
)
```

#### `create_validation_periods(start_date, end_date)`
- **What**: Creates non-overlapping train/test periods for walk-forward analysis
- **Why**: Ensures proper temporal validation without future information
- **What for**: Time-series cross-validation setup
- **Example**:
```python
periods = validator.create_validation_periods(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 31)
)
# Returns: [ValidationPeriod(train_start=..., test_start=...), ...]
```

#### `validate_strategy(strategy, data, optimization_function=None)`
- **What**: Runs complete walk-forward validation on a trading strategy
- **Why**: Provides realistic assessment of strategy performance
- **What for**: Strategy validation and robustness testing
- **Example**:
```python
def optimize_params(train_data):
    # Optimize parameters on training data
    return optimized_parameters

results = await validator.validate_strategy(
    strategy=my_strategy,
    data=historical_data,
    optimization_function=optimize_params
)
# Returns: List[ValidationResult] for each validation period
```

#### `calculate_overfitting_score(train_metrics, test_metrics)`
- **What**: Calculates overfitting score by comparing train vs test performance
- **Why**: Quantifies the degree of overfitting in strategy optimization
- **What for**: Overfitting detection and model selection
- **Example**:
```python
overfitting_score = validator.calculate_overfitting_score(
    train_metrics=train_performance,
    test_metrics=test_performance
)
# Returns: 0.15 (15% performance degradation from train to test)
```

#### `RollingValidator.rolling_validate(strategy, data, window_size, step_size)`
- **What**: Performs rolling window validation with overlapping periods
- **Why**: Provides more validation samples for statistical significance
- **What for**: Robust strategy validation with multiple test periods
- **Example**:
```python
rolling_validator = RollingValidator()
results = rolling_validator.rolling_validate(
    strategy=strategy,
    data=data,
    window_size=timedelta(days=180),
    step_size=timedelta(days=30)
)
# Returns: Multiple validation results from overlapping windows
```

#### `generate_validation_report(validation_results)`
- **What**: Generates comprehensive validation report with statistics
- **Why**: Provides summary analysis of validation results
- **What for**: Strategy assessment and decision making
- **Example**:
```python
report = validator.generate_validation_report(validation_results)
# Returns: {'avg_test_sharpe': 1.25, 'consistency_score': 0.82, ...}
```

#### `CrossValidator.cross_validate(strategy, data, k_folds=5)`
- **What**: Performs k-fold cross-validation adapted for time series
- **Why**: Provides robust validation while respecting temporal order
- **What for**: Model validation and parameter selection
- **Example**:
```python
cv = CrossValidator()
cv_results = cv.cross_validate(
    strategy=strategy,
    data=data,
    k_folds=5
)
# Returns: Cross-validation results with performance metrics
```

#### `calculate_consistency_score(validation_results)`
- **What**: Calculates strategy consistency across different time periods
- **Why**: Measures strategy robustness across varying market conditions
- **What for**: Strategy reliability assessment
- **Example**:
```python
consistency = validator.calculate_consistency_score(validation_results)
# Returns: 0.85 (85% consistency score across validation periods)
```
