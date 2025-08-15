# Reports Directory

## Purpose

The reports directory provides comprehensive performance analysis and reporting capabilities, including end-of-run summaries, HODL benchmarking, and detailed trading analytics. This layer enables thorough evaluation of trading strategies and portfolio performance.

## Architecture

The reports layer provides:
- **Performance Analytics**: Comprehensive trading performance measurement and analysis
- **Benchmarking**: HODL strategy comparison and relative performance evaluation
- **Risk Analysis**: Drawdown analysis, volatility measurement, and risk-adjusted returns
- **Report Generation**: Automated report creation with visualizations and insights
- **Export Capabilities**: Multiple output formats for analysis and presentation

## Files Overview

### Core Reporting Components

#### `end_of_run.py` - Comprehensive Performance Reporting (865 lines)
- **End-of-Run Analysis**: Complete trading session performance evaluation
- **Risk Metrics**: Comprehensive risk analysis including VaR, Sharpe ratio, and drawdown
- **Trade Analytics**: Detailed trade-by-trade analysis with win/loss statistics
- **Portfolio Tracking**: Complete portfolio evolution and performance attribution
- **Visualization**: Charts and graphs for performance visualization and reporting

#### `hodl_benchmark.py` - HODL Comparison Analysis (483 lines)
- **HODL Strategy**: Buy-and-hold benchmark calculation and comparison
- **Relative Performance**: Active trading vs passive holding performance analysis
- **Risk-Adjusted Comparison**: Sharpe ratio and volatility-adjusted performance metrics
- **Market Analysis**: Market condition impact on strategy performance
- **Alpha Generation**: Active management value-add measurement

## Key Functions and Classes

### End-of-Run Performance Analysis

#### Core Performance Analysis
**`EndOfRunReporter.__init__(trade_repo: TradeRepo, start_date: datetime, end_date: datetime)`**
- **Purpose**: Initialize performance reporter with data source and analysis period
- **Parameters**: Trade repository, analysis start date, analysis end date
- **Features**: Data validation, period calculation, metric initialization
- **Usage**: Primary interface for comprehensive performance analysis

**`EndOfRunReporter.generate_performance_report() -> PerformanceReport`**
- **Purpose**: Generate comprehensive performance analysis report
- **Returns**: Complete performance report with all metrics and analysis
- **Features**: Multi-timeframe analysis, risk metrics, trade statistics
- **Components**: Returns, volatility, drawdown, Sharpe ratio, trade analysis
- **Usage**: Primary function for end-of-session performance evaluation

**`EndOfRunReporter.calculate_returns_metrics() -> ReturnsMetrics`**
- **Purpose**: Calculate comprehensive return metrics for the trading period
- **Returns**: ReturnsMetrics object with total, annualized, and risk-adjusted returns
- **Features**: Multiple return calculations, benchmark comparison, time-weighted returns
- **Metrics**: Total return, CAGR, monthly returns, rolling returns
- **Usage**: Core performance measurement and comparison

#### Risk Analysis
**`EndOfRunReporter.calculate_risk_metrics() -> RiskMetrics`**
- **Purpose**: Comprehensive risk analysis for the trading strategy
- **Returns**: RiskMetrics object with volatility, VaR, and drawdown analysis
- **Features**: Multiple risk measures, confidence intervals, stress testing
- **Metrics**: Standard deviation, VaR, CVaR, maximum drawdown, Calmar ratio
- **Usage**: Risk assessment and strategy evaluation

**`EndOfRunReporter.analyze_drawdown_periods() -> List[DrawdownPeriod]`**
- **Purpose**: Identify and analyze all drawdown periods in detail
- **Returns**: List of drawdown periods with duration and recovery analysis
- **Features**: Peak-to-trough analysis, recovery time calculation, frequency analysis
- **Usage**: Risk management evaluation and strategy robustness assessment

**`EndOfRunReporter.calculate_sharpe_ratio(risk_free_rate: float = 0.02) -> float`**
- **Purpose**: Calculate risk-adjusted return using Sharpe ratio
- **Parameters**: Risk-free rate for calculation (default 2%)
- **Returns**: Sharpe ratio value
- **Features**: Annualized calculation, excess return measurement
- **Usage**: Risk-adjusted performance comparison and strategy evaluation

#### Trade Analysis
**`EndOfRunReporter.analyze_trade_statistics() -> TradeStatistics`**
- **Purpose**: Detailed analysis of individual trade performance
- **Returns**: Comprehensive trade statistics object
- **Features**: Win/loss analysis, trade distribution, holding period analysis
- **Metrics**: Win rate, average win/loss, profit factor, expectancy
- **Usage**: Strategy effectiveness evaluation and optimization insights

**`EndOfRunReporter.calculate_monthly_returns() -> Dict[str, float]`**
- **Purpose**: Break down returns by month for seasonal analysis
- **Returns**: Dictionary mapping months to return percentages
- **Features**: Seasonality analysis, consistency measurement
- **Usage**: Performance consistency evaluation and seasonal pattern identification

#### Visualization and Export
**`EndOfRunReporter.generate_performance_charts() -> Dict[str, Any]`**
- **Purpose**: Create comprehensive performance visualization charts
- **Returns**: Dictionary of chart data and configurations
- **Features**: Multiple chart types, interactive elements, export options
- **Charts**: Equity curve, drawdown chart, monthly returns, trade distribution
- **Usage**: Visual performance analysis and presentation

**`EndOfRunReporter.export_detailed_report(format: str = "pdf") -> str`**
- **Purpose**: Export comprehensive report in specified format
- **Parameters**: Output format (pdf, html, xlsx)
- **Returns**: File path of generated report
- **Features**: Multiple formats, charts inclusion, professional formatting
- **Usage**: Report distribution and archival

### HODL Benchmark Analysis

#### Benchmark Calculation
**`HODLBenchmark.__init__(symbol: str, start_date: datetime, end_date: datetime, initial_capital: float)`**
- **Purpose**: Initialize HODL benchmark with trading parameters
- **Parameters**: Trading symbol, analysis period, initial capital amount
- **Features**: Historical price retrieval, capital allocation, period validation
- **Usage**: Benchmark initialization for performance comparison

**`HODLBenchmark.calculate_hodl_performance() -> HODLPerformance`**
- **Purpose**: Calculate buy-and-hold strategy performance
- **Returns**: Complete HODL performance metrics
- **Features**: Simple buy-and-hold simulation, fee calculation, return analysis
- **Metrics**: Total return, CAGR, volatility, maximum drawdown
- **Usage**: Baseline performance calculation for strategy comparison

**`HODLBenchmark.compare_with_active_trading(active_returns: List[float]) -> ComparisonReport`**
- **Purpose**: Compare HODL performance with active trading strategy
- **Parameters**: Active trading return series
- **Returns**: Detailed comparison report
- **Features**: Relative performance, risk-adjusted comparison, alpha calculation
- **Usage**: Strategy evaluation and performance attribution

#### Market Analysis
**`HODLBenchmark.analyze_market_conditions() -> MarketAnalysis`**
- **Purpose**: Analyze market conditions during the trading period
- **Returns**: Market condition analysis with trend and volatility metrics
- **Features**: Trend identification, volatility regimes, correlation analysis
- **Usage**: Context for performance evaluation and strategy effectiveness

**`HODLBenchmark.calculate_alpha_beta(market_returns: List[float]) -> Tuple[float, float]`**
- **Purpose**: Calculate alpha and beta relative to market benchmark
- **Parameters**: Market return series for comparison
- **Returns**: Tuple of alpha and beta values
- **Features**: Regression analysis, risk-adjusted performance measurement
- **Usage**: Performance attribution and risk assessment

#### Risk-Adjusted Comparison
**`HODLBenchmark.compare_sharpe_ratios() -> Dict[str, float]`**
- **Purpose**: Compare Sharpe ratios between HODL and active strategies
- **Returns**: Dictionary with Sharpe ratios for both strategies
- **Features**: Risk-adjusted performance comparison, statistical significance
- **Usage**: Strategy evaluation and risk-adjusted performance assessment

**`HODLBenchmark.analyze_volatility_comparison() -> VolatilityAnalysis`**
- **Purpose**: Compare volatility characteristics between strategies
- **Returns**: Comprehensive volatility analysis
- **Features**: Rolling volatility, volatility clustering, risk comparison
- **Usage**: Risk profile comparison and strategy selection

## Integration Points

### With Storage Layer
- **storage/repo.py**: Trade data retrieval for performance calculation
- **storage/artifacts.py**: Report storage and historical comparison

### With Data Layer
- **data/loader.py**: Market data for benchmark calculation and analysis
- **data/schema.py**: Data structures for performance metrics and reporting

### With Core System
- **core/executor.py**: Trade execution data for performance analysis
- **core/safety.py**: Risk metrics integration and safety analysis

## Usage Examples

### End-of-Run Performance Analysis
```python
# Initialize performance reporter
reporter = EndOfRunReporter(
    trade_repo=repo,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31)
)

# Generate comprehensive performance report
performance_report = await reporter.generate_performance_report()

# Access key metrics
print(f"Total Return: {performance_report.total_return:.2%}")
print(f"Sharpe Ratio: {performance_report.sharpe_ratio:.2f}")
print(f"Max Drawdown: {performance_report.max_drawdown:.2%}")
print(f"Win Rate: {performance_report.win_rate:.1%}")

# Detailed risk analysis
risk_metrics = await reporter.calculate_risk_metrics()
print(f"Volatility: {risk_metrics.annual_volatility:.2%}")
print(f"VaR (95%): {risk_metrics.var_95:.2%}")
print(f"Calmar Ratio: {risk_metrics.calmar_ratio:.2f}")

# Export detailed report
report_path = await reporter.export_detailed_report(format="pdf")
print(f"Report saved to: {report_path}")
```

### HODL Benchmark Comparison
```python
# Initialize HODL benchmark
hodl_benchmark = HODLBenchmark(
    symbol="BTCUSDC",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    initial_capital=10000.0
)

# Calculate HODL performance
hodl_performance = await hodl_benchmark.calculate_hodl_performance()
print(f"HODL Return: {hodl_performance.total_return:.2%}")
print(f"HODL Volatility: {hodl_performance.volatility:.2%}")

# Compare with active trading
active_returns = await get_active_trading_returns()
comparison = await hodl_benchmark.compare_with_active_trading(active_returns)

print(f"Active vs HODL: {comparison.excess_return:.2%}")
print(f"Information Ratio: {comparison.information_ratio:.2f}")
print(f"Alpha: {comparison.alpha:.2%}")
```

### Comprehensive Analysis Workflow
```python
async def generate_complete_analysis():
    # End-of-run analysis
    reporter = EndOfRunReporter(repo, start_date, end_date)
    performance_report = await reporter.generate_performance_report()
    
    # HODL benchmark
    hodl = HODLBenchmark("BTCUSDC", start_date, end_date, 10000)
    hodl_performance = await hodl.calculate_hodl_performance()
    
    # Comparison analysis
    comparison = await hodl.compare_with_active_trading(
        performance_report.daily_returns
    )
    
    # Generate visualizations
    charts = await reporter.generate_performance_charts()
    
    # Export comprehensive report
    report_data = {
        "performance": performance_report,
        "hodl_benchmark": hodl_performance,
        "comparison": comparison,
        "charts": charts
    }
    
    return await export_comprehensive_report(report_data)
```

### Trade Analysis Deep Dive
```python
# Detailed trade statistics
trade_stats = await reporter.analyze_trade_statistics()
print(f"Total Trades: {trade_stats.total_trades}")
print(f"Win Rate: {trade_stats.win_rate:.1%}")
print(f"Profit Factor: {trade_stats.profit_factor:.2f}")
print(f"Average Win: ${trade_stats.avg_win:.2f}")
print(f"Average Loss: ${trade_stats.avg_loss:.2f}")

# Monthly performance breakdown
monthly_returns = await reporter.calculate_monthly_returns()
for month, return_pct in monthly_returns.items():
    print(f"{month}: {return_pct:.2%}")

# Drawdown analysis
drawdown_periods = await reporter.analyze_drawdown_periods()
for i, period in enumerate(drawdown_periods):
    print(f"Drawdown {i+1}: {period.max_drawdown:.2%} "
          f"({period.duration} days, recovered in {period.recovery_days} days)")
```

### Risk Analysis
```python
# Comprehensive risk metrics
risk_metrics = await reporter.calculate_risk_metrics()

# Volatility analysis
print(f"Daily Volatility: {risk_metrics.daily_volatility:.2%}")
print(f"Annual Volatility: {risk_metrics.annual_volatility:.2%}")

# Value at Risk
print(f"VaR (95%): {risk_metrics.var_95:.2%}")
print(f"VaR (99%): {risk_metrics.var_99:.2%}")
print(f"CVaR (95%): {risk_metrics.cvar_95:.2%}")

# Risk-adjusted returns
print(f"Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}")
print(f"Sortino Ratio: {risk_metrics.sortino_ratio:.2f}")
print(f"Calmar Ratio: {risk_metrics.calmar_ratio:.2f}")

# Stress testing
stress_results = await reporter.perform_stress_testing()
print(f"Worst Case Scenario: {stress_results.worst_case:.2%}")
print(f"95% Confidence Interval: {stress_results.confidence_interval}")
```
