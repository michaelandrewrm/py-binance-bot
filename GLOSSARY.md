# Glossary

## Trading Terms

### **ATR (Average True Range)**
A volatility indicator that measures the magnitude of price movements over a specified period. Used to set dynamic stop-losses and position sizing based on market volatility.

### **Volatility Bands**
Price channels constructed around a moving average using volatility measurements (typically ATR multiples). Used for trend-following strategies to identify breakouts and reversals.

### **Mean Reversion**
A trading strategy based on the assumption that asset prices will return to their historical average over time. Involves buying oversold conditions and selling overbought conditions.

### **OHLCV (Open, High, Low, Close, Volume)**
Standard market data format representing the opening price, highest price, lowest price, closing price, and trading volume for a specific time period (candlestick). This is the fundamental data structure used in technical analysis and trading systems.

### **Signal Confidence**
A quantitative measure (typically 0.0-1.0) indicating the reliability or strength of a trading signal. Higher confidence suggests greater probability of successful trade execution.

### **Grid Trading**
A systematic trading approach that places multiple buy and sell orders at predetermined price levels, creating a "grid" of orders to profit from market volatility.

### **Position Sizing**
The process of determining the appropriate quantity of an asset to buy or sell based on risk management parameters, account size, and market conditions.

### **Slippage**
The difference between the expected price of a trade and the actual executed price, typically caused by market movement during order execution or lack of liquidity.

## AI/ML Terms

### **Bayesian Optimization**
An efficient global optimization technique that uses probabilistic models to find optimal hyperparameters with minimal function evaluations. Particularly useful for expensive-to-evaluate objective functions.

### **Acquisition Function**
A function used in Bayesian optimization that determines which parameters to evaluate next by balancing exploration of unknown regions with exploitation of promising areas.

### **Hyperparameter Space**
The multi-dimensional space of all possible parameter combinations that can be optimized for a machine learning model or trading strategy.

### **Walk-Forward Analysis**
A time-series validation technique that uses rolling windows of training and testing data, maintaining temporal order to simulate realistic trading conditions.

### **Overfitting Score**
A quantitative metric (0-1) that measures how much a strategy's performance degrades from training to testing data, indicating potential overfitting.

### **Parameter Stability**
The consistency of a parameter's performance across different time periods and market conditions, measured by coefficient of variation or similar statistical measures.

### **Out-of-Sample Testing**
Validation performed on data that was not used during strategy development or optimization, providing an unbiased estimate of future performance.

### **Ensemble Strategy**
A trading approach that combines multiple individual strategies using weighted voting or other aggregation methods to improve overall robustness and performance.

## Technical Indicators

### **SMA (Simple Moving Average)**
The arithmetic mean of a security's price over a specified number of periods, used to smooth price data and identify trend direction.

### **EMA (Exponential Moving Average)**
A moving average that gives more weight to recent prices, making it more responsive to new information than a simple moving average.

### **RSI (Relative Strength Index)**
A momentum oscillator (0-100) that measures the speed and magnitude of price changes, used to identify overbought (>70) and oversold (<30) conditions.

### **Bollinger Bands**
A volatility indicator consisting of a moving average with upper and lower bands placed at standard deviation distances, used to identify relative price levels.

### **MACD (Moving Average Convergence Divergence)**
A trend-following momentum indicator that shows the relationship between two moving averages of a security's price.

## Risk Management Terms

### **Circuit Breaker**
An automated risk management mechanism that temporarily halts trading when predefined loss thresholds or risk conditions are met.

### **Maximum Drawdown**
The largest peak-to-trough decline in portfolio value over a specific period, expressed as a percentage of the peak value.

### **Sharpe Ratio**
A risk-adjusted performance measure calculated as (return - risk-free rate) / standard deviation, indicating return per unit of risk.

### **Position Limit**
The maximum allowable position size for a single trade or overall portfolio exposure, typically expressed as a percentage of total capital.

### **Stop-Loss**
A predetermined price level at which a losing position is automatically closed to limit further losses.

### **Take-Profit**
A predetermined price level at which a winning position is automatically closed to secure profits.

## System Architecture Terms

### **API Rate Limiting**
The restriction of the number of API requests that can be made within a specific time period to prevent system overload and ensure fair usage.

### **State Persistence**
The ability of a system to save its current state to storage and restore it after a restart, ensuring continuity of operations.

### **Async/Await**
A programming pattern that allows for non-blocking, concurrent execution of operations, particularly useful for I/O-bound tasks like API calls.

### **Paper Trading**
A simulation mode that allows testing of trading strategies using real market data without actual financial risk.

### **Backtesting**
The process of testing a trading strategy on historical data to evaluate its potential effectiveness before deploying it with real money.

## Exchange Terms

### **Testnet**
A test network provided by exchanges that simulates real trading conditions without using actual funds, used for testing and development.

### **Order Book**
A list of buy and sell orders for a specific security, organized by price level, showing market depth and liquidity.

### **Tick Size**
The minimum price increment at which a security can be traded on an exchange.

### **Step Size**
The minimum quantity increment for placing orders, determining the precision of order quantities.

### **Minimum Notional**
The minimum value requirement for an order, calculated as quantity × price, ensuring orders meet exchange standards.

### **Maker/Taker Fees**
Different fee structures where makers add liquidity to the order book (lower fees) and takers remove liquidity (higher fees).

## Performance Metrics

### **Win Rate**
The percentage of trades that result in a profit, calculated as (winning trades / total trades) × 100.

### **Profit Factor**
The ratio of gross profit to gross loss, indicating overall strategy profitability. Values above 1.0 indicate profitable strategies.

### **Calmar Ratio**
A risk-adjusted return measure calculated as annualized return divided by maximum drawdown, focusing on downside risk.

### **Sortino Ratio**
Similar to Sharpe ratio but only considers downside volatility, penalizing negative returns more than positive volatility.

### **Utilization**
The percentage of time a strategy maintains market positions, indicating capital efficiency and market exposure.

### **Turnover**
The rate at which positions are bought and sold within a portfolio, affecting transaction costs and tax implications.

## Simulation & Backtesting Terms

### **Event-Driven Backtesting**
A simulation methodology that processes market events in chronological order, providing realistic strategy testing with proper event sequencing.

### **HODL Strategy**
"Hold On for Dear Life" - a passive investment strategy of buying and holding assets long-term, used as a benchmark for active trading strategies.

### **Slippage Model**
Mathematical framework for simulating realistic trade execution costs, including fixed, linear, and volatility-based slippage calculations.

### **Market Impact**
The effect of large orders on market prices, typically increasing execution costs for larger position sizes.

### **Execution Price**
The actual price at which a trade is filled, which may differ from the expected price due to slippage and market movement.

### **Linear Slippage**
A slippage model where execution costs increase proportionally with order size, simulating realistic market impact for different trade sizes.

### **Volatility Slippage**
A slippage model that adjusts execution costs based on current market volatility, with higher costs during volatile periods.

## Reporting & Analysis Terms

### **Equity Curve**
A time series chart showing the evolution of portfolio value over time, used to visualize strategy performance and calculate risk metrics.

### **Benchmark Comparison**
Performance analysis comparing a trading strategy against standard benchmarks like HODL, market indices, or other strategies.

### **Risk-Adjusted Returns**
Performance metrics that account for the level of risk taken to achieve returns, such as Sharpe ratio, Calmar ratio, and Sortino ratio.

### **Performance Attribution**
Analysis that breaks down strategy returns by various factors, helping identify sources of outperformance or underperformance.

### **Session Metrics**
Performance statistics calculated for specific trading periods, including P&L, trade frequency, win rate, and risk measures.

### **Chart Generation**
Automated creation of performance visualizations including equity curves, trade scatter plots, and monthly return heatmaps.

### **Parameter Importance**
Analysis showing which strategy parameters have the greatest impact on performance, useful for optimization and sensitivity analysis.

### **Convergence Analysis**
Study of how optimization algorithms approach optimal solutions, used to assess optimization quality and stopping criteria.

### **Alpha/Beta Analysis**
Statistical measures comparing strategy performance to benchmarks, where alpha represents excess return and beta measures correlation.

### **Information Ratio**
Performance metric measuring active return relative to benchmark divided by tracking error, assessing manager skill in generating excess returns.

### **Value at Risk (VaR)**
Statistical measure estimating potential portfolio loss at a given confidence level over a specified time period.

### **Conditional VaR (CVaR)**
Expected loss beyond the VaR threshold, providing a more comprehensive measure of tail risk than standard VaR.

### **Tail Ratio**
Risk metric comparing the magnitude of gains in the upper tail to losses in the lower tail of the return distribution.

### **Downside Deviation**
Volatility measure that only considers negative returns, providing a more relevant risk measure for investors concerned with losses.

### **Maximum Consecutive Losses**
Risk metric tracking the longest streak of losing trades, important for understanding psychological pressure and strategy robustness.

### **Skewness**
Statistical measure of return distribution asymmetry, indicating whether extreme gains or losses are more likely.

### **Kurtosis**
Statistical measure of return distribution tail heaviness, indicating the likelihood of extreme events compared to normal distribution.
