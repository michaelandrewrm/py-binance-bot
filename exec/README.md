# Exchange Execution Directory

## Purpose

The exec directory provides exchange-specific execution implementations and rate limiting capabilities for trading operations. This layer handles direct communication with cryptocurrency exchanges, manages API rate limits, and provides both live and paper trading execution modes.

## Architecture

The execution layer provides:
- **Exchange Integration**: Direct integration with Binance and other cryptocurrency exchanges
- **Rate Limiting**: Sophisticated rate limiting to comply with exchange API limits
- **Paper Trading**: Risk-free paper trading simulation for strategy testing
- **Order Management**: Complete order lifecycle management with status tracking
- **Error Handling**: Robust error handling for network issues and exchange errors

## Files Overview

### Core Execution Components

#### `binance.py` - Binance Exchange Integration
- **Binance API Integration**: Complete Binance REST and WebSocket API implementation
- **Order Management**: Full order lifecycle management with status tracking
- **Market Data**: Real-time market data streams and historical data retrieval
- **Account Management**: Balance queries, position tracking, and account information
- **Error Handling**: Comprehensive error handling for all Binance API operations

#### `paper_broker.py` - Paper Trading Implementation
- **Simulated Trading**: Complete paper trading implementation without real money risk
- **Realistic Execution**: Realistic order execution simulation with market conditions
- **Portfolio Tracking**: Full portfolio management in simulation mode
- **Performance Analysis**: Real-time performance tracking during paper trading
- **Risk-Free Testing**: Strategy testing without financial risk

#### `rate_limit.py` - API Rate Limiting
- **Rate Limit Management**: Sophisticated rate limiting for exchange API compliance
- **Request Queuing**: Intelligent request queuing and throttling
- **Burst Handling**: Support for burst requests within rate limits
- **Priority Management**: Priority-based request handling for critical operations
- **Monitoring**: Rate limit usage monitoring and alerting

## Key Functions and Classes

### Binance Exchange Integration

#### Core Exchange Interface
**`BinanceExecutor.__init__(api_key: str, secret_key: str, testnet: bool = False)`**
- **Purpose**: Initialize Binance exchange connection with API credentials
- **Parameters**: API key, secret key, testnet flag for testing
- **Features**: Credential validation, connection testing, rate limit setup
- **Usage**: Primary interface for Binance trading operations

**`BinanceExecutor.place_order(order_request: OrderRequest) -> OrderResult`**
- **Purpose**: Place trading order on Binance exchange
- **Parameters**: Order request with symbol, side, quantity, and type
- **Returns**: Order result with execution details and status
- **Features**: Order validation, rate limiting, error handling, status tracking
- **Usage**: Core function for trade execution on Binance

**`BinanceExecutor.cancel_order(symbol: str, order_id: int) -> CancelResult`**
- **Purpose**: Cancel existing order on Binance exchange
- **Parameters**: Trading symbol, exchange order ID
- **Returns**: Cancellation result with status and details
- **Features**: Order validation, rate limiting, error handling
- **Usage**: Order management and risk control

#### Market Data Operations
**`BinanceExecutor.get_account_info() -> AccountInfo`**
- **Purpose**: Retrieve current account information and balances
- **Returns**: Complete account information with balances and permissions
- **Features**: Real-time data, caching, error handling
- **Usage**: Portfolio monitoring and balance checking

**`BinanceExecutor.get_order_status(symbol: str, order_id: int) -> OrderStatus`**
- **Purpose**: Query current status of specific order
- **Parameters**: Trading symbol, exchange order ID
- **Returns**: Current order status with execution details
- **Features**: Real-time status, partial fill tracking
- **Usage**: Order monitoring and execution tracking

**`BinanceExecutor.get_trade_history(symbol: str, limit: int = 100) -> List[TradeData]`**
- **Purpose**: Retrieve recent trade history for symbol
- **Parameters**: Trading symbol, optional limit on number of trades
- **Returns**: List of recent trades with execution details
- **Features**: Pagination, filtering, rate limiting
- **Usage**: Trade analysis and portfolio reconciliation

#### WebSocket Streams
**`BinanceExecutor.start_price_stream(symbol: str, callback: Callable) -> None`**
- **Purpose**: Start real-time price stream for trading symbol
- **Parameters**: Trading symbol, callback function for price updates
- **Features**: Automatic reconnection, error handling, buffering
- **Usage**: Real-time price monitoring for trading decisions

**`BinanceExecutor.start_user_stream(callback: Callable) -> None`**
- **Purpose**: Start user data stream for account updates
- **Parameters**: Callback function for account update events
- **Features**: Order updates, balance changes, execution notifications
- **Usage**: Real-time account monitoring and trade notifications

### Paper Trading Implementation

#### Paper Broker Core
**`PaperBroker.__init__(initial_balance: Dict[str, Decimal], market_data: MarketData)`**
- **Purpose**: Initialize paper trading environment with starting balances
- **Parameters**: Initial account balances, market data source
- **Features**: Portfolio initialization, market data connection, performance tracking
- **Usage**: Risk-free strategy testing and development

**`PaperBroker.place_order(order_request: OrderRequest) -> OrderResult`**
- **Purpose**: Simulate order execution in paper trading environment
- **Parameters**: Order request with trading details
- **Returns**: Simulated order execution result
- **Features**: Realistic execution simulation, slippage modeling, fee calculation
- **Usage**: Strategy testing without financial risk

**`PaperBroker.get_portfolio_value() -> Decimal`**
- **Purpose**: Calculate current total portfolio value in paper trading
- **Returns**: Current portfolio value including positions and cash
- **Features**: Real-time valuation, mark-to-market calculation
- **Usage**: Performance monitoring during paper trading

#### Realistic Simulation
**`PaperBroker.simulate_order_execution(order: OrderRequest, market_price: Decimal) -> ExecutionResult`**
- **Purpose**: Simulate realistic order execution with market conditions
- **Parameters**: Order request, current market price
- **Returns**: Execution result with realistic fill price and timing
- **Features**: Slippage simulation, partial fills, market impact modeling
- **Usage**: Realistic execution simulation for strategy validation

**`PaperBroker.calculate_unrealized_pnl() -> Dict[str, Decimal]`**
- **Purpose**: Calculate unrealized profit/loss for all positions
- **Returns**: Dictionary mapping symbols to unrealized P&L
- **Features**: Mark-to-market calculation, real-time updates
- **Usage**: Risk monitoring and position tracking

**`PaperBroker.generate_trade_report() -> TradeReport`**
- **Purpose**: Generate comprehensive trade report for paper trading session
- **Returns**: Detailed report with performance metrics and trade analysis
- **Features**: Performance calculation, trade statistics, visualization
- **Usage**: Strategy evaluation and performance analysis

### Rate Limiting System

#### Rate Limit Management
**`RateLimit.__init__(requests_per_second: float, burst_size: int = 10)`**
- **Purpose**: Initialize rate limiting with specified limits
- **Parameters**: Maximum requests per second, burst capacity
- **Features**: Token bucket algorithm, burst handling, thread safety
- **Usage**: API rate limit compliance and request throttling

**`RateLimit.acquire(weight: int = 1) -> bool`**
- **Purpose**: Acquire permission to make API request with specified weight
- **Parameters**: Request weight (default 1)
- **Returns**: True if permission granted, False if rate limited
- **Features**: Weight-based limiting, priority handling, queuing
- **Usage**: Request throttling before API calls

**`RateLimit.wait_for_capacity(weight: int = 1) -> None`**
- **Purpose**: Wait until sufficient capacity is available for request
- **Parameters**: Request weight to wait for
- **Features**: Blocking wait, timeout handling, priority queuing
- **Usage**: Ensuring compliance with rate limits

#### Advanced Rate Limiting
**`RateLimit.get_current_usage() -> RateUsage`**
- **Purpose**: Get current rate limit usage statistics
- **Returns**: Usage statistics with remaining capacity and reset time
- **Features**: Real-time monitoring, usage history, trend analysis
- **Usage**: Rate limit monitoring and optimization

**`RateLimit.set_priority_weight(request_type: str, weight: int) -> None`**
- **Purpose**: Set priority weights for different request types
- **Parameters**: Request type identifier, priority weight
- **Features**: Dynamic prioritization, request classification
- **Usage**: Optimizing critical request handling

**`RateLimit.reset_limits() -> None`**
- **Purpose**: Reset rate limit counters and capacity
- **Features**: Emergency reset, testing support, manual override
- **Usage**: Testing and emergency rate limit management

## Integration Points

### With Core System
- **core/executor.py**: Exchange execution implementation and order management
- **core/safety.py**: Risk management integration and safety checks
- **core/state.py**: State synchronization with exchange data

### With Data Layer
- **data/loader.py**: Market data integration and real-time feeds
- **data/schema.py**: Data structures for orders, trades, and account information

### With Storage Layer
- **storage/repo.py**: Trade execution logging and persistence
- **storage/artifacts.py**: Execution performance tracking and analysis

## Usage Examples

### Binance Exchange Operations
```python
# Initialize Binance executor
binance = BinanceExecutor(
    api_key="your_api_key",
    secret_key="your_secret_key",
    testnet=True  # Use testnet for development
)

# Place market order
order_request = OrderRequest(
    symbol="BTCUSDC",
    side="BUY",
    quantity=Decimal("0.1"),
    order_type="MARKET"
)

order_result = await binance.place_order(order_request)
print(f"Order placed: {order_result.order_id}")
print(f"Fill price: ${order_result.fill_price:.2f}")

# Monitor order status
while order_result.status != "FILLED":
    status = await binance.get_order_status("BTCUSDC", order_result.order_id)
    print(f"Order status: {status.status}")
    await asyncio.sleep(1)

# Get account information
account_info = await binance.get_account_info()
for balance in account_info.balances:
    if balance.free > 0:
        print(f"{balance.asset}: {balance.free}")
```

### Paper Trading Setup
```python
# Initialize paper broker
initial_balances = {
    "USDC": Decimal("10000.00"),
    "BTC": Decimal("0.0")
}

paper_broker = PaperBroker(
    initial_balance=initial_balances,
    market_data=market_data_source
)

# Execute paper trades
paper_order = OrderRequest(
    symbol="BTCUSDC",
    side="BUY",
    quantity=Decimal("0.2"),
    order_type="MARKET"
)

paper_result = await paper_broker.place_order(paper_order)
print(f"Paper trade executed at ${paper_result.fill_price:.2f}")

# Monitor portfolio value
portfolio_value = await paper_broker.get_portfolio_value()
print(f"Portfolio value: ${portfolio_value:.2f}")

# Generate performance report
report = await paper_broker.generate_trade_report()
print(f"Total return: {report.total_return:.2%}")
print(f"Sharpe ratio: {report.sharpe_ratio:.2f}")
```

### Rate Limiting Implementation
```python
# Initialize rate limiter for Binance (1200 requests per minute)
rate_limiter = RateLimit(
    requests_per_second=20.0,  # 1200/60
    burst_size=50
)

# Make rate-limited API calls
async def safe_api_call(api_function, *args, **kwargs):
    # Wait for rate limit capacity
    await rate_limiter.wait_for_capacity(weight=1)
    
    try:
        result = await api_function(*args, **kwargs)
        return result
    except RateLimitError:
        # Handle rate limit exceeded
        print("Rate limit exceeded, waiting...")
        await asyncio.sleep(60)
        return await api_function(*args, **kwargs)

# Use rate-limited calls
account_info = await safe_api_call(binance.get_account_info)
order_result = await safe_api_call(binance.place_order, order_request)
```

### Real-time Data Streams
```python
# Price stream callback
async def price_update_callback(price_data):
    print(f"Price update: {price_data.symbol} = ${price_data.price:.2f}")
    
    # Update trading strategy with new price
    await strategy.update_price(price_data)

# User data stream callback
async def user_data_callback(user_data):
    if user_data.event_type == "executionReport":
        print(f"Order executed: {user_data.order_id}")
        await handle_trade_execution(user_data)
    elif user_data.event_type == "outboundAccountPosition":
        print(f"Balance update: {user_data.asset} = {user_data.free}")
        await update_portfolio_state(user_data)

# Start streams
await binance.start_price_stream("BTCUSDC", price_update_callback)
await binance.start_user_stream(user_data_callback)

# Keep streams running
try:
    while True:
        await asyncio.sleep(1)
except KeyboardInterrupt:
    print("Stopping streams...")
    await binance.stop_all_streams()
```

### Advanced Order Management
```python
# Place limit order with advanced options
limit_order = OrderRequest(
    symbol="BTCUSDC",
    side="BUY",
    quantity=Decimal("0.5"),
    order_type="LIMIT",
    price=Decimal("45000.00"),
    time_in_force="GTC",  # Good Till Cancelled
    stop_price=None,
    iceberg_qty=None
)

order_result = await binance.place_order(limit_order)

# Monitor and manage order
timeout = 300  # 5 minutes
start_time = time.time()

while time.time() - start_time < timeout:
    status = await binance.get_order_status("BTCUSDC", order_result.order_id)
    
    if status.status == "FILLED":
        print("Order completely filled")
        break
    elif status.status == "PARTIALLY_FILLED":
        print(f"Partial fill: {status.executed_qty}/{status.orig_qty}")
    
    await asyncio.sleep(10)

# Cancel if not filled within timeout
if status.status not in ["FILLED", "CANCELED"]:
    cancel_result = await binance.cancel_order("BTCUSDC", order_result.order_id)
    print(f"Order cancelled: {cancel_result.status}")
```

### Error Handling and Recovery
```python
async def robust_order_execution(order_request):
    max_retries = 3
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            # Attempt order execution
            result = await binance.place_order(order_request)
            return result
            
        except ExchangeConnectionError:
            print(f"Connection error, retry {attempt + 1}/{max_retries}")
            await asyncio.sleep(retry_delay * (2 ** attempt))
            
        except InsufficientBalanceError:
            print("Insufficient balance for order")
            raise
            
        except RateLimitError:
            print("Rate limit exceeded, waiting...")
            await asyncio.sleep(60)
            
        except Exception as e:
            print(f"Unexpected error: {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(retry_delay)
    
    raise Exception("Max retries exceeded")

# Use robust execution
try:
    result = await robust_order_execution(order_request)
    print(f"Order executed successfully: {result.order_id}")
except Exception as e:
    print(f"Order execution failed: {e}")
    # Handle failure appropriately
```
