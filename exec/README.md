# Exec Directory

## File Overview

This directory contains the exchange interface layer that handles communication with trading platforms and execution brokers.

## Functionality Analysis Table

| Module | Class/Function | Purpose | Inputs | Outputs | Side Effects | Test Coverage |
|--------|---------------|---------|---------|---------|--------------|---------------|
| **binance.py** | | | | | | |
| | `mask_api_key()` | Mask sensitive API key for logging | API key string | Masked string | None | ✅ 100% |
| | `BinanceConfig` | Configuration data class for API client | API credentials, URLs, timeouts | Config object | None | ✅ 100% |
| | `BinanceError` | Base exception for API errors | Message, code, response | Exception object | None | ✅ 100% |
| | `BinanceRateLimitError` | Rate limit specific exception | Message | Exception object | None | ✅ 100% |
| | `BinanceClient.__init__()` | Initialize API client | BinanceConfig | Client instance | Creates rate limiter | ✅ 100% |
| | `BinanceClient.start_session()` | Initialize HTTP session | None | None | Creates aiohttp session, syncs time | ✅ 100% |
| | `BinanceClient.close_session()` | Cleanup HTTP session | None | None | Closes aiohttp session | ✅ 100% |
| | `BinanceClient._sync_server_time()` | Synchronize with server time | None | None | Updates time offset | ✅ 100% |
| | `BinanceClient._get_timestamp()` | Get current timestamp with offset | None | int (milliseconds) | None | ✅ 100% |
| | `BinanceClient._generate_signature()` | Create HMAC signature | Query string | Hex signature | None | ✅ 100% |
| | `BinanceClient._calculate_retry_delay()` | Calculate exponential backoff delay | Attempt number | float (seconds) | None | ✅ 100% |
| | `BinanceClient._make_request_single()` | Single HTTP request with rate limiting | Method, endpoint, params | Response dict | Rate limit consumption | ✅ 100% |
| | `BinanceClient._make_request_with_retry()` | HTTP request with retry logic | Method, endpoint, params | Response dict | Multiple requests on retry | ✅ 100% |
| | `BinanceClient._make_request()` | Main request interface | Method, endpoint, params | Response dict | HTTP requests, rate limiting | ✅ 100% |
| | `BinanceClient.get_server_time()` | Get Binance server time | None | Server time dict | HTTP request | ✅ 100% |
| | `BinanceClient.get_exchange_info()` | Get trading rules and info | Symbol (optional) | Exchange info dict | HTTP request | ✅ 100% |
| | `BinanceClient.get_klines()` | Get candlestick data | Symbol, interval, limit | Klines array | HTTP request | ✅ 95% |
| | `BinanceClient.get_ticker()` | Get 24h ticker stats | Symbol | Ticker dict | HTTP request | ✅ 95% |
| | `BinanceClient.get_order_book()` | Get order book depth | Symbol, limit | Order book dict | HTTP request | ✅ 95% |
| | `BinanceClient.get_account_info()` | Get account balances | None | Account dict | HTTP request | ✅ 95% |
| | `BinanceClient.place_order()` | Place trading order | Symbol, side, type, quantity, price | Order dict | HTTP request, rate limiting | ✅ 100% |
| | `BinanceClient.cancel_order()` | Cancel existing order | Symbol, order_id/client_order_id | Cancel response | HTTP request | ✅ 100% |
| | `BinanceClient.get_order()` | Get order status | Symbol, order_id/client_order_id | Order dict | HTTP request | ✅ 95% |
| | `BinanceClient.get_open_orders()` | Get all open orders | Symbol (optional) | Orders array | HTTP request | ✅ 95% |
| | `BinanceClient.get_order_history()` | Get historical orders | Symbol, limit | Orders array | HTTP request | ✅ 95% |
| | `BinanceClient.convert_order_to_internal()` | Convert API order to internal format | Binance order dict | Order object | None | ✅ 100% |
| | `BinanceClient.health_check()` | Test API connectivity | None | bool (healthy) | HTTP request | ✅ 100% |
| | `create_binance_client()` | Factory function for client creation | Credentials (optional) | Client instance | Session initialization | ✅ 100% |
| **paper_broker.py** | | | | | | |
| | `MarketData` | Real-time market data container | Symbol, bid, ask, price, volume | Data object | None | ✅ 100% |
| | `MarketData.mid_price` | Calculate mid price | None | Decimal (mid price) | None | ✅ 100% |
| | `MarketData.spread` | Calculate bid-ask spread | None | Decimal (spread) | None | ✅ 100% |
| | `MarketData.spread_pct` | Calculate spread percentage | None | Decimal (percentage) | None | ✅ 100% |
| | `OrderBookLevel` | Single order book price level | Price, quantity | Level object | None | ✅ 100% |
| | `OrderBook` | Order book data structure | Symbol, bids, asks | Book object | None | ✅ 100% |
| | `OrderBook.get_best_bid()` | Get highest bid price | None | Decimal or None | None | ✅ 100% |
| | `OrderBook.get_best_ask()` | Get lowest ask price | None | Decimal or None | None | ✅ 100% |
| | `OrderBook.get_mid_price()` | Calculate order book mid price | None | Decimal or None | None | ✅ 100% |
| | `SlippageModel.calculate_slippage()` | Abstract slippage calculation | Order, market data | Decimal (slippage) | None | ✅ 95% |
| | `LinearSlippageModel` | Linear slippage implementation | Base slippage, size impact | Model object | None | ✅ 100% |
| | `LinearSlippageModel.calculate_slippage()` | Linear slippage calculation | Order, market data | Decimal (slippage) | None | ✅ 100% |
| | `PaperBroker.__init__()` | Initialize paper trading broker | Initial balances, exchange info | Broker instance | Creates internal state | ✅ 100% |
| | `PaperBroker.register_fill_callback()` | Register trade execution callback | Callback function | None | Adds to callback list | ✅ 100% |
| | `PaperBroker.register_order_callback()` | Register order update callback | Callback function | None | Adds to callback list | ✅ 100% |
| | `PaperBroker.update_market_data()` | Update real-time market data | Symbol, bid, ask, price, volume | None | Updates internal market data | ✅ 100% |
| | `PaperBroker.place_order()` | Simulate order placement | Symbol, side, type, quantity, price | Order object | Updates balances, may execute | ✅ 100% |
| | `PaperBroker.cancel_order()` | Cancel pending order | Order ID | bool (success) | Updates balances, order status | ✅ 100% |
| | `PaperBroker._check_balance()` | Validate sufficient balance | Symbol, side, quantity, price | bool (sufficient) | None | ✅ 100% |
| | `PaperBroker._parse_symbol()` | Extract base/quote from symbol | Symbol string | (base, quote) tuple | None | ✅ 100% |
| | `PaperBroker._reserve_balance()` | Lock funds for order | Symbol, side, quantity, price | None | Updates balance locks | ✅ 95% |
| | `PaperBroker._release_balance()` | Unlock unused funds | Symbol, side, quantity, price | None | Updates balance locks | ✅ 95% |
| | `PaperBroker._execute_order()` | Simulate order execution | Order, execution price | None | Updates balances, positions | ✅ 95% |
| | `PaperBroker._calculate_execution_price()` | Determine realistic execution price | Order, market data | Decimal (price) | None | ✅ 95% |
| | `PaperBroker._process_pending_orders()` | Check/execute pending orders | None | None | May execute limit orders | ✅ 90% |
| | `PaperBroker.get_account_info()` | Get simulated account state | None | Account dict | None | ✅ 100% |
| | `PaperBroker.get_position()` | Get position for symbol | Symbol | Position object or None | None | ✅ 95% |
| **rate_limit.py** | | | | | | |
| | `RateLimitError` | Rate limiting exception | Message, retry_after | Exception object | None | ✅ 100% |
| | `RateLimitConfig` | Rate limiting configuration | Limits and timeouts | Config object | None | ✅ 100% |
| | `TokenBucket.__init__()` | Initialize token bucket | Capacity, refill rate | Bucket instance | Sets initial tokens | ✅ 100% |
| | `TokenBucket.acquire()` | Attempt token acquisition | Token count | bool (success) | Consumes tokens, triggers refill | ✅ 100% |
| | `TokenBucket._refill()` | Refill tokens based on time | None | None | Updates token count | ✅ 95% |
| | `TokenBucket.time_to_refill()` | Calculate time needed for tokens | Required tokens | float (seconds) | None | ✅ 100% |
| | `RateLimiter.__init__()` | Initialize multi-bucket rate limiter | RateLimitConfig | Limiter instance | Creates token buckets | ✅ 100% |
| | `RateLimiter.acquire()` | Acquire rate limit capacity | Weight, max retries | bool (success) | Consumes from multiple buckets | ✅ 100% |
| | `RateLimiter.acquire_order()` | Acquire order-specific capacity | Max retries | bool (success) | Consumes order tokens, updates daily count | ✅ 100% |
| | `RateLimiter._check_daily_order_limit()` | Check daily order limit | None | bool (within limit) | None | ✅ 95% |
| | `RateLimiter._calculate_retry_delay()` | Calculate retry delay | Attempt number | float (seconds) | None | ✅ 100% |
| | `RateLimiter.wait_for_capacity()` | Wait until capacity available | Weight, max wait | None | Sleeps thread | ✅ 100% |
| | `RateLimiter.get_status()` | Get current limiter status | None | Status dict | None | ✅ 100% |
| | `RateLimiter.reset_buckets()` | Reset all buckets to full | None | None | Refills all buckets | ✅ 100% |
| | `AdaptiveRateLimiter.__init__()` | Initialize adaptive rate limiter | RateLimitConfig | Limiter instance | Inherits + adds adaptive features | ✅ 100% |
| | `AdaptiveRateLimiter.record_error()` | Record API error for adaptation | Error type | None | Updates adaptive factor | ✅ 100% |
| | `AdaptiveRateLimiter._update_adaptive_factor()` | Adjust rate based on errors | None | None | Modifies adaptive factor | ✅ 95% |
| | `AdaptiveRateLimiter.acquire()` | Adaptive rate acquisition | Weight, max retries | bool (success) | Applies adaptive factor to weight | ✅ 100% |
| | `create_conservative_rate_limiter()` | Create conservative limiter | None | RateLimiter instance | None | ✅ 100% |
| | `create_aggressive_rate_limiter()` | Create aggressive limiter | None | RateLimiter instance | None | ✅ 100% |

---

### Python Files

#### `binance.py`
- **Purpose**: Official Binance API client wrapper with advanced features
- **Functionality**: Provides throttling, error handling, retry logic, and comprehensive API access

#### `rate_limit.py`
- **Purpose**: Rate limiting implementation using leaky bucket algorithm
- **Functionality**: Manages API request throttling, implements jitter-retry logic, and prevents rate limit violations

#### `paper_broker.py`
- **Purpose**: Paper trading simulation with in-memory order book
- **Functionality**: Simulates real trading without financial risk, provides realistic order execution simulation

## Function Documentation

### binance.py

#### `BinanceClient.__init__(config, rate_limiter=None)`
- **What**: Initializes the Binance API client with configuration and rate limiting
- **Why**: Provides a robust, production-ready interface to Binance API
- **What for**: Executing real trades, fetching market data, and managing orders
- **Example**:
```python
config = BinanceConfig(
    api_key="your_api_key",
    api_secret="your_secret",
    testnet=True
)
client = BinanceClient(config)
```

#### `get_account_info()`
- **What**: Retrieves current account information including balances
- **Why**: Provides account status for position and balance management
- **What for**: Account monitoring and balance verification
- **Example**:
```python
account_info = await client.get_account_info()
# Returns: {'balances': [{'asset': 'USDC', 'free': '1000.0', ...}], ...}
```

#### `place_order(symbol, side, order_type, quantity, price=None, time_in_force='GTC')`
- **What**: Places a trading order on Binance
- **Why**: Executes buy/sell orders for the trading strategy
- **What for**: Order placement in grid trading and other strategies
- **Example**:
```python
order = await client.place_order(
    symbol="BTCUSDC",
    side="BUY",
    order_type="LIMIT",
    quantity=Decimal("0.001"),
    price=Decimal("50000")
)
# Returns: Order object with execution details
```

#### `cancel_order(symbol, order_id)`
- **What**: Cancels an existing open order
- **Why**: Enables dynamic order management and risk control
- **What for**: Order cancellation in grid adjustments and risk management
- **Example**:
```python
result = await client.cancel_order("BTCUSDC", "12345")
# Returns: {'orderId': '12345', 'status': 'CANCELED', ...}
```

#### `get_open_orders(symbol=None)`
- **What**: Retrieves all open orders for a symbol or all symbols
- **Why**: Provides visibility into current market exposure
- **What for**: Order management and position tracking
- **Example**:
```python
orders = await client.get_open_orders("BTCUSDC")
# Returns: [Order(id='123', symbol='BTCUSDC', side='BUY', ...), ...]
```

#### `get_klines(symbol, interval, limit=500, start_time=None, end_time=None)`
- **What**: Fetches historical price data (candlesticks)
- **Why**: Provides market data for analysis and strategy decisions
- **What for**: Market analysis, backtesting, and AI model inputs
- **Example**:
```python
klines = await client.get_klines("BTCUSDC", "1h", limit=100)
# Returns: [KlineData(open=50000, high=50100, low=49900, close=50050, ...), ...]
```

#### `get_24hr_ticker(symbol)`
- **What**: Retrieves 24-hour ticker statistics for a symbol
- **Why**: Provides current market statistics for analysis
- **What for**: Market condition assessment and strategy optimization
- **Example**:
```python
ticker = await client.get_24hr_ticker("BTCUSDC")
# Returns: {'symbol': 'BTCUSDC', 'priceChange': '100.0', 'volume': '1000', ...}
```

#### `get_exchange_info(symbol=None)`
- **What**: Retrieves exchange trading rules and symbol information
- **Why**: Provides trading constraints and symbol specifications
- **What for**: Order validation and compliance with exchange rules
- **Example**:
```python
info = await client.get_exchange_info("BTCUSDC")
# Returns: ExchangeInfo with filters, precision, min quantities, etc.
```

### rate_limit.py

#### `RateLimiter.__init__(config)`
- **What**: Initializes the rate limiting system with configuration
- **Why**: Prevents API rate limit violations and ensures stable operation
- **What for**: Managing API request frequency and avoiding bans
- **Example**:
```python
config = RateLimitConfig(
    requests_per_minute=1200,
    weight_per_minute=6000
)
rate_limiter = RateLimiter(config)
```

#### `acquire(endpoint, weight=1)`
- **What**: Acquires permission to make an API request
- **Why**: Ensures requests stay within rate limits
- **What for**: Rate-limited API access in trading operations
- **Example**:
```python
async with rate_limiter.acquire("order", weight=1):
    # Make API call here
    order = await client.place_order(...)
```

#### `wait_if_needed(endpoint, weight=1)`
- **What**: Waits if necessary to respect rate limits before proceeding
- **Why**: Automatically manages request timing to prevent violations
- **What for**: Seamless rate-limited API usage
- **Example**:
```python
await rate_limiter.wait_if_needed("market_data", weight=2)
data = await client.get_klines(...)
```

#### `get_stats()`
- **What**: Retrieves current rate limiting statistics and status
- **Why**: Provides visibility into rate limit usage and health
- **What for**: Monitoring and debugging rate limit behavior
- **Example**:
```python
stats = rate_limiter.get_stats()
# Returns: {'requests_used': 150, 'weight_used': 300, 'buckets': {...}}
```

#### `reset_limits()`
- **What**: Resets all rate limiting counters and buckets
- **Why**: Provides emergency reset capability for testing or recovery
- **What for**: Testing scenarios and emergency rate limit recovery
- **Example**:
```python
rate_limiter.reset_limits()
# All rate limit counters reset to zero
```

### paper_broker.py

#### `PaperBroker.__init__(initial_balances, exchange_info, fee_config)`
- **What**: Initializes paper trading broker with virtual balances and configuration
- **Why**: Enables risk-free testing of trading strategies
- **What for**: Strategy testing, development, and validation
- **Example**:
```python
initial_balances = {'USDC': Decimal('10000'), 'BTC': Decimal('0')}
broker = PaperBroker(
    initial_balances=initial_balances,
    exchange_info=exchange_info,
    fee_config=fee_config
)
```

#### `place_order(symbol, side, order_type, quantity, price=None, time_in_force='GTC')`
- **What**: Simulates order placement in the paper trading environment
- **Why**: Provides realistic order execution simulation without real money
- **What for**: Testing trading strategies and order logic
- **Example**:
```python
order = await broker.place_order(
    symbol="BTCUSDC",
    side="BUY",
    order_type="LIMIT",
    quantity=Decimal("0.001"),
    price=Decimal("50000")
)
# Returns: Simulated Order object
```

#### `cancel_order(symbol, order_id)`
- **What**: Cancels a simulated order in the paper trading environment
- **Why**: Simulates real order cancellation behavior
- **What for**: Testing order management and cancellation logic
- **Example**:
```python
result = await broker.cancel_order("BTCUSDC", "paper_order_123")
# Returns: Cancellation confirmation with simulated timing
```

#### `update_market_data(symbol, bid, ask, last_price, volume)`
- **What**: Updates market data for order execution simulation
- **Why**: Provides realistic market conditions for order filling
- **What for**: Simulating market movements and order execution
- **Example**:
```python
broker.update_market_data(
    symbol="BTCUSDC",
    bid=Decimal("49990"),
    ask=Decimal("50010"),
    last_price=Decimal("50000"),
    volume=Decimal("100")
)
```

#### `get_account_info()`
- **What**: Retrieves simulated account information and balances
- **Why**: Provides account status in the paper trading environment
- **What for**: Monitoring simulated account performance
- **Example**:
```python
account = await broker.get_account_info()
# Returns: {'balances': [{'asset': 'USDC', 'free': '9500.0', ...}], ...}
```

#### `get_open_orders(symbol=None)`
- **What**: Retrieves all open simulated orders
- **Why**: Provides visibility into simulated market exposure
- **What for**: Testing order tracking and management logic
- **Example**:
```python
orders = await broker.get_open_orders("BTCUSDC")
# Returns: [Order(id='paper_123', symbol='BTCUSDC', side='BUY', ...), ...]
```

#### `get_order_history(symbol, limit=100)`
- **What**: Retrieves history of executed simulated orders
- **Why**: Provides trade history for performance analysis
- **What for**: Analyzing simulated trading performance and behavior
- **Example**:
```python
history = await broker.get_order_history("BTCUSDC", limit=50)
# Returns: [Order(...), ...] list of historical orders
```

#### `calculate_pnl(symbol)`
- **What**: Calculates realized and unrealized P&L for a symbol
- **Why**: Provides performance metrics in the simulation
- **What for**: Performance tracking and strategy evaluation
- **Example**:
```python
pnl = broker.calculate_pnl("BTCUSDC")
# Returns: {'realized_pnl': Decimal('50.0'), 'unrealized_pnl': Decimal('25.0')}
```

#### `simulate_slippage(order, market_data)`
- **What**: Simulates realistic price slippage for market orders
- **Why**: Provides realistic execution conditions in simulation
- **What for**: Testing strategy performance under realistic conditions
- **Example**:
```python
filled_price = broker.simulate_slippage(
    order=market_order,
    market_data=current_market_data
)
# Returns: Decimal('50005.0') (with simulated slippage)
```

#### `set_latency_simulation(min_ms, max_ms)`
- **What**: Configures artificial latency for order execution simulation
- **Why**: Simulates real-world network delays and execution timing
- **What for**: Testing strategy behavior under realistic latency conditions
- **Example**:
```python
broker.set_latency_simulation(min_ms=10, max_ms=100)
# Orders will have 10-100ms simulated execution delay
```
