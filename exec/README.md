# Exec Directory

## File Overview

This directory contains the exchange interface layer that handles communication with trading platforms and execution brokers.

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
# Returns: {'balances': [{'asset': 'USDT', 'free': '1000.0', ...}], ...}
```

#### `place_order(symbol, side, order_type, quantity, price=None, time_in_force='GTC')`
- **What**: Places a trading order on Binance
- **Why**: Executes buy/sell orders for the trading strategy
- **What for**: Order placement in grid trading and other strategies
- **Example**:
```python
order = await client.place_order(
    symbol="BTCUSDT",
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
result = await client.cancel_order("BTCUSDT", "12345")
# Returns: {'orderId': '12345', 'status': 'CANCELED', ...}
```

#### `get_open_orders(symbol=None)`
- **What**: Retrieves all open orders for a symbol or all symbols
- **Why**: Provides visibility into current market exposure
- **What for**: Order management and position tracking
- **Example**:
```python
orders = await client.get_open_orders("BTCUSDT")
# Returns: [Order(id='123', symbol='BTCUSDT', side='BUY', ...), ...]
```

#### `get_klines(symbol, interval, limit=500, start_time=None, end_time=None)`
- **What**: Fetches historical price data (candlesticks)
- **Why**: Provides market data for analysis and strategy decisions
- **What for**: Market analysis, backtesting, and AI model inputs
- **Example**:
```python
klines = await client.get_klines("BTCUSDT", "1h", limit=100)
# Returns: [KlineData(open=50000, high=50100, low=49900, close=50050, ...), ...]
```

#### `get_24hr_ticker(symbol)`
- **What**: Retrieves 24-hour ticker statistics for a symbol
- **Why**: Provides current market statistics for analysis
- **What for**: Market condition assessment and strategy optimization
- **Example**:
```python
ticker = await client.get_24hr_ticker("BTCUSDT")
# Returns: {'symbol': 'BTCUSDT', 'priceChange': '100.0', 'volume': '1000', ...}
```

#### `get_exchange_info(symbol=None)`
- **What**: Retrieves exchange trading rules and symbol information
- **Why**: Provides trading constraints and symbol specifications
- **What for**: Order validation and compliance with exchange rules
- **Example**:
```python
info = await client.get_exchange_info("BTCUSDT")
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
initial_balances = {'USDT': Decimal('10000'), 'BTC': Decimal('0')}
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
    symbol="BTCUSDT",
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
result = await broker.cancel_order("BTCUSDT", "paper_order_123")
# Returns: Cancellation confirmation with simulated timing
```

#### `update_market_data(symbol, bid, ask, last_price, volume)`
- **What**: Updates market data for order execution simulation
- **Why**: Provides realistic market conditions for order filling
- **What for**: Simulating market movements and order execution
- **Example**:
```python
broker.update_market_data(
    symbol="BTCUSDT",
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
# Returns: {'balances': [{'asset': 'USDT', 'free': '9500.0', ...}], ...}
```

#### `get_open_orders(symbol=None)`
- **What**: Retrieves all open simulated orders
- **Why**: Provides visibility into simulated market exposure
- **What for**: Testing order tracking and management logic
- **Example**:
```python
orders = await broker.get_open_orders("BTCUSDT")
# Returns: [Order(id='paper_123', symbol='BTCUSDT', side='BUY', ...), ...]
```

#### `get_order_history(symbol, limit=100)`
- **What**: Retrieves history of executed simulated orders
- **Why**: Provides trade history for performance analysis
- **What for**: Analyzing simulated trading performance and behavior
- **Example**:
```python
history = await broker.get_order_history("BTCUSDT", limit=50)
# Returns: [Order(...), ...] list of historical orders
```

#### `calculate_pnl(symbol)`
- **What**: Calculates realized and unrealized P&L for a symbol
- **Why**: Provides performance metrics in the simulation
- **What for**: Performance tracking and strategy evaluation
- **Example**:
```python
pnl = broker.calculate_pnl("BTCUSDT")
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
