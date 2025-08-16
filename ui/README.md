# UI Directory

## Purpose

The UI directory provides comprehensive user interfaces for trading bot management, including command-line interface, web dashboard, session management, and security systems. This layer enables intuitive interaction with all bot functionality through multiple interface channels.

## Architecture

The UI layer provides:
- **Multi-Modal Interfaces**: CLI for automation, web dashboard for visual monitoring
- **Session Management**: Secure user authentication and session handling
- **Real-time Monitoring**: Live trading data, performance metrics, and system health
- **Security Framework**: Authentication, authorization, and secure communication
- **Interactive Controls**: Bot management, strategy configuration, and trade execution

## Files Overview

### Core UI Components

#### `cli.py` - Command Line Interface
- **Comprehensive CLI**: Full-featured command-line interface for all bot operations
- **Interactive Menus**: User-friendly menu systems for complex operations
- **Real-time Display**: Live trading data, portfolio status, and performance metrics
- **Configuration Management**: Interactive bot configuration and strategy setup
- **Automation Support**: Scriptable commands for automated deployment and management

#### `dashboard.py` - Web Dashboard
- **Real-time Dashboard**: Modern web interface with live data updates
- **Performance Visualization**: Interactive charts for portfolio and trade analysis
- **Trading Controls**: Web-based trade execution and strategy management
- **Responsive Design**: Mobile-friendly interface for remote monitoring
- **Data Export**: Trade history and performance report generation

#### `session_manager.py` - Session Management
- **Secure Sessions**: JWT-based session management with refresh tokens
- **User Authentication**: Multi-factor authentication support
- **Session Persistence**: Secure session storage and recovery
- **Access Control**: Role-based permissions and resource authorization
- **Security Monitoring**: Login tracking and suspicious activity detection

#### `security.py` - Security Framework
- **Authentication**: Secure user authentication with password policies
- **Authorization**: Role-based access control (RBAC) system
- **Encryption**: Data encryption for sensitive information
- **API Security**: Secure API key management and request validation
- **Audit Logging**: Comprehensive security event logging

## Key Functions and Classes

### Command Line Interface

#### Core CLI Functions
**`TradingBotCLI.__init__(config: Dict)`**
- **Purpose**: Initialize CLI with configuration and command registration
- **Parameters**: Bot configuration dictionary
- **Features**: Command parsing, help system, configuration validation
- **Usage**: Main entry point for command-line operations

**`TradingBotCLI.start_interactive_mode()`**
- **Purpose**: Launch interactive CLI with menu-driven interface
- **Features**: Real-time data display, command completion, error handling
- **Display**: Portfolio status, active trades, performance metrics
- **Usage**: Primary interface for manual bot operation and monitoring

**`TradingBotCLI.show_portfolio_status()`**
- **Purpose**: Display current portfolio status with real-time updates
- **Features**: Position details, unrealized P&L, balance information
- **Formatting**: Colored output, tabular data, percentage calculations
- **Usage**: Quick portfolio overview and position monitoring

#### Trading Operations
**`TradingBotCLI.execute_trade_command(symbol: str, side: str, quantity: str)`**
- **Purpose**: Execute trades through command-line interface
- **Parameters**: Trading symbol, buy/sell side, quantity amount
- **Features**: Order validation, confirmation prompts, execution feedback
- **Safety**: Risk checks, position limits, balance verification
- **Usage**: Manual trade execution and strategy testing

**`TradingBotCLI.show_trade_history(days: int = 30)`**
- **Purpose**: Display historical trade data with filtering options
- **Parameters**: Number of days to retrieve (default 30)
- **Features**: Pagination, sorting, P&L calculation, export options
- **Display**: Trade details, execution prices, fees, performance metrics
- **Usage**: Trade analysis, performance review, audit trails

#### Configuration Management
**`TradingBotCLI.configure_strategy(strategy_name: str)`**
- **Purpose**: Interactive strategy configuration through CLI
- **Parameters**: Strategy name to configure
- **Features**: Parameter validation, help text, configuration persistence
- **Interface**: Step-by-step configuration wizard with validation
- **Usage**: Strategy setup, parameter tuning, configuration updates

### Web Dashboard

#### Dashboard Core
**`TradingDashboard.__init__(config: Dict, auth_manager: SessionManager)`**
- **Purpose**: Initialize web dashboard with authentication and configuration
- **Parameters**: Dashboard configuration, session manager instance
- **Features**: Route registration, template engine, security middleware
- **Usage**: Web interface initialization and server setup

**`TradingDashboard.get_portfolio_data() -> Dict`**
- **Purpose**: Retrieve real-time portfolio data for dashboard display
- **Returns**: Dictionary with positions, balances, and performance metrics
- **Features**: Real-time updates, caching, error handling
- **Usage**: Dashboard data updates and visualization components

**`TradingDashboard.get_performance_charts() -> Dict`**
- **Purpose**: Generate performance chart data for web visualization
- **Returns**: Chart configuration and data for multiple time periods
- **Features**: Multiple chart types, responsive design, export options
- **Usage**: Performance visualization and trend analysis

#### Real-time Features
**`TradingDashboard.websocket_handler(websocket: WebSocket)`**
- **Purpose**: Handle WebSocket connections for real-time data updates
- **Parameters**: WebSocket connection object
- **Features**: Live price feeds, trade notifications, system alerts
- **Usage**: Real-time dashboard updates without page refresh

**`TradingDashboard.stream_trade_updates()`**
- **Purpose**: Stream live trade execution updates to connected clients
- **Features**: Real-time notifications, trade confirmation, error alerts
- **Usage**: Live trading monitoring and execution feedback

### Session Management

#### Authentication System
**`SessionManager.__init__(secret_key: str, session_timeout: int = 3600)`**
- **Purpose**: Initialize session management with security configuration
- **Parameters**: JWT secret key, session timeout in seconds
- **Features**: Token generation, validation, refresh handling
- **Usage**: User authentication and session lifecycle management

**`SessionManager.authenticate_user(username: str, password: str) -> Optional[str]`**
- **Purpose**: Authenticate user credentials and create session token
- **Parameters**: Username and password for authentication
- **Returns**: JWT token on success, None on failure
- **Features**: Password hashing, rate limiting, security logging
- **Usage**: User login and session creation

**`SessionManager.validate_session(token: str) -> Optional[Dict]`**
- **Purpose**: Validate session token and extract user information
- **Parameters**: JWT token to validate
- **Returns**: User data dictionary or None if invalid
- **Features**: Token expiration, signature validation, user lookup
- **Usage**: Request authentication and authorization

#### Session Lifecycle
**`SessionManager.refresh_session(refresh_token: str) -> Optional[str]`**
- **Purpose**: Refresh expired session using refresh token
- **Parameters**: Valid refresh token
- **Returns**: New access token or None if invalid
- **Features**: Secure token rotation, expiration handling
- **Usage**: Maintain user sessions without re-authentication

**`SessionManager.logout_user(token: str) -> bool`**
- **Purpose**: Invalidate user session and cleanup resources
- **Parameters**: Session token to invalidate
- **Returns**: Success status
- **Features**: Token blacklisting, cleanup, security logging
- **Usage**: User logout and session termination

### Security Framework

#### Access Control
**`SecurityManager.check_permission(user: Dict, resource: str, action: str) -> bool`**
- **Purpose**: Check user permissions for specific resource and action
- **Parameters**: User data, resource identifier, action type
- **Returns**: Permission granted status
- **Features**: Role-based access control, resource hierarchies
- **Usage**: Authorization checks throughout the application

**`SecurityManager.encrypt_sensitive_data(data: str) -> str`**
- **Purpose**: Encrypt sensitive data for secure storage
- **Parameters**: Plain text data to encrypt
- **Returns**: Encrypted data string
- **Features**: AES encryption, key management, salt generation
- **Usage**: API key storage, configuration encryption

**`SecurityManager.audit_log(user: str, action: str, resource: str, result: str)`**
- **Purpose**: Log security-relevant events for audit trails
- **Parameters**: User identifier, action performed, resource accessed, result
- **Features**: Structured logging, log rotation, compliance support
- **Usage**: Security monitoring and compliance reporting

## Integration Points

### With Core System
- **core/executor.py**: Trade execution commands and real-time status updates
- **core/safety.py**: Risk monitoring displays and safety control interfaces
- **core/state.py**: State management through UI configuration and monitoring

### With Storage Layer
- **storage/repo.py**: Trade history retrieval and performance data display
- **storage/artifacts.py**: Model management interfaces and deployment controls

### With AI Layer
- **ai/baseline.py**: Strategy configuration and performance monitoring
- **ai/bo_suggester.py**: Optimization parameter tuning and result visualization

## Usage Examples

### CLI Operations
```python
# Initialize CLI
cli = TradingBotCLI(config)

# Start interactive mode
await cli.start_interactive_mode()

# Execute trade via CLI
await cli.execute_trade_command("BTCUSDC", "BUY", "0.1")

# Show portfolio status
await cli.show_portfolio_status()

# Configure strategy
await cli.configure_strategy("grid_trader")

# Display trade history
await cli.show_trade_history(days=7)
```

### Web Dashboard Setup
```python
# Initialize dashboard
dashboard = TradingDashboard(config, session_manager)

# Start web server
await dashboard.run(host="0.0.0.0", port=8080)

# Get portfolio data for API
portfolio_data = await dashboard.get_portfolio_data()

# Generate performance charts
chart_data = await dashboard.get_performance_charts()
```

### Session Management
```python
# Initialize session manager
session_mgr = SessionManager(
    secret_key="your-secret-key",
    session_timeout=3600
)

# User authentication
token = await session_mgr.authenticate_user("admin", "password")
if token:
    print("Authentication successful")

# Validate session
user_data = await session_mgr.validate_session(token)
if user_data:
    print(f"Valid session for user: {user_data['username']}")

# Refresh expired session
new_token = await session_mgr.refresh_session(refresh_token)

# Logout user
await session_mgr.logout_user(token)
```

### Security Operations
```python
# Initialize security manager
security = SecurityManager(config)

# Check permissions
has_permission = security.check_permission(
    user=user_data,
    resource="trading",
    action="execute"
)

# Encrypt sensitive data
encrypted_api_key = security.encrypt_sensitive_data(api_key)

# Audit logging
security.audit_log(
    user="admin",
    action="trade_execution",
    resource="BTCUSDC",
    result="success"
)
```

### Real-time Dashboard Features
```python
# WebSocket handling for real-time updates
@dashboard.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Stream real-time data
    while True:
        portfolio_data = await dashboard.get_portfolio_data()
        await websocket.send_json(portfolio_data)
        await asyncio.sleep(1)

# API endpoints for data retrieval
@dashboard.get("/api/trades")
async def get_trades(days: int = 30):
    return await dashboard.get_trade_history(days)

@dashboard.post("/api/execute-trade")
async def execute_trade(trade_request: TradeRequest):
    return await dashboard.execute_trade(trade_request)
```

### Configuration Management
```python
# Interactive configuration through CLI
async def configure_trading_parameters():
    cli = TradingBotCLI(config)
    
    # Grid trading configuration
    await cli.configure_strategy("grid_trader")
    
    # Risk management settings
    await cli.configure_risk_parameters()
    
    # Exchange connection setup
    await cli.configure_exchange_settings()
    
    # Save configuration
    await cli.save_configuration()
```
