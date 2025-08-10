# 🤖 Py-Binance-Bot

**Enterprise-Grade AI-Powered Cryptocurrency Trading System**

A sophisticated, production-ready trading bot featuring advanced grid trading strategies, AI-powered optimization, comprehensive risk management, and a unified configuration system. Built with Python 3.12+ and designed for both testnet and live trading.

## ✨ **Key Features**

### 🧠 **AI-Powered Trading Intelligence**
- **Multiple AI Models**: LSTM, GRU, and Transformer networks for market prediction
- **Bayesian Optimization**: Advanced hyperparameter tuning using Optuna
- **Walk-Forward Validation**: Robust backtesting with time-series cross-validation
- **Technical Indicators**: 50+ indicators including custom oscillators and momentum signals
- **Performance Analytics**: Comprehensive metrics including Sharpe ratio, maximum drawdown, and profit factor

### 🔧 **Unified Configuration System**
- **Multi-Source Config**: Environment variables, `.env` files, and YAML configuration
- **Proper Precedence**: OS env vars → .env → config.yaml (env-specific) → defaults
- **Type Validation**: Pydantic-based schema validation with clear error messages
- **Security First**: Secrets only in environment variables, never in source control
- **Multi-Environment**: Support for dev, stage, and prod with specific overrides
- **Variable Interpolation**: `${VAR}` and `${VAR:default}` syntax in YAML files

### 🏗️ **Professional Architecture**
- **Modular Design**: Clean separation of concerns with dependency injection
- **State Persistence**: Automatic state saving/loading for seamless restarts
- **Async/Await**: High-performance asynchronous execution
- **Comprehensive Testing**: 167+ unit tests with high coverage:
  - Safety Systems: 78% coverage
  - Position Sizing: 86% coverage  
  - Rate Limiting: 69% coverage
  - API Client: 68% coverage

### 🔌 **Grid Trading System**
- **Intelligent Grid Placement**: Dynamic grid level calculation
- **Rebalancing Logic**: Automatic grid adjustment based on market conditions
- **Position Tracking**: Real-time monitoring of grid performance
- **Risk Controls**: Position size limits and exposure management

### 🔌 **Robust Exchange Integration**
- **Smart Rate Limiting**: Adaptive request throttling to avoid API limits
- **Retry Logic**: Exponential backoff with jitter for failed requests
- **Health Monitoring**: Continuous API connectivity checks
- **Paper Trading**: Full simulation mode for strategy testing

---

## 🎯 **Core Components**

### 🛡️ **Safety & Risk Management** (`core/safety.py`)
The safety system is the cornerstone of the trading bot, providing multiple layers of protection:

- **Circuit Breakers**: Automatic trading suspension when thresholds are exceeded
- **Risk Limits**: Configurable position size, daily loss, and drawdown limits
- **Performance Monitoring**: Real-time tracking of trading metrics
- **Safety Events**: Comprehensive logging and alert system
- **Emergency Controls**: Instant position flattening capabilities

```python
# Example safety configuration
safety_manager = SafetyManager(
    max_position_size=Decimal("0.1"),    # 10% of portfolio
    max_daily_loss=Decimal("0.05"),      # 5% daily loss limit
    max_drawdown=Decimal("0.15"),        # 15% max drawdown
    max_consecutive_losses=5             # Circuit breaker threshold
)
```

### ⚙️ **Trading Executor** (`core/executor.py`)
The main coordination engine that orchestrates all trading activities:

- **Strategy Execution**: Coordinates between AI models and trading strategies
- **Safety Integration**: All trades pass through safety checks
- **State Management**: Maintains trading state and position tracking
- **Event Handling**: Processes market events and user commands

### 📏 **Position Sizing** (`core/sizing.py`)
Intelligent position sizing with exchange-specific calculations:

- **Dynamic Sizing**: Calculates optimal position sizes based on risk parameters
- **Fee Calculation**: Accurate fee estimation for maker/taker orders
- **Exchange Rules**: Respects tick size, step size, and minimum notional requirements
- **Grid Optimization**: Specialized sizing for grid trading strategies

### 🔄 **Grid Trading Engine** (`core/grid_engine.py`)
Advanced grid trading implementation:

- **Grid Calculation**: Dynamic level calculation based on market volatility
- **Order Management**: Automated placement and cancellation of grid orders
- **Rebalancing**: Intelligent grid adjustment as market moves
- **Performance Tracking**: Real-time grid efficiency monitoring

### 🚨 **Emergency Flattening** (`core/flatten.py`)
Emergency position management system:

- **Instant Flattening**: Rapid position closure in emergency situations
- **Order Cancellation**: Mass cancellation of open orders
- **TWAP Execution**: Time-weighted average price execution for large positions
- **Result Tracking**: Detailed logging of flattening operations

### 💾 **State Persistence** (`core/state.py`)
Robust state management for operational continuity:

- **Automatic Persistence**: Regular state snapshots to disk
- **Crash Recovery**: Seamless restart after unexpected shutdowns
- **Position Tracking**: Maintains accurate position and order state
- **Historical Records**: Complete audit trail of all operations

---

## 🤖 **AI & Machine Learning**

### 🧠 **Baseline Strategies** (`ai/baseline.py`)
Technical analysis and baseline trading strategies:

- **Technical Indicators**: 20+ indicators including SMA, EMA, RSI, Bollinger Bands
- **Volatility Strategies**: Advanced volatility-based trading algorithms
- **Signal Generation**: Multi-timeframe signal aggregation
- **Feature Engineering**: Automated feature extraction from price data

### 🔬 **Model Validation** (`ai/validation.py`)
Rigorous model testing and validation framework:

- **Walk-Forward Analysis**: Time-series aware validation methodology
- **Cross-Validation**: Multiple validation periods for robust testing
- **Performance Metrics**: Comprehensive evaluation including Sharpe ratio, drawdown
- **Overfitting Detection**: Advanced techniques to prevent data leakage

### 🎯 **Hyperparameter Optimization** (`ai/bo_suggester.py`)
Automated model optimization using Bayesian methods:

- **Bayesian Optimization**: Efficient hyperparameter search
- **Multi-Objective Optimization**: Balance between returns and risk
- **Automated Tuning**: Hands-off optimization process
- **Parameter Tracking**: Complete optimization history and results

---

## 🔌 **Exchange Integration**

### 📡 **Binance Client** (`exec/binance.py`)
Production-ready Binance API integration:

- **Async Operations**: High-performance asynchronous API calls
- **Authentication**: Secure API key management and signature generation
- **Error Handling**: Comprehensive error handling with automatic retries
- **Health Monitoring**: Continuous API status monitoring
- **Order Management**: Full order lifecycle management

### ⏱️ **Rate Limiting** (`exec/rate_limit.py`)
Intelligent request throttling system:

- **Dynamic Throttling**: Adaptive rate limiting based on API response
- **Weight Management**: Tracks API weight consumption
- **Burst Handling**: Manages short bursts while maintaining long-term limits
- **Multiple Endpoints**: Per-endpoint rate limiting configuration

### 📄 **Paper Trading** (`exec/paper_broker.py`)
Full-featured simulation environment:

- **Realistic Execution**: Simulates real market conditions
- **Slippage Modeling**: Accurate transaction cost simulation
- **Performance Tracking**: Complete performance analytics
- **Risk-Free Testing**: Test strategies without capital risk

---

## 🧪 **Simulation & Backtesting**

### 🏃 **Simulation Engine** (`sim/engine.py`)
High-performance backtesting framework:

- **Historical Simulation**: Run strategies on historical data
- **Event-Driven Architecture**: Realistic order execution simulation
- **Multi-Timeframe**: Support for multiple data frequencies
- **Portfolio Simulation**: Full portfolio-level backtesting

### 📊 **Performance Evaluation** (`sim/evaluation.py`)
Comprehensive performance analysis:

- **Standard Metrics**: Returns, Sharpe ratio, maximum drawdown
- **Risk Analysis**: Value at Risk (VaR), conditional VaR
- **Trade Analysis**: Win rate, average trade duration, profit factor
- **Benchmark Comparison**: Performance vs. buy-and-hold strategies

### 💰 **Slippage Modeling** (`sim/slippage.py`)
Realistic transaction cost simulation:

- **Market Impact**: Models price impact of large orders
- **Bid-Ask Spread**: Simulates spread costs
- **Liquidity Modeling**: Order book depth simulation
- **Dynamic Costs**: Time-varying transaction costs

---

## 📊 **Data Management**

### 📈 **Data Loader** (`data/loader.py`)
Efficient historical data management:

- **Multi-Source**: Support for multiple data providers
- **Caching System**: Intelligent data caching for performance
- **Real-Time Updates**: Live data streaming capabilities
- **Data Validation**: Comprehensive data quality checks

### 🏗️ **Data Schema** (`data/schema.py`)
Structured data models and validation:

- **Type Safety**: Pydantic-based data validation
- **Schema Evolution**: Backward-compatible schema updates
- **Serialization**: Efficient data serialization/deserialization
- **API Contracts**: Clear data interfaces between components

---

## 🧪 **Testing & Quality Assurance**

The trading bot features a comprehensive test suite with **167+ unit tests** ensuring reliability and correctness:

### 🛡️ **Safety System Tests** (`tests/test_safety.py`)
- **78% Code Coverage** of critical safety components
- Circuit breaker functionality testing
- Risk limit validation and enforcement
- Performance metrics tracking accuracy
- Safety event generation and callbacks

### ⚙️ **Execution Tests** (`tests/test_executor.py`)
- Trading executor initialization and configuration
- Safety manager integration testing
- State persistence and recovery testing
- Component interaction validation

### 📏 **Sizing Tests** (`tests/test_sizing.py`)
- **86% Code Coverage** of position sizing logic
- Fee calculation accuracy testing
- Exchange rule compliance validation
- Grid order sizing optimization

### ⏱️ **Rate Limiting Tests** (`tests/test_rate_limiting.py`)
- **69% Code Coverage** of rate limiting system
- API throttling behavior validation
- Concurrent request handling testing
- Rate limit recovery mechanisms

### 📡 **API Client Tests** (`tests/test_binance_client.py`)
- **68% Code Coverage** of Binance integration
- Authentication and signature testing
- Retry logic and error handling validation
- Health check and connectivity monitoring

### 🚀 **Running Tests**
```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov --cov-report=html

# Run specific test module
pytest tests/test_safety.py -v

# Run tests in quiet mode
pytest -q
```

---

## 🚀 **Deployment & Operations**

### 🐳 **Containerization** (Planned)
- Docker containerization for consistent deployments
- Multi-stage builds for optimized production images
- Environment-specific configuration management
- Health check endpoints for monitoring

### 📊 **Monitoring & Observability**
- **Performance Metrics**: Real-time trading performance tracking
- **System Health**: API connectivity and system status monitoring
- **Alert System**: Configurable alerts for safety events and system issues
- **Audit Logging**: Complete audit trail of all trading activities

### 🔒 **Security**
- **API Key Security**: Secure credential management
- **Permission Validation**: Minimum required API permissions
- **Network Security**: HTTPS-only communication
- **Input Validation**: Comprehensive input sanitizationisticated, production-ready automated trading system built with Python that combines advanced AI models, robust risk management, and professional-grade architecture for cryptocurrency trading on Binance. This bot features comprehensive safety systems, grid trading strategies, machine learning capabilities, and enterprise-level testing with 167+ test cases covering critical trading components.

## 🏗️ **System Architecture**

This trading bot is built with a modular, enterprise-grade architecture that separates concerns and ensures reliability, testability, and maintainability.

```bash
py-binance-bot/
├── 🎯 core/                    # Core Trading Engine
│   ├── executor.py             # Main trading execution coordinator
│   ├── safety.py               # Risk management & circuit breakers
│   ├── sizing.py               # Position sizing & fee calculations
│   ├── flatten.py              # Emergency position flattening
│   ├── grid_engine.py          # Grid trading strategy engine
│   └── state.py                # State persistence & management
├── 🔌 exec/                    # Exchange Interface Layer
│   ├── binance.py              # Binance API client with retry logic
│   ├── rate_limit.py           # Smart rate limiting & request throttling
│   └── paper_broker.py         # Paper trading simulation
├── 🤖 ai/                      # AI & Machine Learning
│   ├── baseline.py             # Technical indicators & baseline strategies
│   ├── validation.py           # Model validation & walk-forward analysis
│   └── bo_suggester.py         # Bayesian optimization for hyperparameters
├── 📊 data/                    # Data Management
│   ├── loader.py               # Historical data fetching & caching
│   └── schema.py               # Data models & validation schemas
├── 🧪 sim/                     # Simulation & Backtesting
│   ├── engine.py               # Backtesting simulation engine
│   ├── evaluation.py           # Performance metrics & analysis
│   └── slippage.py             # Realistic slippage modeling
├── 💾 storage/                 # Data Persistence
│   ├── artifacts.py            # Model & config artifact management
│   └── repo.py                 # Data repository layer
├── 📈 reports/                 # Analytics & Reporting
│   ├── end_of_run.py           # Comprehensive trading reports
│   └── hodl_benchmark.py       # Buy-and-hold benchmark comparison
├── 🖥️ ui/                      # User Interface
│   ├── cli.py                  # Command-line interface
│   └── dashboard.py            # Web dashboard (future)
├── 🧪 tests/                   # Comprehensive Test Suite (167+ tests)
│   ├── test_safety.py          # Risk management tests (78% coverage)
│   ├── test_executor.py        # Trading execution tests
│   ├── test_sizing.py          # Position sizing tests (86% coverage)
│   ├── test_rate_limiting.py   # Rate limiting tests (69% coverage)
│   ├── test_binance_client.py  # API client tests (68% coverage)
│   ├── test_safety_clean.py    # Additional safety system tests
│   └── test_executor_simple.py # Simplified executor tests
├── main.py                     # Application entry point
├── config.py                   # Configuration management
└── requirements.txt            # Python dependencies
```

## ⚡ **Key Features**

### 🛡️ **Enterprise-Grade Safety**
- **Multi-Layer Risk Management**: Circuit breakers, position limits, drawdown protection
- **Real-Time Monitoring**: Performance metrics tracking with automatic alerts
- **Emergency Controls**: Instant position flattening and order cancellation
- **Safety Events**: Comprehensive logging and callback system for risk events

### 🤖 **Advanced AI Trading**
- **LSTM Neural Networks**: Deep learning models for price prediction
- **Technical Indicators**: 20+ built-in indicators (SMA, EMA, RSI, Bollinger Bands, etc.)
- **Strategy Validation**: Walk-forward analysis and cross-validation
- **Hyperparameter Optimization**: Automated model tuning with Bayesian optimization

### 🏗️ **Professional Architecture**
- **Modular Design**: Clean separation of concerns with dependency injection
- **State Persistence**: Automatic state saving/loading for seamless restarts
- **Async/Await**: High-performance asynchronous execution
- **Comprehensive Testing**: 167+ unit tests with 78% safety system coverage

### � **Grid Trading System**
- **Intelligent Grid Placement**: Dynamic grid level calculation
- **Rebalancing Logic**: Automatic grid adjustment based on market conditions
- **Position Tracking**: Real-time monitoring of grid performance
- **Risk Controls**: Position size limits and exposure management

### 🔌 **Robust Exchange Integration**
- **Smart Rate Limiting**: Adaptive request throttling to avoid API limits
- **Retry Logic**: Exponential backoff with jitter for failed requests
- **Health Monitoring**: Continuous API connectivity checks
- **Paper Trading**: Full simulation mode for strategy testing

---

## ⚙️ **Configuration System**

The trading bot features a comprehensive, unified configuration system that provides:

### 🔧 **Multi-Source Configuration**
- **Environment Variables**: OS-level configuration (highest priority)
- **`.env` Files**: Local development secrets and overrides
- **YAML Configuration**: Structured, version-controlled settings with environment-specific sections
- **Hardcoded Defaults**: Sensible defaults for all settings (lowest priority)

### 🌍 **Environment Support**
```bash
# Development environment
ENVIRONMENT=dev     # Debug mode, testnet, smaller positions

# Staging environment  
ENVIRONMENT=stage   # Production-like, testnet, testing

# Production environment
ENVIRONMENT=prod    # Live trading, enhanced security, monitoring
```

### 🔒 **Security-First Design**
- **Secrets Management**: API keys and sensitive data only in environment variables
- **Type Validation**: Pydantic-based schema validation with clear error messages
- **Environment Isolation**: Separate configurations for dev/stage/prod
- **Git Safety**: No secrets in version control, comprehensive `.env.example`

### 📝 **Easy Configuration**
```python
from config import get_settings

# Get typed, validated configuration
settings = get_settings()

# Access configuration values
print(f"Trading symbol: {settings.trading.default_symbol}")
print(f"Grid levels: {settings.grid.n_grids}")
print(f"AI enabled: {settings.ai.enabled}")

# Environment-specific behavior
if settings.is_development():
    print("Running in development mode")
```

### 🛠️ **Configuration Tools**
```bash
# Validate your configuration
python validate_config.py

# Explore configuration features
python config_demo.py

# See integration examples
python integration_example.py
```

For complete configuration documentation, see [Configuration Guide](CONFIG_README.md).

---

## 🚀 **Quick Start**

### Prerequisites
- **Python 3.10+** (3.12+ recommended)
- **Binance account** with API keys (for live trading)
- **Virtual environment** (strongly recommended)
- **Git** for cloning the repository

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/michaelandrewrm/py-binance-bot.git
cd py-binance-bot
```

2. **Set up virtual environment**
```bash
# Create virtual environment
python -m venv trading_bot_env

# Activate (Linux/macOS)
source trading_bot_env/bin/activate

# Activate (Windows)
trading_bot_env\Scripts\activate
```

3. **Install dependencies**
```bash
# Install all dependencies
pip install -r requirements.txt

# Or install minimal dependencies only
pip install -r requirements-minimal.txt
```

4. **Configure environment**
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env  # or use your preferred editor
```

5. **Configure the system** (see [Configuration Guide](CONFIG_README.md))
```bash
# Validate your configuration
python validate_config.py

# Test configuration system
python config_demo.py

# See integration examples
python integration_example.py
```

6. **Configure API credentials** (for live trading)
```bash
# Add to .env file
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here
BINANCE_TESTNET=true  # Set to false for live trading
```

7. **Verify installation**
```bash
# Run tests to verify everything works
pytest -q

# Check CLI help
python main.py --help
```

### First Steps

1. **Run in simulation mode**
```bash
# Start with paper trading to test the system
python main.py backtest --symbol BTCUSDC --strategy grid --paper-trading
```

2. **Explore available commands**
```bash
# View all available commands
python main.py --help

# Get help for specific commands
python main.py backtest --help
python main.py live --help
```

3. **Run the test suite**
```bash
# Run all 167+ tests
pytest

# Run with detailed coverage report
pytest --cov --cov-report=html
open htmlcov/index.html  # View coverage report
```

### Configuration Examples

---

## 🎛️ **Configuration**

### Environment Variables (.env)
```bash
# API Configuration
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
BINANCE_TESTNET=true  # Use testnet for testing

# Application Settings
LOG_LEVEL=INFO
ENABLE_PAPER_TRADING=true
STATE_SAVE_INTERVAL=60  # seconds

# Database/Storage
DATA_CACHE_DIR=./data_cache
STATE_SAVE_PATH=./state
MODEL_ARTIFACTS_PATH=./models
```

### Trading Parameters (config.py)
```python
# Symbol configuration
SYMBOLS = {
    "BTCUSDC": {
        "base_asset": "BTC",
        "quote_asset": "USDC",
        "tick_size": 0.1,
        "step_size": 0.00001,
        "min_qty": 0.0001,
        "min_notional": 10,
        "investment_amount": 1000,
        "default_grid_levels": 7,
        "default_grid_range_pct": 0.06,
    }
}

# Grid trading settings
GRID_CONFIG = {
    "default_levels": 10,
    "spacing_percentage": 0.5,  # 0.5% between levels
    "rebalance_threshold": 0.02,  # 2% price movement triggers rebalance
    "max_position_ratio": 0.8,  # Max 80% of capital in positions
}
```

### Risk Management Configuration
```python
# Safety limits (strongly recommended to customize for your risk tolerance)
RISK_LIMITS = {
    "max_position_size": Decimal("0.1"),    # 10% of portfolio per position
    "max_daily_loss": Decimal("0.05"),      # 5% daily loss limit
    "max_drawdown": Decimal("0.15"),        # 15% maximum drawdown
    "max_consecutive_losses": 5,            # Circuit breaker threshold
    "position_timeout_hours": 24,           # Force close old positions
}

# Circuit breaker settings
CIRCUIT_BREAKER = {
    "failure_threshold": 5,      # Failures before activation
    "recovery_timeout": 300,     # Seconds before attempting recovery
    "max_recovery_attempts": 3,  # Maximum recovery attempts
}
```

### Advanced Configuration
```python
# AI/ML settings
AI_CONFIG = {
    "model_retrain_interval": 168,  # Hours between model retraining
    "prediction_confidence_threshold": 0.6,  # Minimum confidence for trades
    "feature_engineering": {
        "lookback_periods": [5, 10, 20, 50],
        "technical_indicators": ["sma", "ema", "rsi", "bollinger"],
    }
}

# Rate limiting configuration
RATE_LIMITS = {
    "requests_per_minute": 1200,   # Binance default limit
    "orders_per_minute": 100,      # Conservative order limit
    "weight_per_minute": 1200,     # API weight limit
    "burst_allowance": 0.1,        # 10% burst capacity
}
```

---