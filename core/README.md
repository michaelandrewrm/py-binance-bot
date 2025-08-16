# Core Directory

## Purpose

The core directory contains the fundamental trading infrastructure that orchestrates all trading operations, including the main trading executor, safety systems, state management, and specialized trading engines.

## Architecture

The core layer serves as the central nervous system of the trading bot, coordinating between:
- Main trading execution loop with strategy integration
- Multi-layer safety and risk management systems
- Persistent state management and recovery
- Specialized trading engines (grid, flatten operations)
- Advanced error handling and market monitoring

## Files Overview

### Core Trading Infrastructure

#### `executor.py` - Trading Executor
- **Main Trading Loop**: Asynchronous execution engine with strategy callbacks and market monitoring
- **Strategy Integration**: Pluggable strategy system supporting multiple trading algorithms
- **Risk Integration**: Real-time safety monitoring with circuit breaker integration
- **State Persistence**: Automatic state saving and recovery with configurable intervals
- **Market Data Processing**: Real-time price monitoring and signal processing
- **Error Recovery**: Comprehensive error handling with graceful degradation

#### `safety.py` - Safety Manager
- **Multi-layer Protection**: Position limits, daily loss limits, and portfolio risk monitoring
- **Circuit Breakers**: Automatic trading halt on excessive losses or unusual market conditions
- **Real-time Monitoring**: Continuous risk assessment with configurable thresholds
- **Manual Controls**: Emergency flatten operations and manual trading halts
- **Risk Metrics**: Advanced risk calculations including VaR and position correlation

#### `state.py` - State Management
- **Persistent State**: Complete trading bot state serialization and recovery
- **Thread Safety**: Concurrent access protection for multi-threaded operations
- **State Validation**: Integrity checking and error recovery for corrupted state
- **Migration Support**: State schema evolution and backward compatibility
- **Performance Optimization**: Efficient state updates and lazy loading

### Specialized Engines

#### `grid_engine.py` - Grid Trading Engine
- **Grid Management**: Dynamic grid level creation and order management
- **Order Coordination**: Intelligent order placement and execution tracking
- **Profit Calculation**: Real-time P&L tracking and profit realization
- **Grid Rebalancing**: Automatic grid adjustment based on market movement
- **Risk Controls**: Grid-specific risk limits and position sizing

#### `flatten.py` - Emergency Operations
- **Emergency Flatten**: Immediate position closure and order cancellation
- **Graceful Shutdown**: Controlled trading bot shutdown with state preservation
- **Recovery Operations**: Post-flatten state cleanup and recovery procedures
- **Safety Integration**: Coordinated emergency response with safety systems

### Enhanced Systems

#### `advanced_safety.py` - Advanced Risk Management
- **Market Regime Detection**: Real-time market condition analysis and adaptation
- **Dynamic Risk Adjustment**: Adaptive risk limits based on market volatility
- **Alert System**: Multi-level alert system with notification integration
- **Correlation Monitoring**: Cross-asset correlation analysis for portfolio risk
- **Stress Testing**: Real-time stress testing and scenario analysis

#### `enhanced_error_handling.py` - Error Management
- **Categorized Error Handling**: Systematic error classification and response
- **Recovery Strategies**: Automatic error recovery with fallback mechanisms
- **Error Analytics**: Error pattern analysis and predictive error prevention
- **Monitoring Integration**: Error metric tracking and alerting

## Key Functions and Classes

### Trading Execution
- `TradingExecutor`: Main trading loop with strategy integration and risk monitoring
- `run_trading_bot()`: Primary entry point for trading bot execution
- `process_strategy_signal()`: Strategy signal processing and order execution

### Safety Management
- `SafetyManager`: Comprehensive risk monitoring and circuit breaker system
- `check_risk_limits()`: Real-time risk assessment and limit checking
- `execute_emergency_halt()`: Emergency trading halt with state preservation

### State Management
- `TradingState`: Complete bot state representation with serialization
- `load_state()`: State loading with validation and migration support
- `save_state()`: Atomic state persistence with backup management

### Grid Trading
- `GridEngine`: Complete grid trading implementation with dynamic management
- `GridConfig`: Grid configuration with intelligent parameter validation
- `calculate_grid_levels()`: Dynamic grid level calculation and optimization

## Integration Points

### With AI Layer
- **ai/baseline.py**: Strategy signal generation and market analysis
- **ai/bo_suggester.py**: Parameter optimization and strategy tuning

### With Data Layer
- **data/loader.py**: Market data consumption for decision making
- **data/schema.py**: Market data structures and validation

### With Storage Layer
- **storage/repo.py**: Trade execution and state persistence
- **storage/artifacts.py**: Strategy configuration and model storage

### With Execution Layer
- **exec/binance.py**: Exchange API integration and order execution
- **exec/paper_broker.py**: Paper trading simulation and testing

## Usage Examples

### Basic Trading Execution
```python
# Initialize and run trading bot
executor = TradingExecutor(client, state, strategy_callback)
await executor.run(risk_limits, state_save_path)

# Emergency operations
await execute_flatten_operation(client, state, "market_crash")
```

### Grid Trading
```python
# Setup grid trading
grid_config = GridConfig(
    center_price=50000,
    grid_spacing=100,
    num_levels_up=5,
    num_levels_down=5
)
grid_engine = GridEngine(grid_config)
await grid_engine.execute_strategy(client, state)
```

### Safety Management
```python
# Configure safety systems
safety_manager = SafetyManager(risk_limits)
risk_status = await safety_manager.check_risk_limits(state, current_price)

# Advanced safety with market monitoring
advanced_safety = AdvancedSafetyManager()
market_regime = await advanced_safety.detect_market_regime(market_data)
```
