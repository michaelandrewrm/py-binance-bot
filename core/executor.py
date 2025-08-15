"""
Trading Loop Executor

This module provides the main trading execution loop that integrates
safety checks, flatten operations, and strategy execution.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from core.state import BotState, BotStatus, save_state
from core.safety import SafetyManager, SafetyAction, RiskLimits
from core.flatten import FlattenExecutor, FlattenConfig, execute_flatten_operation
from exec.binance import BinanceClient
from core.sizing import FeeConfig

# Import unified configuration
try:
    from config import get_settings

    _unified_config_available = True
except ImportError:
    _unified_config_available = False

logger = logging.getLogger(__name__)


class TradingExecutor:
    """Main trading executor that runs the bot's trading loop with safety integration.

    This class coordinates the main trading activities including:
    - Safety checks and risk management
    - Strategy execution
    - Order management
    - State persistence
    - Emergency handling

    Attributes:
        client: Binance API client for order execution
        state: Current bot state including positions and orders
        safety_manager: Risk management and safety system
        flatten_config: Configuration for flatten operations
        state_save_path: Path where bot state is persisted
        running: Whether the trading loop is currently active
        last_safety_check: Timestamp of last safety evaluation
    """

    def __init__(
        self,
        client: BinanceClient,
        state: BotState,
        safety_manager: SafetyManager,
        flatten_config: Optional[FlattenConfig] = None,
        state_save_path: str = "bot_state.json",
    ) -> None:
        """Initialize trading executor with required components.

        Args:
            client: Configured Binance API client
            state: Bot state object containing positions and orders
            safety_manager: Safety system for risk management
            flatten_config: Optional configuration for flatten operations
            state_save_path: Path to save bot state (default: "bot_state.json")
        """
        self.client = client
        self.state = state
        self.safety_manager = safety_manager
        self.flatten_config = flatten_config or FlattenConfig()
        self.state_save_path = state_save_path
        self.running = False
        self.last_safety_check = datetime.now(timezone.utc)

    async def start_trading_loop(
        self, strategy_callback: Optional[Callable] = None
    ) -> None:
        """Start the main trading loop with continuous safety monitoring.

        The trading loop performs these steps continuously:
        1. Safety checks and risk evaluation
        2. Strategy execution (if provided)
        3. State persistence
        4. Error handling and recovery

        Args:
            strategy_callback: Optional async function that implements trading strategy.
                              Should accept (client, state) parameters and return None.

        Raises:
            Exception: If critical errors occur that cannot be handled
        """
        logger.info("Starting trading loop...")
        self.running = True
        self.state.status = BotStatus.RUNNING

        try:
            while self.running:
                loop_start = datetime.now(timezone.utc)

                # Check if manual resume is required
                if self.state.requires_manual_resume:
                    logger.warning("Manual resume required - pausing execution")
                    logger.warning(f"Flatten reason: {self.state.flatten_reason}")
                    self.state.status = BotStatus.PAUSED
                    await asyncio.sleep(30)  # Wait 30 seconds before checking again
                    continue

                # Execute safety checks
                safety_action = await self._execute_safety_checks()

                # Handle safety actions
                if safety_action == SafetyAction.EMERGENCY_STOP:
                    logger.critical("EMERGENCY STOP triggered - shutting down")
                    await self._handle_emergency_stop("Safety system emergency stop")
                    break

                elif safety_action == SafetyAction.FLATTEN:
                    logger.critical("FLATTEN action triggered")
                    await self._handle_flatten("Safety system triggered flatten")
                    continue  # Continue loop but with manual resume flag set

                elif safety_action == SafetyAction.PAUSE:
                    logger.warning(
                        "PAUSE action triggered - skipping strategy execution"
                    )
                    self.state.status = BotStatus.PAUSED
                    await asyncio.sleep(60)  # Wait 1 minute before retrying
                    continue

                elif safety_action == SafetyAction.CANCEL_ORDERS:
                    logger.warning("CANCEL_ORDERS action triggered")
                    await self._cancel_open_orders()
                    # Continue with strategy execution

                # If we reach here, trading is allowed (CONTINUE action)
                self.state.status = BotStatus.RUNNING

                # Execute trading strategy
                if strategy_callback:
                    try:
                        await strategy_callback(self.client, self.state)
                    except Exception as e:
                        logger.error(f"Strategy execution error: {e}")
                        self.state.record_error(str(e))

                        # Update safety metrics
                        self.safety_manager.update_metrics(
                            api_errors=self.state.error_count
                        )

                # Save state periodically
                await self._save_state_if_needed()

                # Sleep before next iteration
                loop_duration = (
                    datetime.now(timezone.utc) - loop_start
                ).total_seconds()
                sleep_time = max(1.0, 5.0 - loop_duration)  # Minimum 1 second sleep
                await asyncio.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Received interrupt signal - shutting down gracefully")
        except Exception as e:
            logger.critical(f"Critical error in trading loop: {e}")
            await self._handle_emergency_stop(f"Critical error: {e}")
        finally:
            await self._shutdown()

    async def stop_trading_loop(self) -> None:
        """Stop the trading loop gracefully.

        Sets the running flag to False, causing the main loop to exit cleanly
        after completing the current iteration.
        """
        logger.info("Stopping trading loop...")
        self.running = False

    async def manual_flatten(self, reason: str) -> bool:
        """Manually trigger a flatten operation to close all positions.

        This method allows operators to manually trigger position flattening
        when safety conditions are met or manual intervention is required.

        Args:
            reason: Human-readable reason for the manual flatten operation

        Returns:
            True if flatten operation was successful, False otherwise

        Raises:
            Exception: If flatten operation encounters critical errors
        """
        logger.warning(f"Manual flatten requested: {reason}")
        return await self._handle_flatten(f"Manual: {reason}")

    async def clear_manual_resume(self) -> bool:
        """Clear the manual resume flag after operator intervention.

        This method allows operators to resume automated trading after
        manual intervention has resolved any issues that caused the bot
        to require manual resume.

        Returns:
            True if manual resume flag was cleared successfully

        Note:
            Also resets all safety circuit breakers and saves state
        """
        if not self.state.requires_manual_resume:
            logger.info("No manual resume flag to clear")
            return True

        logger.warning("Clearing manual resume flag - operator intervention")
        self.state.clear_manual_resume()

        # Reset safety breakers
        self.safety_manager.reset_all_breakers()

        # Save state
        save_state(self.state, self.state_save_path)

        logger.info("Manual resume flag cleared - trading can resume")
        return True

    async def _execute_safety_checks(self) -> SafetyAction:
        """Execute comprehensive safety checks and return required action.

        This method coordinates all safety evaluations including:
        - PnL monitoring and drawdown analysis
        - Risk limit checks
        - Circuit breaker status
        - API error tracking

        Returns:
            SafetyAction indicating what action should be taken (CONTINUE, FLATTEN, HALT)

        Note:
            Safety status is logged every 5 minutes to reduce log noise
        """

        # Update performance metrics for safety evaluation
        total_pnl = self.state.get_realized_pnl() + self.state.get_unrealized_pnl()

        self.safety_manager.update_metrics(
            total_pnl=total_pnl,
            daily_pnl=self._calculate_daily_pnl(),
            unrealized_pnl=self.state.get_unrealized_pnl(),
            max_drawdown=self.state.metrics.max_drawdown,
            consecutive_losses=self.state.metrics.losing_trades,
            api_errors=self.state.error_count,
        )

        # Get safety evaluation
        action = self.safety_manager.evaluate_safety()

        # Log safety status periodically
        now = datetime.now(timezone.utc)
        if (now - self.last_safety_check).total_seconds() > 300:  # Every 5 minutes
            self._log_safety_status(action)
            self.last_safety_check = now

        return action

    async def _handle_flatten(self, reason: str) -> bool:
        """Handle position flattening operation with proper error handling.

        Executes a complete position flatten operation using the configured
        flatten strategy and handles any errors that occur during execution.

        Args:
            reason: Human-readable reason for the flatten operation

        Returns:
            True if flatten operation completed successfully, False otherwise

        Raises:
            Exception: If critical errors occur that cannot be handled
        """
        try:
            result = await execute_flatten_operation(
                self.client, self.state, reason, self.flatten_config
            )

            if result.success:
                logger.info(f"Flatten operation completed successfully")
                logger.info(f"Orders canceled: {result.orders_canceled}")
                logger.info(f"Positions closed: {len(result.positions_closed)}")
                logger.info(f"Total proceeds: {result.total_proceeds}")

                # Save state immediately
                save_state(self.state, self.state_save_path)
                return True
            else:
                logger.error(f"Flatten operation failed: {result.error_message}")
                # State still has manual resume flag set
                save_state(self.state, self.state_save_path)
                return False

        except Exception as e:
            logger.critical(f"Critical error during flatten: {e}")
            # Set manual resume flag anyway
            self.state.set_flatten_state(f"{reason} (ERROR: {e})")
            save_state(self.state, self.state_save_path)
            return False

    async def _handle_emergency_stop(self, reason: str) -> None:
        """Handle emergency stop with immediate position flattening.

        This method executes when critical safety conditions are met and
        trading must be halted immediately. It attempts to flatten positions
        if any exist and sets the bot status to error state.

        Args:
            reason: Human-readable reason for the emergency stop

        Note:
            Forces the trading loop to stop and saves state immediately
        """
        logger.critical(f"EMERGENCY STOP: {reason}")

        self.state.status = BotStatus.ERROR
        self.state.record_error(f"Emergency stop: {reason}")

        # Try to flatten if we have positions
        if any(not pos.is_flat for pos in self.state.positions.values()):
            logger.critical("Emergency stop with open positions - attempting flatten")
            await self._handle_flatten(f"Emergency stop: {reason}")

        # Force stop
        self.running = False
        save_state(self.state, self.state_save_path)

    async def _cancel_open_orders(self) -> None:
        """Cancel all open orders as a safety measure.

        Uses the FlattenExecutor to cancel all outstanding orders
        when safety conditions require order cancellation.

        Raises:
            Exception: If order cancellation encounters critical errors
        """
        try:
            flatten_executor = FlattenExecutor(self.client, self.flatten_config)
            canceled = await flatten_executor._cancel_all_orders(self.state)
            logger.info(f"Canceled {canceled} open orders due to safety action")

        except Exception as e:
            logger.error(f"Error canceling orders: {e}")
            self.state.record_error(f"Cancel orders error: {e}")

    async def _save_state_if_needed(self) -> None:
        """Save bot state periodically or when significant changes occur.

        State is saved in these conditions:
        - Active orders exist (to ensure order tracking persistence)
        - 5 minutes have passed since last save (periodic backup)
        - Manual save triggers
        """
        # Save every 5 minutes or if there are active orders
        if (
            len(self.state.active_orders) > 0
            or (datetime.now(timezone.utc) - self.state.last_update).total_seconds()
            > 300
        ):

            self.state.last_update = datetime.now(timezone.utc)
            save_state(self.state, self.state_save_path)

    async def _shutdown(self) -> None:
        """Execute graceful shutdown procedures.

        Performs cleanup operations including:
        - Setting bot status to stopping/stopped
        - Final state save
        - Closing API client sessions
        - Logging completion
        """
        logger.info("Executing shutdown procedures...")

        self.state.status = BotStatus.STOPPING

        # Final state save
        save_state(self.state, self.state_save_path)

        # Close client session if needed
        if hasattr(self.client, "close_session"):
            await self.client.close_session()

        self.state.status = BotStatus.STOPPED
        logger.info("Shutdown complete")

    def _calculate_daily_pnl(self) -> Decimal:
        """Calculate profit and loss for the current trading day.

        Returns:
            Decimal: Current day's PnL. In production this would track
                    daily trades, but simplified to total PnL for now

        Note:
            This is a simplified implementation. Production version should
            track trades by date and calculate daily performance accurately
        """
        # Simplified - in production would track daily trades
        return self.state.metrics.total_pnl

    def _log_safety_status(self, current_action: SafetyAction) -> None:
        """Log comprehensive safety status information.

        Provides detailed logging of safety system state including:
        - Current safety action recommendation
        - Active circuit breakers
        - Key performance metrics

        Args:
            current_action: Current safety action recommendation
        """
        status = self.safety_manager.get_safety_status()

        logger.info(f"Safety Status - Action: {current_action.value}")
        if status["active_breakers"]:
            logger.warning(f"Active breakers: {', '.join(status['active_breakers'])}")

        metrics = status["metrics"]
        logger.info(
            f"Metrics - PnL: {metrics['total_pnl']:.2f}, "
            f"Drawdown: {metrics['max_drawdown']:.2%}, "
            f"Errors: {metrics['api_errors']}"
        )

    @classmethod
    async def from_unified_config(
        cls, state: BotState, config=None
    ) -> "TradingExecutor":
        """Create TradingExecutor from unified configuration system.

        This factory method creates a fully configured TradingExecutor using
        the unified configuration system, automatically setting up all
        required components with appropriate defaults.

        Args:
            state: Bot state object containing positions and orders
            config: Settings instance. If None, gets current settings

        Returns:
            TradingExecutor: Fully configured executor instance

        Raises:
            ImportError: If unified config system is not available
            Exception: If configuration or component setup fails
        """
        if not _unified_config_available:
            raise ImportError("Unified config system not available")

        if config is None:
            config = get_settings()

        # Create Binance client from config
        from exec.binance import BinanceConfig

        binance_config = BinanceConfig.from_unified_config(config)
        client = BinanceClient(binance_config)
        await client.start_session()

        # Create risk limits from config
        risk_limits = RiskLimits(
            max_position_size=config.trading.max_position_size,
            max_daily_loss=config.trading.max_position_size
            * Decimal("0.1"),  # 10% of max position
            max_drawdown=Decimal("0.2"),  # 20% max drawdown
            max_order_size=config.grid.invest_per_grid * 2,  # 2x grid investment
            max_open_orders=config.grid.n_grids * 2,  # 2x grid levels
            price_deviation_threshold=Decimal("0.05"),  # 5% price movement
            consecutive_losses_limit=5,
            api_error_threshold=20,
        )

        # Create safety manager
        safety_manager = SafetyManager(risk_limits)

        # Create flatten config
        flatten_config = FlattenConfig()

        # Get state save path from cache directory
        state_save_path = str(config.get_cache_dir() / "bot_state.json")

        return cls(
            client=client,
            state=state,
            safety_manager=safety_manager,
            flatten_config=flatten_config,
            state_save_path=state_save_path,
        )


# Convenience function for CLI usage
async def run_trading_bot(
    client: Optional[BinanceClient] = None,
    state: Optional[BotState] = None,
    strategy_callback: Optional[Callable] = None,
    risk_limits: Optional[RiskLimits] = None,
    flatten_config: Optional[FlattenConfig] = None,
    state_save_path: Optional[str] = None,
    use_unified_config: bool = True,
) -> None:
    """Run the trading bot with comprehensive safety integration.

    This convenience function sets up and runs a complete trading bot instance
    with proper safety management, state persistence, and error handling.

    Args:
        client: Binance API client. If None and use_unified_config=True,
               will create from unified configuration
        state: Bot state object. If None, creates empty state for default symbol
        strategy_callback: Optional async function implementing trading strategy.
                          Should accept (client, state) parameters
        risk_limits: Risk management configuration. If None, uses defaults
        flatten_config: Position flattening configuration. If None, uses defaults
        state_save_path: Path to save bot state. If None, uses cache directory
        use_unified_config: Whether to use unified configuration system for
                           missing parameters

    Raises:
        Exception: If bot setup or execution encounters critical errors

    Note:
        This function runs indefinitely until stopped or error occurs
    """

    # Use unified config if requested and available
    if use_unified_config and _unified_config_available:
        if state is None:
            from core.state import create_empty_state

            config = get_settings()
            state = create_empty_state(config.trading.default_symbol)

        # Use unified config for missing parameters
        executor = await TradingExecutor.from_unified_config(state)

        # Override with provided parameters
        if client is not None:
            await executor.client.close_session()
            executor.client = client
        if flatten_config is not None:
            executor.flatten_config = flatten_config
        if state_save_path is not None:
            executor.state_save_path = state_save_path
        if risk_limits is not None:
            executor.safety_manager = SafetyManager(risk_limits)
    else:
        # Traditional initialization
        if client is None:
            raise ValueError("Binance client is required when not using unified config")
        if state is None:
            from core.state import create_empty_state

            # Use a default symbol if no config available
            state = create_empty_state("BTCUSDC")

        # Initialize safety manager
        safety_manager = SafetyManager(risk_limits or RiskLimits())

        # Create executor
        executor = TradingExecutor(
            client=client,
            state=state,
            safety_manager=safety_manager,
            flatten_config=flatten_config or FlattenConfig(),
            state_save_path=state_save_path or "bot_state.json",
        )

    # Start trading loop
    await executor.start_trading_loop(strategy_callback)
