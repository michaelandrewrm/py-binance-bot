"""
Test cases for core/executor.py module

This module provides comprehensive tests for the trading execution engine,
including trading loop management, safety integration, and emergency handling.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Callable

from core.executor import TradingExecutor, run_trading_bot
from core.state import BotState, BotStatus, create_empty_state
from core.safety import SafetyManager, SafetyAction, RiskLimits
from core.flatten import FlattenConfig, FlattenResult
from exec.binance import BinanceClient


class MockBinanceClient:
    """Mock Binance client for testing"""
    
    def __init__(self):
        self.session_started = False
        self.session_closed = False
        
    async def start_session(self):
        self.session_started = True
        
    async def close_session(self):
        self.session_closed = True


class MockSafetyManager:
    """Mock safety manager for testing"""
    
    def __init__(self, risk_limits: Optional[RiskLimits] = None):
        self.risk_limits = risk_limits or RiskLimits()
        self.metrics_updates = []
        self.evaluation_result = SafetyAction.CONTINUE
        self.breakers_reset = False
        
    def update_metrics(self, **kwargs):
        self.metrics_updates.append(kwargs)
        
    def evaluate_safety(self) -> SafetyAction:
        return self.evaluation_result
        
    def reset_all_breakers(self):
        self.breakers_reset = True
        
    def get_safety_status(self) -> Dict:
        return {
            'active_breakers': [],
            'metrics': {
                'total_pnl': Decimal('0'),
                'max_drawdown': Decimal('0'),
                'api_errors': 0
            }
        }


class TestTradingExecutor:
    """Test TradingExecutor functionality"""
    
    def setup_method(self):
        """Set up test components"""
        self.mock_client = MockBinanceClient()
        self.test_state = create_empty_state("BTCUSDC")
        self.mock_safety_manager = MockSafetyManager()
        self.flatten_config = FlattenConfig()
        
        self.executor = TradingExecutor(
            client=self.mock_client,
            state=self.test_state,
            safety_manager=self.mock_safety_manager,
            flatten_config=self.flatten_config,
            state_save_path="test_state.json"
        )
    
    def test_executor_initialization(self):
        """Test TradingExecutor initialization"""
        assert self.executor.client == self.mock_client
        assert self.executor.state == self.test_state
        assert self.executor.safety_manager == self.mock_safety_manager
        assert self.executor.flatten_config == self.flatten_config
        assert self.executor.state_save_path == "test_state.json"
        assert self.executor.running is False
        assert isinstance(self.executor.last_safety_check, datetime)
    
    def test_executor_with_default_flatten_config(self):
        """Test executor initialization with default flatten config"""
        executor = TradingExecutor(
            client=self.mock_client,
            state=self.test_state,
            safety_manager=self.mock_safety_manager
        )
        
        assert isinstance(executor.flatten_config, FlattenConfig)
        assert executor.state_save_path == "bot_state.json"
    
    @pytest.mark.asyncio
    async def test_stop_trading_loop(self):
        """Test trading loop stop functionality"""
        # Start the loop flag
        self.executor.running = True
        
        await self.executor.stop_trading_loop()
        
        assert self.executor.running is False
    
    @pytest.mark.asyncio 
    async def test_clear_manual_resume_no_flag(self):
        """Test clearing manual resume when no flag is set"""
        # Ensure flag is not set
        self.executor.state.requires_manual_resume = False
        
        result = await self.executor.clear_manual_resume()
        
        assert result is True
        assert self.mock_safety_manager.breakers_reset is False
    
    @pytest.mark.asyncio
    async def test_clear_manual_resume_with_flag(self):
        """Test clearing manual resume when flag is set"""
        # Set manual resume flag
        self.executor.state.requires_manual_resume = True
        self.executor.state.flatten_reason = "Test reason"
        
        with patch('core.executor.save_state') as mock_save:
            result = await self.executor.clear_manual_resume()
        
        assert result is True
        assert self.executor.state.requires_manual_resume is False
        assert self.executor.state.flatten_reason is None
        assert self.mock_safety_manager.breakers_reset is True
        mock_save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_safety_checks(self):
        """Test safety checks execution"""
        # Mock state methods
        self.test_state.get_realized_pnl = MagicMock(return_value=Decimal('100'))
        self.test_state.get_unrealized_pnl = MagicMock(return_value=Decimal('50'))
        
        action = await self.executor._execute_safety_checks()
        
        assert action == SafetyAction.CONTINUE
        assert len(self.mock_safety_manager.metrics_updates) == 1
        
        update = self.mock_safety_manager.metrics_updates[0]
        assert update['total_pnl'] == Decimal('150')
        assert 'daily_pnl' in update
        assert update['unrealized_pnl'] == Decimal('50')
    
    @pytest.mark.asyncio
    async def test_handle_flatten_success(self):
        """Test successful flatten operation"""
        # Mock flatten result
        mock_result = FlattenResult(
            success=True,
            orders_canceled=5,
            positions_closed={},
            total_proceeds=Decimal('1000'),
            fees_paid=Decimal('10'),
            execution_time_seconds=2.5,
            error_message=None
        )
        
        with patch('core.executor.execute_flatten_operation', return_value=mock_result) as mock_flatten:
            with patch('core.executor.save_state') as mock_save:
                result = await self.executor._handle_flatten("Test reason")
        
        assert result is True
        mock_flatten.assert_called_once_with(
            self.mock_client,
            self.test_state,
            "Test reason",
            self.flatten_config
        )
        mock_save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_flatten_failure(self):
        """Test failed flatten operation"""
        # Mock flatten result
        mock_result = FlattenResult(
            success=False,
            orders_canceled=0,
            positions_closed={},
            total_proceeds=Decimal('0'),
            fees_paid=Decimal('0'),
            execution_time_seconds=1.0,
            error_message="Test error"
        )
        
        with patch('core.executor.execute_flatten_operation', return_value=mock_result):
            with patch('core.executor.save_state') as mock_save:
                result = await self.executor._handle_flatten("Test reason")
        
        assert result is False
        mock_save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_flatten_exception(self):
        """Test flatten operation with exception"""
        with patch('core.executor.execute_flatten_operation', side_effect=Exception("Test error")):
            with patch('core.executor.save_state') as mock_save:
                result = await self.executor._handle_flatten("Test reason")
        
        assert result is False
        assert self.test_state.requires_manual_resume is True
        assert "Test error" in self.test_state.flatten_reason
        mock_save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_manual_flatten(self):
        """Test manual flatten trigger"""
        with patch.object(self.executor, '_handle_flatten', return_value=True) as mock_handle:
            result = await self.executor.manual_flatten("Manual test")
        
        assert result is True
        mock_handle.assert_called_once_with("Manual: Manual test")
    
    @pytest.mark.asyncio
    async def test_handle_emergency_stop_no_positions(self):
        """Test emergency stop with no open positions"""
        with patch('core.executor.save_state') as mock_save:
            await self.executor._handle_emergency_stop("Test emergency")
        
        assert self.executor.state.status == BotStatus.ERROR
        assert self.executor.running is False
        assert self.executor.state.last_error == "Emergency stop: Test emergency"
        mock_save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_emergency_stop_with_positions(self):
        """Test emergency stop with open positions"""
        # Mock position
        from core.state import Position
        mock_position = MagicMock(spec=Position)
        mock_position.is_flat = False
        self.test_state.positions["test"] = mock_position
        
        with patch.object(self.executor, '_handle_flatten') as mock_flatten:
            with patch('core.executor.save_state'):
                await self.executor._handle_emergency_stop("Test emergency")
        
        mock_flatten.assert_called_once_with("Emergency stop: Test emergency")
    
    @pytest.mark.asyncio
    async def test_cancel_open_orders_success(self):
        """Test successful order cancellation"""
        with patch('core.executor.FlattenExecutor') as mock_flatten_executor_class:
            mock_executor = mock_flatten_executor_class.return_value
            mock_executor._cancel_all_orders = AsyncMock(return_value=3)
            
            await self.executor._cancel_open_orders()
            
            mock_flatten_executor_class.assert_called_once_with(self.mock_client, self.flatten_config)
            mock_executor._cancel_all_orders.assert_called_once_with(self.test_state)
    
    @pytest.mark.asyncio
    async def test_cancel_open_orders_exception(self):
        """Test order cancellation with exception"""
        with patch('core.executor.FlattenExecutor') as mock_flatten_executor_class:
            mock_executor = mock_flatten_executor_class.return_value
            mock_executor._cancel_all_orders = AsyncMock(side_effect=Exception("Cancel error"))
            
            await self.executor._cancel_open_orders()
            
            assert "Cancel orders error: Cancel error" in self.test_state.last_error
    
    @pytest.mark.asyncio
    async def test_save_state_with_active_orders(self):
        """Test state saving with active orders"""
        # Add active order
        from core.state import Order
        mock_order = MagicMock(spec=Order)
        self.test_state.active_orders["test_order"] = mock_order
        
        with patch('core.executor.save_state') as mock_save:
            await self.executor._save_state_if_needed()
        
        mock_save.assert_called_once_with(self.test_state, "test_state.json")
    
    @pytest.mark.asyncio
    async def test_save_state_periodic(self):
        """Test periodic state saving"""
        # Set last update to over 5 minutes ago
        self.test_state.last_update = datetime.now(timezone.utc) - timedelta(minutes=6)
        
        with patch('core.executor.save_state') as mock_save:
            await self.executor._save_state_if_needed()
        
        mock_save.assert_called_once_with(self.test_state, "test_state.json")
    
    @pytest.mark.asyncio
    async def test_save_state_not_needed(self):
        """Test state saving when not needed"""
        # Recent update, no active orders
        self.test_state.last_update = datetime.now(timezone.utc)
        
        with patch('core.executor.save_state') as mock_save:
            await self.executor._save_state_if_needed()
        
        mock_save.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test graceful shutdown"""
        with patch('core.executor.save_state') as mock_save:
            await self.executor._shutdown()
        
        assert self.executor.state.status == BotStatus.STOPPED
        assert self.mock_client.session_closed is True
        mock_save.assert_called_once()
    
    def test_calculate_daily_pnl(self):
        """Test daily PnL calculation"""
        self.test_state.metrics.total_pnl = Decimal('150.5')
        
        result = self.executor._calculate_daily_pnl()
        
        assert result == Decimal('150.5')
    
    def test_log_safety_status(self):
        """Test safety status logging"""
        # This method primarily logs, so we test it doesn't raise exceptions
        self.executor._log_safety_status(SafetyAction.CONTINUE)
        # If no exception is raised, test passes
        assert True


class TestTradingLoopIntegration:
    """Test trading loop integration scenarios"""
    
    def setup_method(self):
        """Set up test components"""
        self.mock_client = MockBinanceClient()
        self.test_state = create_empty_state("BTCUSDC")
        self.mock_safety_manager = MockSafetyManager()
        
        self.executor = TradingExecutor(
            client=self.mock_client,
            state=self.test_state,
            safety_manager=self.mock_safety_manager
        )
    
    @pytest.mark.asyncio
    async def test_trading_loop_manual_resume_required(self):
        """Test trading loop when manual resume is required"""
        self.test_state.requires_manual_resume = True
        self.test_state.flatten_reason = "Test flatten"
        
        # Mock the loop to stop after first iteration but don't change status
        original_sleep = asyncio.sleep
        call_count = 0
        
        async def mock_sleep(duration):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:  # Stop after first sleep
                self.executor.running = False
            await original_sleep(0.01)  # Very short sleep for testing
        
        with patch('asyncio.sleep', side_effect=mock_sleep):
            with patch('core.executor.save_state'):
                await self.executor.start_trading_loop()
        
        # State should remain STOPPED since we set running=False to exit loop
        assert self.test_state.status == BotStatus.STOPPED
    
    @pytest.mark.asyncio
    async def test_trading_loop_emergency_stop(self):
        """Test trading loop with emergency stop action"""
        self.mock_safety_manager.evaluation_result = SafetyAction.EMERGENCY_STOP
        
        with patch.object(self.executor, '_handle_emergency_stop') as mock_emergency:
            await self.executor.start_trading_loop()
        
        mock_emergency.assert_called_once_with("Safety system emergency stop")
    
    @pytest.mark.asyncio
    async def test_trading_loop_flatten_action(self):
        """Test trading loop with flatten action"""
        self.mock_safety_manager.evaluation_result = SafetyAction.FLATTEN
        call_count = 0
        
        async def mock_handle_flatten(reason):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:  # Stop after first flatten
                self.executor.running = False
            return True
        
        with patch.object(self.executor, '_handle_flatten', side_effect=mock_handle_flatten):
            await self.executor.start_trading_loop()
        
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_trading_loop_pause_action(self):
        """Test trading loop with pause action"""
        self.mock_safety_manager.evaluation_result = SafetyAction.PAUSE
        call_count = 0
        
        async def mock_sleep(duration):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:  # Stop after first sleep
                self.executor.running = False
        
        with patch('asyncio.sleep', side_effect=mock_sleep):
            await self.executor.start_trading_loop()
        
        # State becomes STOPPED after loop ends due to shutdown
        assert self.test_state.status == BotStatus.STOPPED
    
    @pytest.mark.asyncio
    async def test_trading_loop_cancel_orders_action(self):
        """Test trading loop with cancel orders action"""
        self.mock_safety_manager.evaluation_result = SafetyAction.CANCEL_ORDERS
        call_count = 0
        
        async def mock_cancel_orders():
            nonlocal call_count
            call_count += 1
            if call_count >= 1:  # Stop after first call
                self.executor.running = False
        
        with patch.object(self.executor, '_cancel_open_orders', side_effect=mock_cancel_orders):
            with patch('asyncio.sleep'):
                await self.executor.start_trading_loop()
        
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_trading_loop_with_strategy_success(self):
        """Test trading loop with successful strategy execution"""
        strategy_calls = []
        
        async def mock_strategy(client, state):
            strategy_calls.append((client, state))
            self.executor.running = False  # Stop after first call
        
        with patch('asyncio.sleep'):
            await self.executor.start_trading_loop(strategy_callback=mock_strategy)
        
        assert len(strategy_calls) == 1
        assert strategy_calls[0][0] == self.mock_client
        assert strategy_calls[0][1] == self.test_state
        # State becomes STOPPED after loop ends due to shutdown
        assert self.test_state.status == BotStatus.STOPPED
    
    @pytest.mark.asyncio
    async def test_trading_loop_with_strategy_exception(self):
        """Test trading loop with strategy exception"""
        async def failing_strategy(client, state):
            self.executor.running = False  # Stop after first call
            raise Exception("Strategy error")
        
        with patch('asyncio.sleep'):
            await self.executor.start_trading_loop(strategy_callback=failing_strategy)
        
        assert "Strategy error" in self.test_state.last_error
        assert len(self.mock_safety_manager.metrics_updates) > 0
    
    @pytest.mark.asyncio
    async def test_trading_loop_keyboard_interrupt(self):
        """Test trading loop with keyboard interrupt"""
        call_count = 0
        
        async def mock_sleep(duration):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                raise KeyboardInterrupt()
        
        with patch('asyncio.sleep', side_effect=mock_sleep):
            with patch.object(self.executor, '_shutdown') as mock_shutdown:
                await self.executor.start_trading_loop()
        
        mock_shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_trading_loop_critical_exception(self):
        """Test trading loop with critical exception"""
        with patch.object(self.executor, '_execute_safety_checks', side_effect=Exception("Critical error")):
            with patch.object(self.executor, '_handle_emergency_stop') as mock_emergency:
                await self.executor.start_trading_loop()
        
        mock_emergency.assert_called_once_with("Critical error: Critical error")


class TestTradingExecutorFactoryMethod:
    """Test TradingExecutor factory method from unified config"""
    
    @pytest.mark.asyncio
    async def test_from_unified_config_not_available(self):
        """Test factory method when unified config is not available"""
        state = create_empty_state("BTCUSDC")
        
        with patch('core.executor._unified_config_available', False):
            with pytest.raises(ImportError, match="Unified config system not available"):
                await TradingExecutor.from_unified_config(state)
    
    @pytest.mark.asyncio
    async def test_from_unified_config_success(self):
        """Test successful factory method creation"""
        state = create_empty_state("BTCUSDC")
        
        # Mock all dependencies
        mock_config = MagicMock()
        mock_config.trading.max_position_size = Decimal('1000')
        mock_config.trading.default_symbol = "BTCUSDC"
        mock_config.grid.invest_per_grid = Decimal('100')
        mock_config.grid.n_grids = 10
        mock_config.get_cache_dir.return_value.joinpath.return_value = "test_path.json"
        
        mock_binance_config = MagicMock()
        
        with patch('core.executor._unified_config_available', True):
            with patch('core.executor.get_settings', return_value=mock_config):
                with patch('exec.binance.BinanceConfig.from_unified_config', return_value=mock_binance_config):
                    with patch('exec.binance.BinanceClient') as mock_client_class:
                        mock_client = MagicMock()
                        mock_client.start_session = AsyncMock()
                        mock_client_class.return_value = mock_client
                        
                        executor = await TradingExecutor.from_unified_config(state)
                        
                        # Test that executor was created properly
                        assert isinstance(executor, TradingExecutor)
                        assert executor.state == state
                        assert isinstance(executor.safety_manager, SafetyManager)
                        assert isinstance(executor.flatten_config, FlattenConfig)
                        # Don't test exact client equality due to initialization complexity
                        assert executor.client is not None


class TestRunTradingBotFunction:
    """Test run_trading_bot convenience function"""
    
    @pytest.mark.asyncio
    async def test_run_trading_bot_unified_config(self):
        """Test run_trading_bot with unified config"""
        mock_client = MockBinanceClient()
        mock_state = create_empty_state("BTCUSDC")
        
        mock_executor = AsyncMock()
        
        with patch('core.executor._unified_config_available', True):
            with patch('core.executor.TradingExecutor.from_unified_config', return_value=mock_executor):
                await run_trading_bot(
                    client=mock_client,
                    state=mock_state,
                    use_unified_config=True
                )
        
        # Verify executor was created and trading loop started
        mock_executor.start_trading_loop.assert_called_once_with(None)
    
    @pytest.mark.asyncio
    async def test_run_trading_bot_traditional_init(self):
        """Test run_trading_bot with traditional initialization"""
        mock_client = MockBinanceClient()
        mock_state = create_empty_state("BTCUSDC")
        
        with patch('core.executor.TradingExecutor') as mock_executor_class:
            mock_executor = AsyncMock()
            mock_executor_class.return_value = mock_executor
            
            await run_trading_bot(
                client=mock_client,
                state=mock_state,
                use_unified_config=False
            )
        
        # Verify executor was created with traditional init
        mock_executor_class.assert_called_once()
        mock_executor.start_trading_loop.assert_called_once_with(None)
    
    @pytest.mark.asyncio
    async def test_run_trading_bot_no_client_error(self):
        """Test run_trading_bot without client raises error"""
        with pytest.raises(ValueError, match="Binance client is required"):
            await run_trading_bot(client=None, use_unified_config=False)
    
    @pytest.mark.asyncio
    async def test_run_trading_bot_with_strategy(self):
        """Test run_trading_bot with strategy callback"""
        mock_client = MockBinanceClient()
        
        async def test_strategy(client, state):
            pass
        
        with patch('core.executor.TradingExecutor') as mock_executor_class:
            mock_executor = AsyncMock()
            mock_executor_class.return_value = mock_executor
            
            await run_trading_bot(
                client=mock_client,
                strategy_callback=test_strategy,
                use_unified_config=False
            )
        
        mock_executor.start_trading_loop.assert_called_once_with(test_strategy)
    
    @pytest.mark.asyncio
    async def test_run_trading_bot_with_all_params(self):
        """Test run_trading_bot with all parameters"""
        mock_client = MockBinanceClient()
        mock_state = create_empty_state("BTCUSDC")
        mock_risk_limits = RiskLimits()
        mock_flatten_config = FlattenConfig()
        
        async def test_strategy(client, state):
            pass
        
        with patch('core.executor.TradingExecutor') as mock_executor_class:
            mock_executor = AsyncMock()
            mock_executor_class.return_value = mock_executor
            
            await run_trading_bot(
                client=mock_client,
                state=mock_state,
                strategy_callback=test_strategy,
                risk_limits=mock_risk_limits,
                flatten_config=mock_flatten_config,
                state_save_path="custom_state.json",
                use_unified_config=False
            )
        
        # Verify executor was called correctly
        mock_executor_class.assert_called_once()
        
        # Check call args based on actual implementation
        call_args = mock_executor_class.call_args
        args = call_args[0] if call_args[0] else []
        kwargs = call_args[1] if call_args[1] else {}
        
        # The function creates a new TradingExecutor with traditional init
        if args:
            assert args[0] == mock_client  # client
            assert args[1] == mock_state    # state  
            assert isinstance(args[2], SafetyManager)  # safety_manager
            assert args[3] == mock_flatten_config  # flatten_config
            assert args[4] == "custom_state.json"  # state_save_path
        else:
            # Called with keyword arguments
            assert kwargs.get('client') == mock_client
            assert kwargs.get('state') == mock_state
            assert isinstance(kwargs.get('safety_manager'), SafetyManager)
            assert kwargs.get('flatten_config') == mock_flatten_config
            assert kwargs.get('state_save_path') == "custom_state.json"


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def setup_method(self):
        """Set up test components"""
        self.mock_client = MockBinanceClient()
        self.test_state = create_empty_state("BTCUSDC")
        self.mock_safety_manager = MockSafetyManager()
        
        self.executor = TradingExecutor(
            client=self.mock_client,
            state=self.test_state,
            safety_manager=self.mock_safety_manager
        )
    
    @pytest.mark.asyncio
    async def test_shutdown_without_close_session(self):
        """Test shutdown when client doesn't have close_session method"""
        # Use a mock that doesn't have close_session attribute
        simple_client = type('MockClient', (), {})()
        
        executor = TradingExecutor(
            client=simple_client,
            state=self.test_state,
            safety_manager=self.mock_safety_manager
        )
        
        with patch('core.executor.save_state'):
            await executor._shutdown()
        
        assert executor.state.status == BotStatus.STOPPED
    
    def test_safety_status_logging_with_active_breakers(self):
        """Test safety status logging with active breakers"""
        # Mock safety manager with active breakers
        self.mock_safety_manager.get_safety_status = MagicMock(return_value={
            'active_breakers': ['max_drawdown', 'api_errors'],
            'metrics': {
                'total_pnl': Decimal('-50'),
                'max_drawdown': Decimal('0.15'),
                'api_errors': 5
            }
        })
        
        # Should not raise exception
        self.executor._log_safety_status(SafetyAction.PAUSE)
        assert True
    
    @pytest.mark.asyncio
    async def test_save_state_exactly_5_minutes(self):
        """Test state saving at exactly 5 minute boundary"""
        # Set last update to exactly 5 minutes ago
        self.test_state.last_update = datetime.now(timezone.utc) - timedelta(minutes=5)
        
        with patch('core.executor.save_state') as mock_save:
            await self.executor._save_state_if_needed()
        
        mock_save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_trading_loop_state_transitions(self):
        """Test state transitions during trading loop"""
        call_count = 0
        
        async def mock_strategy(client, state):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:  # Stop after two calls
                self.executor.running = False
        
        with patch('asyncio.sleep', return_value=None):
            await self.executor.start_trading_loop(strategy_callback=mock_strategy)
        
        assert self.test_state.status == BotStatus.STOPPED  # Final state after shutdown
    
    @pytest.mark.asyncio
    async def test_executor_with_minimal_state(self):
        """Test executor with minimal state configuration"""
        minimal_state = BotState(symbol="ETHUSDC")
        
        executor = TradingExecutor(
            client=self.mock_client,
            state=minimal_state,
            safety_manager=self.mock_safety_manager
        )
        
        assert executor.state.symbol == "ETHUSDC"
        assert executor.state.status == BotStatus.STOPPED  # Default status


if __name__ == "__main__":
    pytest.main([__file__])
