"""
Comprehensive tests for core/safety.py

This module contains tests for:
- Enums (AlertLevel, SafetyAction)
- Data classes (SafetyEvent, RiskLimits, PerformanceMetrics)
- CircuitBreaker class and functionality
- SafetyManager class and all safety checks
- Safety event handling and callbacks
- Circuit breaker operations and auto-reset
- Emergency scenarios and complex safety evaluations
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from core.safety import (
    AlertLevel, SafetyAction, SafetyEvent, RiskLimits, PerformanceMetrics,
    CircuitBreaker, SafetyManager
)


@pytest.fixture
def risk_limits():
    """Fixture for basic risk limits"""
    return RiskLimits(
        max_position_size=Decimal('1000'),
        max_daily_loss=Decimal('100'),
        max_drawdown=Decimal('0.1'),
        max_order_size=Decimal('50')
    )


@pytest.fixture
def safety_manager(risk_limits):
    """Fixture for SafetyManager instance"""
    return SafetyManager(risk_limits)


class TestEnums:
    """Test enum definitions"""
    
    def test_alert_level_enum(self):
        """Test AlertLevel enum values"""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.CRITICAL.value == "critical"
        assert AlertLevel.EMERGENCY.value == "emergency"
        
    def test_safety_action_enum(self):
        """Test SafetyAction enum values"""
        assert SafetyAction.NONE.value == "none"
        assert SafetyAction.CONTINUE.value == "continue"
        assert SafetyAction.PAUSE.value == "pause"
        assert SafetyAction.FLATTEN.value == "flatten"
        assert SafetyAction.CANCEL_ORDERS.value == "cancel_orders"
        assert SafetyAction.CLOSE_POSITIONS.value == "close_positions"
        assert SafetyAction.EMERGENCY_STOP.value == "emergency_stop"


class TestSafetyEvent:
    """Test SafetyEvent data class"""
    
    def test_safety_event_creation(self):
        """Test basic SafetyEvent creation"""
        timestamp = datetime.now(timezone.utc)
        event = SafetyEvent(
            timestamp=timestamp,
            rule_name="test_rule",
            alert_level=AlertLevel.WARNING,
            action=SafetyAction.PAUSE,
            message="Test message"
        )
        
        assert event.timestamp == timestamp
        assert event.rule_name == "test_rule"
        assert event.alert_level == AlertLevel.WARNING
        assert event.action == SafetyAction.PAUSE
        assert event.message == "Test message"
        assert event.data is None
        
    def test_safety_event_with_data(self):
        """Test SafetyEvent creation with data"""
        timestamp = datetime.now(timezone.utc)
        data = {"key": "value", "number": 123}
        
        event = SafetyEvent(
            timestamp=timestamp,
            rule_name="test_rule",
            alert_level=AlertLevel.CRITICAL,
            action=SafetyAction.FLATTEN,
            message="Test message",
            data=data
        )
        
        assert event.data == data


class TestRiskLimits:
    """Test RiskLimits data class"""
    
    def test_risk_limits_defaults(self):
        """Test RiskLimits default values"""
        limits = RiskLimits()
        
        assert limits.max_position_size == Decimal('1000')
        assert limits.max_daily_loss == Decimal('100')
        assert limits.max_drawdown == Decimal('0.1')
        assert limits.max_order_size == Decimal('100')
        assert limits.max_open_orders == 20
        assert limits.price_deviation_threshold == Decimal('0.05')
        assert limits.consecutive_losses_limit == 5
        assert limits.api_error_threshold == 10
        
    def test_risk_limits_custom_values(self):
        """Test RiskLimits with custom values"""
        limits = RiskLimits(
            max_position_size=Decimal('2000'),
            max_daily_loss=Decimal('200'),
            max_drawdown=Decimal('0.15'),
            max_order_size=Decimal('150'),
            max_open_orders=30,
            price_deviation_threshold=Decimal('0.08'),
            consecutive_losses_limit=8,
            api_error_threshold=15
        )
        
        assert limits.max_position_size == Decimal('2000')
        assert limits.max_daily_loss == Decimal('200')
        assert limits.max_drawdown == Decimal('0.15')
        assert limits.max_order_size == Decimal('150')
        assert limits.max_open_orders == 30
        assert limits.price_deviation_threshold == Decimal('0.08')
        assert limits.consecutive_losses_limit == 8
        assert limits.api_error_threshold == 15


class TestPerformanceMetrics:
    """Test PerformanceMetrics data class"""
    
    def test_performance_metrics_defaults(self):
        """Test PerformanceMetrics default values"""
        metrics = PerformanceMetrics()
        
        assert metrics.total_pnl == Decimal('0')
        assert metrics.daily_pnl == Decimal('0')
        assert metrics.unrealized_pnl == Decimal('0')
        assert metrics.max_drawdown == Decimal('0')
        assert metrics.consecutive_losses == 0
        assert metrics.api_errors == 0
        assert metrics.last_trade_time is None
        
    def test_performance_metrics_custom_values(self):
        """Test PerformanceMetrics with custom values"""
        timestamp = datetime.now(timezone.utc)
        metrics = PerformanceMetrics(
            total_pnl=Decimal('500'),
            daily_pnl=Decimal('-50'),
            unrealized_pnl=Decimal('25'),
            max_drawdown=Decimal('0.08'),
            consecutive_losses=3,
            api_errors=2,
            last_trade_time=timestamp
        )
        
        assert metrics.total_pnl == Decimal('500')
        assert metrics.daily_pnl == Decimal('-50')
        assert metrics.unrealized_pnl == Decimal('25')
        assert metrics.max_drawdown == Decimal('0.08')
        assert metrics.consecutive_losses == 3
        assert metrics.api_errors == 2
        assert metrics.last_trade_time == timestamp


class TestCircuitBreaker:
    """Test CircuitBreaker class"""
    
    def test_circuit_breaker_initialization(self):
        """Test CircuitBreaker initialization"""
        breaker = CircuitBreaker("test_breaker", Decimal('100'), 30)
        
        assert breaker.name == "test_breaker"
        assert breaker.threshold == Decimal('100')
        assert breaker.window == timedelta(minutes=30)
        assert not breaker.triggered
        assert breaker.trigger_time is None
        assert breaker.reset_time is None
        
    def test_circuit_breaker_default_window(self):
        """Test CircuitBreaker with default window"""
        breaker = CircuitBreaker("test", Decimal('50'))
        assert breaker.window == timedelta(minutes=60)
        
    def test_circuit_breaker_not_triggered_below_threshold(self):
        """Test circuit breaker doesn't trigger below threshold"""
        breaker = CircuitBreaker("test", Decimal('100'))
        
        result = breaker.check(Decimal('50'))
        assert not result
        assert not breaker.triggered
        
    def test_circuit_breaker_triggers_at_threshold(self):
        """Test circuit breaker triggers at threshold"""
        breaker = CircuitBreaker("test", Decimal('100'))
        
        with patch('core.safety.datetime') as mock_datetime:
            now = datetime.now(timezone.utc)
            mock_datetime.now.return_value = now
            
            result = breaker.check(Decimal('100'))
            assert result
            assert breaker.triggered
            assert breaker.trigger_time == now
            assert breaker.reset_time == now + timedelta(minutes=60)
            
    def test_circuit_breaker_triggers_above_threshold(self):
        """Test circuit breaker triggers above threshold"""
        breaker = CircuitBreaker("test", Decimal('100'))
        
        result = breaker.check(Decimal('150'))
        assert result
        assert breaker.triggered
        
    def test_circuit_breaker_triggers_with_negative_value(self):
        """Test circuit breaker triggers with negative value (uses absolute value)"""
        breaker = CircuitBreaker("test", Decimal('100'))
        
        result = breaker.check(Decimal('-150'))
        assert result
        assert breaker.triggered
        
    def test_circuit_breaker_stays_triggered(self):
        """Test circuit breaker stays triggered on subsequent checks"""
        breaker = CircuitBreaker("test", Decimal('100'))
        
        # First trigger
        breaker.check(Decimal('150'))
        assert breaker.triggered
        
        # Should still be triggered
        result = breaker.check(Decimal('50'))
        assert result
        assert breaker.triggered
        
    def test_circuit_breaker_auto_reset(self):
        """Test circuit breaker auto-resets after window"""
        breaker = CircuitBreaker("test", Decimal('100'), 1)  # 1 minute window
        
        # Trigger the breaker
        past_time = datetime.now(timezone.utc) - timedelta(minutes=2)
        breaker.trigger(past_time)
        
        # Check should auto-reset and not trigger
        result = breaker.check(Decimal('50'))
        assert not result
        assert not breaker.triggered
        
    def test_circuit_breaker_manual_reset(self):
        """Test manual circuit breaker reset"""
        breaker = CircuitBreaker("test", Decimal('100'))
        
        # Trigger the breaker
        breaker.check(Decimal('150'))
        assert breaker.triggered
        
        # Manual reset
        breaker.force_reset()
        assert not breaker.triggered
        assert breaker.trigger_time is None
        assert breaker.reset_time is None
        
    def test_circuit_breaker_trigger_method(self):
        """Test direct trigger method"""
        breaker = CircuitBreaker("test", Decimal('100'))
        timestamp = datetime.now(timezone.utc)
        
        breaker.trigger(timestamp)
        
        assert breaker.triggered
        assert breaker.trigger_time == timestamp
        assert breaker.reset_time == timestamp + timedelta(minutes=60)
        
    def test_circuit_breaker_reset_method(self):
        """Test direct reset method"""
        breaker = CircuitBreaker("test", Decimal('100'))
        
        # Trigger first
        breaker.trigger(datetime.now(timezone.utc))
        assert breaker.triggered
        
        # Reset
        breaker.reset()
        assert not breaker.triggered
        assert breaker.trigger_time is None
        assert breaker.reset_time is None


class TestSafetyManager:
    """Test SafetyManager class"""
        
    def test_safety_manager_initialization(self, safety_manager, risk_limits):
        """Test SafetyManager initialization"""
        assert safety_manager.risk_limits == risk_limits
        assert isinstance(safety_manager.metrics, PerformanceMetrics)
        assert len(safety_manager.circuit_breakers) == 4
        assert "daily_loss" in safety_manager.circuit_breakers
        assert "drawdown" in safety_manager.circuit_breakers
        assert "price_deviation" in safety_manager.circuit_breakers
        assert "api_errors" in safety_manager.circuit_breakers
        assert len(safety_manager.safety_events) == 0
        
    def test_safety_manager_circuit_breaker_initialization(self, safety_manager, risk_limits):
        """Test circuit breakers are properly initialized"""
        daily_loss_breaker = safety_manager.circuit_breakers["daily_loss"]
        assert daily_loss_breaker.threshold == risk_limits.max_daily_loss
        assert daily_loss_breaker.window == timedelta(minutes=60)
        
        drawdown_breaker = safety_manager.circuit_breakers["drawdown"]
        assert drawdown_breaker.threshold == risk_limits.max_drawdown
        
        price_deviation_breaker = safety_manager.circuit_breakers["price_deviation"]
        assert price_deviation_breaker.threshold == risk_limits.price_deviation_threshold
        assert price_deviation_breaker.window == timedelta(minutes=5)
        
        api_errors_breaker = safety_manager.circuit_breakers["api_errors"]
        assert api_errors_breaker.threshold == risk_limits.api_error_threshold
        assert api_errors_breaker.window == timedelta(minutes=30)
        
    def test_register_callback(self, safety_manager):
        """Test callback registration"""
        callback_mock = Mock()
        
        safety_manager.register_callback(SafetyAction.PAUSE, callback_mock)
        
        assert callback_mock in safety_manager.callbacks[SafetyAction.PAUSE]
        
    def test_update_metrics(self, safety_manager):
        """Test metrics update"""
        safety_manager.update_metrics(
            total_pnl=Decimal('500'),
            daily_pnl=Decimal('-25'),
            consecutive_losses=2
        )
        
        assert safety_manager.metrics.total_pnl == Decimal('500')
        assert safety_manager.metrics.daily_pnl == Decimal('-25')
        assert safety_manager.metrics.consecutive_losses == 2
        
    def test_update_metrics_invalid_attribute(self, safety_manager):
        """Test updating metrics with invalid attribute is ignored"""
        safety_manager.update_metrics(
            total_pnl=Decimal('500'),
            invalid_attr="should_be_ignored"
        )
        
        assert safety_manager.metrics.total_pnl == Decimal('500')
        assert not hasattr(safety_manager.metrics, 'invalid_attr')
        
    def test_check_position_size_within_limit(self, safety_manager):
        """Test position size check within limit"""
        action = safety_manager.check_position_size(Decimal('500'))
        assert action == SafetyAction.CONTINUE
        assert len(safety_manager.safety_events) == 0
        
    def test_check_position_size_exceeds_limit(self, safety_manager):
        """Test position size check exceeding limit"""
        action = safety_manager.check_position_size(Decimal('1500'))
        assert action == SafetyAction.FLATTEN
        assert len(safety_manager.safety_events) == 1
        
        event = safety_manager.safety_events[0]
        assert event.rule_name == "position_size_exceeded"
        assert event.alert_level == AlertLevel.CRITICAL
        assert event.action == SafetyAction.FLATTEN
        assert "1500" in event.message
        assert "1000" in event.message
        
    def test_check_order_size_within_limit(self, safety_manager):
        """Test order size check within limit"""
        action = safety_manager.check_order_size(Decimal('25'))
        assert action == SafetyAction.CONTINUE
        assert len(safety_manager.safety_events) == 0
        
    def test_check_order_size_exceeds_limit(self, safety_manager):
        """Test order size check exceeding limit"""
        action = safety_manager.check_order_size(Decimal('75'))
        assert action == SafetyAction.CANCEL_ORDERS
        assert len(safety_manager.safety_events) == 1
        
        event = safety_manager.safety_events[0]
        assert event.rule_name == "order_size_exceeded"
        assert event.alert_level == AlertLevel.WARNING
        assert event.action == SafetyAction.CANCEL_ORDERS
        
    def test_check_daily_loss_within_limit(self, safety_manager):
        """Test daily loss check within limit"""
        safety_manager.metrics.daily_pnl = Decimal('-50')
        action = safety_manager.check_daily_loss()
        assert action == SafetyAction.CONTINUE
        
    def test_check_daily_loss_exceeds_limit(self, safety_manager):
        """Test daily loss check exceeding limit"""
        safety_manager.metrics.daily_pnl = Decimal('-150')
        action = safety_manager.check_daily_loss()
        assert action == SafetyAction.FLATTEN
        assert len(safety_manager.safety_events) == 1
        
        event = safety_manager.safety_events[0]
        assert event.rule_name == "daily_loss_limit"
        assert event.alert_level == AlertLevel.CRITICAL
        assert event.action == SafetyAction.FLATTEN
        
    def test_check_drawdown_within_limit(self, safety_manager):
        """Test drawdown check within limit"""
        safety_manager.metrics.max_drawdown = Decimal('0.05')
        action = safety_manager.check_drawdown()
        assert action == SafetyAction.CONTINUE
        
    def test_check_drawdown_exceeds_limit(self, safety_manager):
        """Test drawdown check exceeding limit"""
        safety_manager.metrics.max_drawdown = Decimal('0.15')
        action = safety_manager.check_drawdown()
        assert action == SafetyAction.FLATTEN
        assert len(safety_manager.safety_events) == 1
        
        event = safety_manager.safety_events[0]
        assert event.rule_name == "max_drawdown"
        assert event.alert_level == AlertLevel.CRITICAL
        assert event.action == SafetyAction.FLATTEN
        
    def test_check_consecutive_losses_within_limit(self, safety_manager):
        """Test consecutive losses check within limit"""
        safety_manager.metrics.consecutive_losses = 3
        action = safety_manager.check_consecutive_losses()
        assert action == SafetyAction.CONTINUE
        
    def test_check_consecutive_losses_exceeds_limit(self, safety_manager):
        """Test consecutive losses check exceeding limit"""
        safety_manager.metrics.consecutive_losses = 6
        action = safety_manager.check_consecutive_losses()
        assert action == SafetyAction.PAUSE
        assert len(safety_manager.safety_events) == 1
        
        event = safety_manager.safety_events[0]
        assert event.rule_name == "consecutive_losses"
        assert event.alert_level == AlertLevel.WARNING
        assert event.action == SafetyAction.PAUSE
        
    def test_check_api_errors_within_limit(self, safety_manager):
        """Test API errors check within limit"""
        safety_manager.metrics.api_errors = 5
        action = safety_manager.check_api_errors()
        assert action == SafetyAction.CONTINUE
        
    def test_check_api_errors_exceeds_limit(self, safety_manager):
        """Test API errors check exceeding limit"""
        safety_manager.metrics.api_errors = 15
        action = safety_manager.check_api_errors()
        assert action == SafetyAction.EMERGENCY_STOP
        assert len(safety_manager.safety_events) == 1
        
        event = safety_manager.safety_events[0]
        assert event.rule_name == "api_error_threshold"
        assert event.alert_level == AlertLevel.CRITICAL
        assert event.action == SafetyAction.EMERGENCY_STOP
        
    def test_check_price_deviation_within_limit(self, safety_manager):
        """Test price deviation check within limit"""
        action = safety_manager.check_price_deviation(Decimal('0.03'))
        assert action == SafetyAction.CONTINUE
        
    def test_check_price_deviation_exceeds_limit(self, safety_manager):
        """Test price deviation check exceeding limit"""
        action = safety_manager.check_price_deviation(Decimal('0.08'))
        assert action == SafetyAction.CANCEL_ORDERS
        assert len(safety_manager.safety_events) == 1
        
        event = safety_manager.safety_events[0]
        assert event.rule_name == "price_deviation"
        assert event.alert_level == AlertLevel.WARNING
        assert event.action == SafetyAction.CANCEL_ORDERS
        
    def test_check_price_deviation_negative_value(self, safety_manager):
        """Test price deviation check with negative value"""
        action = safety_manager.check_price_deviation(Decimal('-0.08'))
        assert action == SafetyAction.CANCEL_ORDERS
        
    def test_emergency_stop(self, safety_manager):
        """Test manual emergency stop"""
        safety_manager.emergency_stop("Market crash detected")
        
        assert len(safety_manager.safety_events) == 1
        event = safety_manager.safety_events[0]
        assert event.rule_name == "manual_emergency_stop"
        assert event.alert_level == AlertLevel.EMERGENCY
        assert event.action == SafetyAction.EMERGENCY_STOP
        assert "Market crash detected" in event.message
        
    def test_evaluate_safety_no_issues(self, safety_manager):
        """Test safety evaluation with no issues"""
        action = safety_manager.evaluate_safety()
        assert action == SafetyAction.CONTINUE
        
    def test_evaluate_safety_multiple_issues_returns_most_critical(self, safety_manager):
        """Test safety evaluation returns most critical action"""
        # Set up multiple issues
        safety_manager.metrics.consecutive_losses = 6  # Would trigger PAUSE
        safety_manager.metrics.daily_pnl = Decimal('-150')  # Would trigger FLATTEN
        
        action = safety_manager.evaluate_safety()
        assert action == SafetyAction.FLATTEN  # Most critical
        
    def test_evaluate_safety_emergency_stop_highest_priority(self, safety_manager):
        """Test that emergency stop has highest priority"""
        # Set up multiple issues including emergency stop
        safety_manager.metrics.consecutive_losses = 6  # PAUSE
        safety_manager.metrics.daily_pnl = Decimal('-150')  # FLATTEN
        safety_manager.metrics.api_errors = 15  # EMERGENCY_STOP
        
        action = safety_manager.evaluate_safety()
        assert action == SafetyAction.EMERGENCY_STOP  # Highest priority
        
    def test_callback_execution_on_safety_event(self, safety_manager):
        """Test that callbacks are executed on safety events"""
        callback_mock = Mock()
        safety_manager.register_callback(SafetyAction.FLATTEN, callback_mock)
        
        # Trigger an event that causes FLATTEN
        safety_manager.check_position_size(Decimal('1500'))
        
        callback_mock.assert_called_once()
        
    def test_callback_execution_handles_exceptions(self, safety_manager):
        """Test that callback exceptions don't crash the system"""
        def failing_callback(event):
            raise Exception("Callback failed")
            
        safety_manager.register_callback(SafetyAction.FLATTEN, failing_callback)
        
        # Should not raise exception
        safety_manager.check_position_size(Decimal('1500'))
        
    def test_get_safety_status(self, safety_manager):
        """Test safety status report"""
        # Create some events and triggered breakers
        safety_manager.metrics.api_errors = 15  # Trigger breaker
        safety_manager.check_api_errors()
        
        status = safety_manager.get_safety_status()
        
        assert "active_breakers" in status
        assert "recent_events" in status
        assert "metrics" in status
        
        assert "api_errors" in status["active_breakers"]
        assert len(status["recent_events"]) == 1
        assert status["metrics"]["api_errors"] == 15
        
    def test_reset_all_breakers(self, safety_manager):
        """Test resetting all circuit breakers"""
        # Trigger multiple breakers
        safety_manager.metrics.api_errors = 15
        safety_manager.metrics.daily_pnl = Decimal('-150')
        safety_manager.check_api_errors()
        safety_manager.check_daily_loss()
        
        # Verify breakers are triggered
        assert safety_manager.circuit_breakers["api_errors"].triggered
        assert safety_manager.circuit_breakers["daily_loss"].triggered
        
        # Reset all
        safety_manager.reset_all_breakers()
        
        # Verify all are reset
        for breaker in safety_manager.circuit_breakers.values():
            assert not breaker.triggered
            
    def test_is_trading_allowed_when_safe(self, safety_manager):
        """Test trading allowed when conditions are safe"""
        allowed, reason, action = safety_manager.is_trading_allowed()
        
        assert allowed
        assert reason == "Trading allowed"
        assert action == SafetyAction.CONTINUE
        
    def test_is_trading_allowed_with_emergency_stop(self, safety_manager):
        """Test trading not allowed with emergency stop"""
        safety_manager.metrics.api_errors = 15
        
        allowed, reason, action = safety_manager.is_trading_allowed()
        
        assert not allowed
        assert "emergency_stop" in reason
        assert action == SafetyAction.EMERGENCY_STOP
        
    def test_is_trading_allowed_with_flatten(self, safety_manager):
        """Test trading not allowed with flatten requirement"""
        safety_manager.metrics.daily_pnl = Decimal('-150')
        
        allowed, reason, action = safety_manager.is_trading_allowed()
        
        assert not allowed
        assert "flatten" in reason
        assert action == SafetyAction.FLATTEN
        
    def test_is_trading_allowed_with_pause(self, safety_manager):
        """Test trading not allowed with pause requirement"""
        safety_manager.metrics.consecutive_losses = 6
        
        allowed, reason, action = safety_manager.is_trading_allowed()
        
        assert not allowed
        assert "pause" in reason
        assert action == SafetyAction.PAUSE
        
    def test_is_trading_allowed_with_cancel_orders(self, safety_manager):
        """Test trading allowed but should cancel orders"""
        # Set metrics that would trigger price deviation check
        # We need to check price deviation directly since it's not checked in evaluate_safety
        action = safety_manager.check_price_deviation(Decimal('0.08'))
        assert action == SafetyAction.CANCEL_ORDERS
        
        # For this test, let's manually test the trading allowed logic
        allowed, reason, action = safety_manager.is_trading_allowed()
        
        # Since evaluate_safety doesn't check price deviation, this will be CONTINUE
        assert allowed
        assert action == SafetyAction.CONTINUE


class TestComplexScenarios:
    """Test complex safety scenarios and edge cases"""
    
    @pytest.fixture
    def strict_limits(self):
        """Fixture for strict risk limits"""
        return RiskLimits(
            max_position_size=Decimal('500'),
            max_daily_loss=Decimal('50'),
            max_drawdown=Decimal('0.05'),
            max_order_size=Decimal('25'),
            consecutive_losses_limit=3,
            api_error_threshold=5
        )
        
    @pytest.fixture
    def strict_safety_manager(self, strict_limits):
        """Fixture for SafetyManager with strict limits"""
        return SafetyManager(strict_limits)
        
    def test_cascading_safety_events(self, strict_safety_manager):
        """Test multiple safety events triggered simultaneously"""
        # Set up conditions that trigger multiple safety rules
        strict_safety_manager.metrics.daily_pnl = Decimal('-75')  # Exceeds daily loss
        strict_safety_manager.metrics.max_drawdown = Decimal('0.08')  # Exceeds drawdown
        strict_safety_manager.metrics.consecutive_losses = 4  # Exceeds consecutive losses
        
        # Trigger all checks
        action1 = strict_safety_manager.check_daily_loss()
        action2 = strict_safety_manager.check_drawdown()
        action3 = strict_safety_manager.check_consecutive_losses()
        
        assert action1 == SafetyAction.FLATTEN
        assert action2 == SafetyAction.FLATTEN
        assert action3 == SafetyAction.PAUSE
        assert len(strict_safety_manager.safety_events) == 3
        
    def test_circuit_breaker_recovery_scenario(self):
        """Test circuit breaker recovery after cool-down period"""
        breaker = CircuitBreaker("test", Decimal('100'), 1)  # 1 minute window
        
        # First trigger
        first_time = datetime.now(timezone.utc)
        with patch('core.safety.datetime') as mock_datetime:
            mock_datetime.now.return_value = first_time
            result = breaker.check(Decimal('150'))
            assert result
            assert breaker.triggered
            
        # Still triggered within window
        within_window = first_time + timedelta(seconds=30)
        with patch('core.safety.datetime') as mock_datetime:
            mock_datetime.now.return_value = within_window
            result = breaker.check(Decimal('50'))
            assert result
            assert breaker.triggered
            
        # Should reset after window
        after_window = first_time + timedelta(minutes=2)
        with patch('core.safety.datetime') as mock_datetime:
            mock_datetime.now.return_value = after_window
            result = breaker.check(Decimal('50'))
            assert not result
            assert not breaker.triggered
            
    def test_rapid_fire_safety_events(self, safety_manager):
        """Test handling of rapid consecutive safety events"""
        # Trigger multiple events rapidly
        for i in range(5):
            safety_manager.check_position_size(Decimal('1500'))
            
        # Should have 5 events
        assert len(safety_manager.safety_events) == 5
        
        # All should be the same type
        for event in safety_manager.safety_events:
            assert event.rule_name == "position_size_exceeded"
            assert event.action == SafetyAction.FLATTEN
            
    def test_safety_status_with_multiple_active_breakers(self, safety_manager):
        """Test safety status with multiple active breakers"""
        # Trigger multiple breakers
        safety_manager.metrics.api_errors = 15
        safety_manager.metrics.daily_pnl = Decimal('-150')
        
        safety_manager.check_api_errors()
        safety_manager.check_daily_loss()
        
        status = safety_manager.get_safety_status()
        
        assert len(status["active_breakers"]) == 2
        assert "api_errors" in status["active_breakers"]
        assert "daily_loss" in status["active_breakers"]
        assert len(status["recent_events"]) == 2
        
    def test_safety_event_data_preservation(self, safety_manager):
        """Test that safety event data is properly preserved"""
        custom_data = {
            "position_symbol": "BTCUSDT",
            "position_value": 1500,
            "limit_exceeded_by": 500
        }
        
        # Manually trigger event with data
        safety_manager._trigger_safety_event(
            "custom_rule",
            AlertLevel.WARNING,
            SafetyAction.PAUSE,
            "Custom safety event",
            custom_data
        )
        
        event = safety_manager.safety_events[0]
        assert event.data == custom_data
        assert event.data["position_symbol"] == "BTCUSDT"
        
    def test_multiple_callbacks_for_same_action(self, safety_manager):
        """Test multiple callbacks registered for same action"""
        callback1 = Mock()
        callback2 = Mock()
        callback3 = Mock()
        
        safety_manager.register_callback(SafetyAction.FLATTEN, callback1)
        safety_manager.register_callback(SafetyAction.FLATTEN, callback2)
        safety_manager.register_callback(SafetyAction.FLATTEN, callback3)
        
        # Trigger flatten action
        safety_manager.check_position_size(Decimal('1500'))
        
        # All callbacks should be called
        callback1.assert_called_once()
        callback2.assert_called_once()
        callback3.assert_called_once()
        
    def test_log_level_mapping(self, safety_manager):
        """Test alert level to log level mapping"""
        import logging
        
        assert safety_manager._get_log_level(AlertLevel.INFO) == logging.INFO
        assert safety_manager._get_log_level(AlertLevel.WARNING) == logging.WARNING
        assert safety_manager._get_log_level(AlertLevel.CRITICAL) == logging.CRITICAL
        assert safety_manager._get_log_level(AlertLevel.EMERGENCY) == logging.CRITICAL
        
    def test_safety_manager_with_zero_thresholds(self):
        """Test SafetyManager behavior with zero thresholds"""
        limits = RiskLimits(
            max_position_size=Decimal('0'),
            max_daily_loss=Decimal('0'),
            max_drawdown=Decimal('0')
        )
        manager = SafetyManager(limits)
        
        # Any position should trigger
        action = manager.check_position_size(Decimal('1'))
        assert action == SafetyAction.FLATTEN
        
        # Any loss should trigger
        manager.metrics.daily_pnl = Decimal('-1')
        action = manager.check_daily_loss()
        assert action == SafetyAction.FLATTEN
        
    def test_extreme_values_handling(self, safety_manager):
        """Test handling of extreme decimal values"""
        # Very large position
        large_position = Decimal('999999999999.99')
        action = safety_manager.check_position_size(large_position)
        assert action == SafetyAction.FLATTEN
        
        # Very small but over threshold
        small_over = safety_manager.risk_limits.max_order_size + Decimal('0.01')
        action = safety_manager.check_order_size(small_over)
        assert action == SafetyAction.CANCEL_ORDERS
        
    def test_timestamp_consistency(self, safety_manager):
        """Test that timestamps are consistent and timezone-aware"""
        safety_manager.check_position_size(Decimal('1500'))
        
        event = safety_manager.safety_events[0]
        assert event.timestamp.tzinfo is not None
        assert event.timestamp.tzinfo == timezone.utc
        
        # Should be very recent
        now = datetime.now(timezone.utc)
        time_diff = now - event.timestamp
        assert time_diff.total_seconds() < 1  # Within 1 second
