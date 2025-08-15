"""
Test Suite for Core Module - Enhanced Error Handling Component

This module provides comprehensive unit tests for the enhanced error handling system
including error classification, recovery mechanisms, connection monitoring, and retry logic.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Callable
import threading
import time

# Import the modules under test
from core.enhanced_error_handling import (
    ErrorSeverity,
    ErrorCategory,
    RecoveryAction,
    ErrorEvent,
    ConnectionHealthMonitor,
    RetryManager,
    EnhancedErrorHandler,
    enhanced_error_handler,
)


class TestErrorSeverity:
    """Test ErrorSeverity enum functionality"""

    def test_error_severity_values(self):
        """Test that ErrorSeverity enum has correct values"""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"
        assert ErrorSeverity.FATAL.value == "fatal"

    def test_error_severity_ordering(self):
        """Test error severity can be compared for ordering"""
        severities = [
            ErrorSeverity.LOW,
            ErrorSeverity.MEDIUM,
            ErrorSeverity.HIGH,
            ErrorSeverity.CRITICAL,
            ErrorSeverity.FATAL,
        ]

        # Test that all values are unique
        assert len(set(severities)) == len(severities)


class TestErrorCategory:
    """Test ErrorCategory enum functionality"""

    def test_error_category_values(self):
        """Test that ErrorCategory enum has correct values"""
        assert ErrorCategory.NETWORK.value == "network"
        assert ErrorCategory.API.value == "api"
        assert ErrorCategory.DATA.value == "data"
        assert ErrorCategory.TRADING.value == "trading"
        assert ErrorCategory.SYSTEM.value == "system"
        assert ErrorCategory.CONFIGURATION.value == "configuration"
        assert ErrorCategory.SAFETY.value == "safety"

    def test_error_category_completeness(self):
        """Test that error categories cover expected error types"""
        expected_categories = [
            "network",
            "api",
            "data",
            "trading",
            "system",
            "configuration",
            "safety",
        ]

        actual_values = [cat.value for cat in ErrorCategory]
        for expected in expected_categories:
            assert expected in actual_values


class TestRecoveryAction:
    """Test RecoveryAction enum functionality"""

    def test_recovery_action_values(self):
        """Test that RecoveryAction enum has correct values"""
        assert RecoveryAction.RETRY.value == "retry"
        assert RecoveryAction.PAUSE.value == "pause"
        assert RecoveryAction.STOP.value == "stop"
        assert RecoveryAction.RESTART.value == "restart"
        assert RecoveryAction.DEGRADE.value == "degrade"
        assert RecoveryAction.EMERGENCY_STOP.value == "emergency_stop"
        assert RecoveryAction.NOTIFY_ONLY.value == "notify_only"


class TestErrorEvent:
    """Test ErrorEvent data class functionality"""

    def test_error_event_creation(self):
        """Test creating ErrorEvent with required fields"""
        timestamp = datetime.now(timezone.utc)
        event = ErrorEvent(
            timestamp=timestamp,
            session_id="test-session",
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.HIGH,
            error_code="NET_001",
            title="Connection Timeout",
            message="Failed to connect to exchange API",
            details={"endpoint": "api.binance.com", "timeout": 30},
        )

        assert event.timestamp == timestamp
        assert event.session_id == "test-session"
        assert event.category == ErrorCategory.NETWORK
        assert event.severity == ErrorSeverity.HIGH
        assert event.error_code == "NET_001"
        assert event.title == "Connection Timeout"
        assert event.message == "Failed to connect to exchange API"
        assert event.details["endpoint"] == "api.binance.com"
        assert event.stack_trace is None
        assert event.recovery_attempted is False
        assert event.recovery_action is None
        assert event.resolved is False

    def test_error_event_with_optional_fields(self):
        """Test creating ErrorEvent with optional fields"""
        event = ErrorEvent(
            timestamp=datetime.now(timezone.utc),
            session_id=None,
            category=ErrorCategory.API,
            severity=ErrorSeverity.MEDIUM,
            error_code="API_002",
            title="Rate Limited",
            message="API rate limit exceeded",
            details={},
            stack_trace="Traceback...",
            recovery_attempted=True,
            recovery_action=RecoveryAction.RETRY,
            resolved=True,
        )

        assert event.session_id is None
        assert event.stack_trace == "Traceback..."
        assert event.recovery_attempted is True
        assert event.recovery_action == RecoveryAction.RETRY
        assert event.resolved is True


class TestConnectionHealthMonitor:
    """Test ConnectionHealthMonitor functionality"""

    def setup_method(self):
        """Setup fresh ConnectionHealthMonitor for each test"""
        self.monitor = ConnectionHealthMonitor()

    def test_connection_health_monitor_initialization(self):
        """Test ConnectionHealthMonitor initializes with correct state"""
        assert self.monitor.connection_status["binance_api"] is True
        assert self.monitor.connection_status["data_feed"] is True
        assert self.monitor.connection_status["websocket"] is True
        assert self.monitor.monitoring_active is False

        # Check that timestamps are recent
        now = datetime.now(timezone.utc)
        for service in ["binance_api", "data_feed", "websocket"]:
            last_call = self.monitor.last_successful_call[service]
            assert (now - last_call) < timedelta(seconds=1)
            assert self.monitor.failure_counts[service] == 0

    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring"""
        assert not self.monitor.monitoring_active

        self.monitor.start_monitoring()
        assert self.monitor.monitoring_active

        self.monitor.stop_monitoring()
        assert not self.monitor.monitoring_active

    def test_record_success(self):
        """Test recording successful operations"""
        service = "binance_api"
        original_timestamp = self.monitor.last_successful_call[service]

        # Wait a bit to ensure timestamp difference
        time.sleep(0.01)

        self.monitor.record_success(service)

        # Timestamp should be updated
        assert self.monitor.last_successful_call[service] > original_timestamp
        assert self.monitor.connection_status[service] is True
        assert self.monitor.failure_counts[service] == 0

    def test_record_failure(self):
        """Test recording failed operations"""
        service = "data_feed"

        self.monitor.record_failure(service)

        assert self.monitor.failure_counts[service] == 1
        assert self.monitor.connection_status[service] is False

        # Multiple failures should accumulate
        self.monitor.record_failure(service)
        assert self.monitor.failure_counts[service] == 2

    def test_is_healthy(self):
        """Test health status checking"""
        service = "websocket"

        # Initially healthy
        assert self.monitor.is_healthy(service) is True

        # After failure, unhealthy
        self.monitor.record_failure(service)
        assert self.monitor.is_healthy(service) is False

        # After success, healthy again
        self.monitor.record_success(service)
        assert self.monitor.is_healthy(service) is True

    def test_get_health_report(self):
        """Test getting comprehensive health report"""
        # Set up some test state
        self.monitor.record_failure("binance_api")
        self.monitor.record_failure("binance_api")
        self.monitor.record_success("data_feed")

        report = self.monitor.get_health_report()

        # Verify report structure
        assert "overall_healthy" in report
        assert "services" in report
        assert "failure_counts" in report
        assert "last_successful_calls" in report

        # Verify specific service data
        assert report["services"]["binance_api"] is False  # Failed
        assert report["services"]["data_feed"] is True  # Successful
        assert report["failure_counts"]["binance_api"] == 2
        assert report["failure_counts"]["data_feed"] == 0

        # Overall health should be False due to binance_api failure
        assert report["overall_healthy"] is False

    def test_unknown_service_handling(self):
        """Test handling of unknown services"""
        unknown_service = "unknown_service"

        # Should handle gracefully without crashing
        self.monitor.record_success(unknown_service)
        self.monitor.record_failure(unknown_service)

        # Should return False for unknown services
        assert self.monitor.is_healthy(unknown_service) is False

    def test_thread_safety(self):
        """Test thread safety of connection monitor"""
        service = "binance_api"
        results = []

        def worker():
            for _ in range(10):
                self.monitor.record_success(service)
                self.monitor.record_failure(service)
                results.append(self.monitor.is_healthy(service))

        # Run multiple threads
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Should not crash and should have results
        assert len(results) > 0


class TestRetryManager:
    """Test RetryManager functionality"""

    def setup_method(self):
        """Setup fresh RetryManager for each test"""
        self.retry_manager = RetryManager()

    def test_retry_manager_initialization(self):
        """Test RetryManager initializes with correct configuration"""
        assert self.retry_manager.max_retries[ErrorCategory.NETWORK] == 3
        assert self.retry_manager.max_retries[ErrorCategory.API] == 5
        assert self.retry_manager.max_retries[ErrorCategory.TRADING] == 2

        assert self.retry_manager.base_delay[ErrorCategory.NETWORK] == 1.0
        assert self.retry_manager.base_delay[ErrorCategory.API] == 2.0
        assert self.retry_manager.base_delay[ErrorCategory.TRADING] == 0.5

        assert self.retry_manager.max_delay == 60.0

    @pytest.mark.asyncio
    async def test_retry_with_backoff_success_on_first_try(self):
        """Test retry mechanism when operation succeeds on first try"""
        operation = AsyncMock(return_value="success")

        result = await self.retry_manager.retry_with_backoff(
            operation=operation,
            category=ErrorCategory.NETWORK,
            operation_name="test_operation",
        )

        assert result == "success"
        operation.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_with_backoff_success_after_retries(self):
        """Test retry mechanism when operation succeeds after failures"""
        # Mock operation that fails twice then succeeds
        operation = AsyncMock(
            side_effect=[
                Exception("First failure"),
                Exception("Second failure"),
                "success",
            ]
        )

        with patch("asyncio.sleep") as mock_sleep:
            result = await self.retry_manager.retry_with_backoff(
                operation=operation,
                category=ErrorCategory.NETWORK,
                operation_name="test_operation",
            )

        assert result == "success"
        assert operation.call_count == 3
        assert mock_sleep.call_count == 2  # Two delays before success

    @pytest.mark.asyncio
    async def test_retry_with_backoff_max_retries_exceeded(self):
        """Test retry mechanism when max retries are exceeded"""
        operation = AsyncMock(side_effect=Exception("Persistent failure"))

        with patch("asyncio.sleep"):
            with pytest.raises(Exception, match="Persistent failure"):
                await self.retry_manager.retry_with_backoff(
                    operation=operation,
                    category=ErrorCategory.NETWORK,
                    operation_name="test_operation",
                )

        # Should try initial + max_retries (3) = 4 times for NETWORK
        assert operation.call_count == 4

    @pytest.mark.asyncio
    async def test_retry_with_backoff_exponential_delay(self):
        """Test that retry delays follow exponential backoff"""
        operation = AsyncMock(
            side_effect=[Exception("Failure 1"), Exception("Failure 2"), "success"]
        )

        with patch("asyncio.sleep") as mock_sleep:
            await self.retry_manager.retry_with_backoff(
                operation=operation,
                category=ErrorCategory.API,  # base_delay = 2.0
                operation_name="test_operation",
            )

        # Check exponential backoff: 2.0, 4.0
        calls = mock_sleep.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0] == 2.0  # First retry: base_delay
        assert calls[1][0][0] == 4.0  # Second retry: base_delay * 2

    @pytest.mark.asyncio
    async def test_retry_with_backoff_max_delay_cap(self):
        """Test that retry delays are capped at max_delay"""
        operation = AsyncMock(side_effect=Exception("Always fails"))

        # Use very long delays to test capping
        self.retry_manager.base_delay[ErrorCategory.NETWORK] = 100.0

        with patch("asyncio.sleep") as mock_sleep:
            with pytest.raises(Exception):
                await self.retry_manager.retry_with_backoff(
                    operation=operation,
                    category=ErrorCategory.NETWORK,
                    operation_name="test_operation",
                )

        # All delays should be capped at max_delay (60.0)
        for call in mock_sleep.call_args_list:
            assert call[0][0] <= 60.0

    @pytest.mark.asyncio
    async def test_retry_with_backoff_different_categories(self):
        """Test retry behavior varies by error category"""
        operation_network = AsyncMock(side_effect=Exception("Network error"))
        operation_trading = AsyncMock(side_effect=Exception("Trading error"))

        with patch("asyncio.sleep"):
            # Network errors should retry more times (3)
            with pytest.raises(Exception):
                await self.retry_manager.retry_with_backoff(
                    operation=operation_network,
                    category=ErrorCategory.NETWORK,
                    operation_name="network_op",
                )

            # Trading errors should retry fewer times (2)
            with pytest.raises(Exception):
                await self.retry_manager.retry_with_backoff(
                    operation=operation_trading,
                    category=ErrorCategory.TRADING,
                    operation_name="trading_op",
                )

        assert operation_network.call_count == 4  # initial + 3 retries
        assert operation_trading.call_count == 3  # initial + 2 retries


class TestEnhancedErrorHandler:
    """Test EnhancedErrorHandler functionality"""

    def setup_method(self):
        """Setup fresh EnhancedErrorHandler for each test"""
        self.error_handler = EnhancedErrorHandler()

    def test_enhanced_error_handler_initialization(self):
        """Test EnhancedErrorHandler initializes correctly"""
        assert isinstance(
            self.error_handler.connection_monitor, ConnectionHealthMonitor
        )
        assert isinstance(self.error_handler.retry_manager, RetryManager)
        assert self.error_handler.error_events == []
        assert self.error_handler.monitoring_active is False
        assert self.error_handler.recovery_callbacks == {}

    def test_start_stop_monitoring(self):
        """Test starting and stopping error monitoring"""
        assert not self.error_handler.monitoring_active

        self.error_handler.start_monitoring()
        assert self.error_handler.monitoring_active

        self.error_handler.stop_monitoring()
        assert not self.error_handler.monitoring_active

    @pytest.mark.asyncio
    async def test_handle_error_network_category(self):
        """Test handling network category errors"""
        error = ConnectionError("Network timeout")
        session_id = "test-session"

        with patch.object(
            self.error_handler, "_classify_error"
        ) as mock_classify, patch.object(
            self.error_handler, "_determine_recovery_action"
        ) as mock_recovery, patch.object(
            self.error_handler, "_execute_recovery"
        ) as mock_execute:

            mock_classify.return_value = (ErrorCategory.NETWORK, ErrorSeverity.HIGH)
            mock_recovery.return_value = RecoveryAction.RETRY
            mock_execute.return_value = True

            result = await self.error_handler.handle_error(
                error=error, session_id=session_id, context={"operation": "fetch_price"}
            )

        assert result is not None
        assert len(self.error_handler.error_events) == 1

        event = self.error_handler.error_events[0]
        assert event.session_id == session_id
        assert event.category == ErrorCategory.NETWORK
        assert event.severity == ErrorSeverity.HIGH
        assert event.recovery_attempted is True
        assert event.recovery_action == RecoveryAction.RETRY

    @pytest.mark.asyncio
    async def test_handle_error_without_session(self):
        """Test handling errors without session context"""
        error = ValueError("Invalid parameter")

        with patch.object(self.error_handler, "_classify_error") as mock_classify:
            mock_classify.return_value = (
                ErrorCategory.CONFIGURATION,
                ErrorSeverity.MEDIUM,
            )

            result = await self.error_handler.handle_error(error=error)

        assert result is not None
        event = self.error_handler.error_events[0]
        assert event.session_id is None
        assert event.category == ErrorCategory.CONFIGURATION

    def test_get_error_summary(self):
        """Test getting error summary statistics"""
        # Add some test error events
        now = datetime.now(timezone.utc)

        events = [
            ErrorEvent(
                timestamp=now - timedelta(hours=1),
                session_id="session1",
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.HIGH,
                error_code="NET_001",
                title="Connection Error",
                message="Timeout",
                details={},
            ),
            ErrorEvent(
                timestamp=now - timedelta(hours=2),
                session_id="session1",
                category=ErrorCategory.API,
                severity=ErrorSeverity.MEDIUM,
                error_code="API_001",
                title="Rate Limit",
                message="Too many requests",
                details={},
            ),
            ErrorEvent(
                timestamp=now - timedelta(hours=25),  # Outside 24h window
                session_id="session2",
                category=ErrorCategory.TRADING,
                severity=ErrorSeverity.CRITICAL,
                error_code="TRD_001",
                title="Order Failed",
                message="Insufficient balance",
                details={},
            ),
        ]

        self.error_handler.error_events = events

        summary = self.error_handler.get_error_summary(hours=24)

        # Should only include events from last 24 hours
        assert summary["total_errors"] == 2
        assert summary["by_category"]["NETWORK"] == 1
        assert summary["by_category"]["API"] == 1
        assert "TRADING" not in summary["by_category"]  # Outside time window

        assert summary["by_severity"]["HIGH"] == 1
        assert summary["by_severity"]["MEDIUM"] == 1

    def test_classify_error_network_errors(self):
        """Test error classification for network errors"""
        connection_error = ConnectionError("Connection failed")
        timeout_error = TimeoutError("Request timeout")

        category, severity = self.error_handler._classify_error(connection_error)
        assert category == ErrorCategory.NETWORK
        assert severity == ErrorSeverity.HIGH

        category, severity = self.error_handler._classify_error(timeout_error)
        assert category == ErrorCategory.NETWORK
        assert severity == ErrorSeverity.MEDIUM

    def test_classify_error_api_errors(self):
        """Test error classification for API errors"""
        # Simulate API errors with specific messages
        rate_limit_error = Exception("Rate limit exceeded")
        api_error = Exception("Invalid API key")

        with patch.object(self.error_handler, "_classify_error") as mock_classify:
            mock_classify.side_effect = [
                (ErrorCategory.API, ErrorSeverity.MEDIUM),
                (ErrorCategory.API, ErrorSeverity.CRITICAL),
            ]

            cat1, sev1 = self.error_handler._classify_error(rate_limit_error)
            cat2, sev2 = self.error_handler._classify_error(api_error)

            assert cat1 == ErrorCategory.API
            assert sev1 == ErrorSeverity.MEDIUM
            assert cat2 == ErrorCategory.API
            assert sev2 == ErrorSeverity.CRITICAL

    def test_determine_recovery_action(self):
        """Test recovery action determination"""
        # Test different severity levels
        action = self.error_handler._determine_recovery_action(
            ErrorCategory.NETWORK, ErrorSeverity.LOW
        )
        assert action == RecoveryAction.RETRY

        action = self.error_handler._determine_recovery_action(
            ErrorCategory.TRADING, ErrorSeverity.CRITICAL
        )
        assert action == RecoveryAction.STOP

        action = self.error_handler._determine_recovery_action(
            ErrorCategory.SAFETY, ErrorSeverity.FATAL
        )
        assert action == RecoveryAction.EMERGENCY_STOP


class TestGlobalErrorHandler:
    """Test global error handler instance"""

    def test_global_error_handler_exists(self):
        """Test that global error handler instance exists"""
        assert enhanced_error_handler is not None
        assert isinstance(enhanced_error_handler, EnhancedErrorHandler)

    def test_global_error_handler_singleton(self):
        """Test that global error handler behaves like singleton"""
        # Import again should give same instance
        from core.enhanced_error_handling import enhanced_error_handler as handler2

        assert enhanced_error_handler is handler2


class TestIntegrationScenarios:
    """Test complex integration scenarios"""

    def setup_method(self):
        """Setup components for integration tests"""
        self.connection_monitor = ConnectionHealthMonitor()
        self.retry_manager = RetryManager()
        self.error_handler = EnhancedErrorHandler()

    @pytest.mark.asyncio
    async def test_complete_error_recovery_workflow(self):
        """Test complete error handling and recovery workflow"""
        # Simulate a network operation that fails then succeeds
        call_count = 0

        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network unavailable")
            return "success"

        # Handle the error with full workflow
        with patch("asyncio.sleep"):  # Speed up test
            result = await self.retry_manager.retry_with_backoff(
                operation=flaky_operation,
                category=ErrorCategory.NETWORK,
                operation_name="test_operation",
            )

        assert result == "success"
        assert call_count == 3  # Failed twice, succeeded on third try

    def test_connection_health_monitoring_integration(self):
        """Test integration between connection monitoring and error handling"""
        # Start monitoring
        self.connection_monitor.start_monitoring()

        # Simulate service failures
        services = ["binance_api", "data_feed", "websocket"]
        for service in services:
            self.connection_monitor.record_failure(service)

        # Check overall health
        health_report = self.connection_monitor.get_health_report()
        assert health_report["overall_healthy"] is False

        # Simulate recovery
        for service in services:
            self.connection_monitor.record_success(service)

        health_report = self.connection_monitor.get_health_report()
        assert health_report["overall_healthy"] is True

    @pytest.mark.asyncio
    async def test_error_handler_with_connection_monitoring(self):
        """Test error handler integration with connection monitoring"""
        # Start monitoring
        self.error_handler.start_monitoring()

        # Simulate various errors
        errors = [
            (ConnectionError("Network down"), "network_session"),
            (Exception("API rate limit"), "api_session"),
            (ValueError("Invalid config"), None),
        ]

        for error, session_id in errors:
            await self.error_handler.handle_error(error, session_id)

        # Check that errors were recorded
        assert len(self.error_handler.error_events) == 3

        # Check error summary
        summary = self.error_handler.get_error_summary()
        assert summary["total_errors"] == 3

    def test_concurrent_error_handling(self):
        """Test error handling under concurrent access"""
        results = []

        def worker():
            for i in range(5):
                service = f"service_{i % 3}"
                self.connection_monitor.record_failure(service)
                self.connection_monitor.record_success(service)
                results.append(self.connection_monitor.is_healthy(service))

        # Run concurrent workers
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Should complete without errors
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__])
