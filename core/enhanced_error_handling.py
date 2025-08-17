"""
Enhanced Error Handling & Recovery - Phase 3 Advanced Features

This module provides comprehensive error handling, automatic recovery,
and resilience features for the trading bot including:
- Connection failure recovery
- API error handling
- Automatic retry mechanisms
- Graceful degradation
- Error classification and routing
"""

from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import logging
import asyncio
import time
import threading
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity classifications"""

    LOW = "low"  # Minor issues, can continue
    MEDIUM = "medium"  # Moderate issues, may need attention
    HIGH = "high"  # Serious issues, should pause/adjust
    CRITICAL = "critical"  # Critical issues, must stop immediately
    FATAL = "fatal"  # Fatal errors, system shutdown required


class ErrorCategory(Enum):
    """Error category classifications"""

    NETWORK = "network"  # Connection, timeout issues
    API = "api"  # Exchange API errors
    DATA = "data"  # Data validation, parsing errors
    TRADING = "trading"  # Order execution errors
    SYSTEM = "system"  # System resource errors
    CONFIGURATION = "configuration"  # Config/parameter errors
    SAFETY = "safety"  # Safety system errors


class RecoveryAction(Enum):
    """Available recovery actions"""

    RETRY = "retry"  # Retry the operation
    PAUSE = "pause"  # Pause trading temporarily
    STOP = "stop"  # Stop the session
    RESTART = "restart"  # Restart with fresh state
    DEGRADE = "degrade"  # Switch to degraded mode
    EMERGENCY_STOP = "emergency_stop"  # Emergency shutdown
    NOTIFY_ONLY = "notify_only"  # Just log and notify


@dataclass
class ErrorEvent:
    """Error event details"""

    timestamp: datetime
    session_id: Optional[str]
    category: ErrorCategory
    severity: ErrorSeverity
    error_code: str
    title: str
    message: str
    details: Dict[str, Any]
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    recovery_action: Optional[RecoveryAction] = None
    resolved: bool = False


class ConnectionHealthMonitor:
    """Monitor connection health and detect failures"""

    def __init__(self):
        self.connection_status = {
            "binance_api": True,
            "data_feed": True,
            "websocket": True,
        }
        self.last_successful_call = {
            "binance_api": datetime.now(timezone.utc),
            "data_feed": datetime.now(timezone.utc),
            "websocket": datetime.now(timezone.utc),
        }
        self.failure_counts = {"binance_api": 0, "data_feed": 0, "websocket": 0}
        self.monitoring_active = False
        self._lock = threading.Lock()

    def start_monitoring(self):
        """Start connection health monitoring"""
        self.monitoring_active = True
        logger.info("Connection health monitoring started")

    def stop_monitoring(self):
        """Stop connection health monitoring"""
        self.monitoring_active = False
        logger.info("Connection health monitoring stopped")

    def record_success(self, service: str):
        """Record successful operation"""
        with self._lock:
            self.connection_status[service] = True
            self.last_successful_call[service] = datetime.now(timezone.utc)
            self.failure_counts[service] = 0

    def record_failure(self, service: str):
        """Record failed operation"""
        with self._lock:
            self.failure_counts[service] += 1
            if self.failure_counts[service] >= 3:
                self.connection_status[service] = False

    def is_healthy(self, service: str) -> bool:
        """Check if service is healthy"""
        return self.connection_status.get(service, False)

    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        with self._lock:
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "overall_healthy": all(self.connection_status.values()),
                "services": {
                    service: {
                        "healthy": status,
                        "last_success": self.last_successful_call[service].isoformat(),
                        "failure_count": self.failure_counts[service],
                    }
                    for service, status in self.connection_status.items()
                },
            }


class RetryManager:
    """Intelligent retry management with exponential backoff"""

    def __init__(self):
        self.retry_policies = {
            ErrorCategory.NETWORK: {
                "max_retries": 5,
                "base_delay": 1.0,
                "max_delay": 60.0,
            },
            ErrorCategory.API: {
                "max_retries": 3,
                "base_delay": 2.0,
                "max_delay": 30.0,
            },
            ErrorCategory.DATA: {
                "max_retries": 2,
                "base_delay": 0.5,
                "max_delay": 5.0,
            },
            ErrorCategory.TRADING: {
                "max_retries": 2,
                "base_delay": 1.0,
                "max_delay": 10.0,
            },
            ErrorCategory.SYSTEM: {
                "max_retries": 1,
                "base_delay": 5.0,
                "max_delay": 30.0,
            },
            ErrorCategory.CONFIGURATION: {
                "max_retries": 0,
                "base_delay": 0.0,
                "max_delay": 0.0,
            },
            ErrorCategory.SAFETY: {
                "max_retries": 0,
                "base_delay": 0.0,
                "max_delay": 0.0,
            },
        }

        self.active_retries: Dict[str, int] = {}
        self._lock = threading.Lock()

    async def retry_with_backoff(
        self,
        operation: Callable,
        category: ErrorCategory,
        operation_id: str,
        *args,
        **kwargs,
    ) -> Any:
        """Retry operation with exponential backoff"""
        policy = self.retry_policies[category]

        with self._lock:
            retry_count = self.active_retries.get(operation_id, 0)

        if retry_count >= policy["max_retries"]:
            raise Exception(
                f"Max retries ({policy['max_retries']}) exceeded for {operation_id}"
            )

        try:
            result = await operation(*args, **kwargs)
            # Success - reset retry count
            with self._lock:
                self.active_retries[operation_id] = 0
            return result

        except Exception as e:
            with self._lock:
                self.active_retries[operation_id] = retry_count + 1

            if retry_count < policy["max_retries"]:
                # Calculate delay with exponential backoff
                delay = min(
                    policy["base_delay"] * (2**retry_count), policy["max_delay"]
                )

                logger.warning(
                    f"Retry {retry_count + 1}/{policy['max_retries']} for {operation_id} in {delay}s"
                )
                await asyncio.sleep(delay)

                # Recursive retry
                return await self.retry_with_backoff(
                    operation, category, operation_id, *args, **kwargs
                )
            else:
                # Max retries exceeded
                with self._lock:
                    self.active_retries[operation_id] = 0
                raise e


class EnhancedErrorHandler:
    """Enhanced error handling and recovery system"""

    def __init__(self):
        self.error_history: List[ErrorEvent] = []
        self.recovery_handlers: Dict[ErrorCategory, Callable] = {}
        self.connection_monitor = ConnectionHealthMonitor()
        self.retry_manager = RetryManager()

        # Error pattern tracking
        self.error_patterns: Dict[str, List[datetime]] = {}
        self._lock = threading.Lock()

        # Register default recovery handlers
        self._register_default_handlers()

    def start_monitoring(self):
        """Start error monitoring and recovery system"""
        self.connection_monitor.start_monitoring()
        logger.info("Enhanced error handling system started")

    def stop_monitoring(self):
        """Stop error monitoring"""
        self.connection_monitor.stop_monitoring()
        logger.info("Enhanced error handling system stopped")

    async def handle_error(
        self,
        error: Exception,
        session_id: Optional[str] = None,
        category: Optional[ErrorCategory] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RecoveryAction:
        """Handle error with automatic classification and recovery"""

        # Classify error if not provided
        if category is None:
            category = self._classify_error(error)

        # Determine severity
        severity = self._determine_severity(error, category)

        # Create error event
        error_event = ErrorEvent(
            timestamp=datetime.now(timezone.utc),
            session_id=session_id,
            category=category,
            severity=severity,
            error_code=type(error).__name__,
            title=f"{category.value.title()} Error",
            message=str(error),
            details=context or {},
        )

        # Store error
        with self._lock:
            self.error_history.append(error_event)
            if len(self.error_history) > 1000:
                self.error_history = self.error_history[-1000:]

        # Track error patterns
        self._track_error_pattern(error_event)

        # Determine recovery action
        recovery_action = self._determine_recovery_action(error_event)
        error_event.recovery_action = recovery_action

        # Execute recovery
        if recovery_action != RecoveryAction.NOTIFY_ONLY:
            await self._execute_recovery(error_event, recovery_action)
            error_event.recovery_attempted = True

        logger.error(f"Error handled: {error_event.title} - {recovery_action.value}")
        return recovery_action

    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the specified time period"""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        recent_errors = [e for e in self.error_history if e.timestamp >= cutoff]

        # Group by category and severity
        by_category = {}
        by_severity = {}

        for error in recent_errors:
            category = error.category.value
            severity = error.severity.value

            by_category[category] = by_category.get(category, 0) + 1
            by_severity[severity] = by_severity.get(severity, 0) + 1

        return {
            "period_hours": hours,
            "total_errors": len(recent_errors),
            "by_category": by_category,
            "by_severity": by_severity,
            "connection_health": self.connection_monitor.get_health_report(),
            "recent_errors": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "category": e.category.value,
                    "severity": e.severity.value,
                    "title": e.title,
                    "message": e.message,
                    "recovery_action": (
                        e.recovery_action.value if e.recovery_action else None
                    ),
                    "resolved": e.resolved,
                }
                for e in recent_errors[-10:]  # Last 10 errors
            ],
        }

    def _classify_error(self, error: Exception) -> ErrorCategory:
        """Classify error based on type and message"""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        if any(
            term in error_str
            for term in ["connection", "timeout", "network", "unreachable"]
        ):
            return ErrorCategory.NETWORK
        elif any(
            term in error_str
            for term in ["api", "rate limit", "unauthorized", "forbidden"]
        ):
            return ErrorCategory.API
        elif any(
            term in error_str for term in ["json", "parse", "decode", "invalid data"]
        ):
            return ErrorCategory.DATA
        elif any(
            term in error_str for term in ["order", "balance", "position", "trade"]
        ):
            return ErrorCategory.TRADING
        elif any(term in error_str for term in ["memory", "disk", "cpu", "resource"]):
            return ErrorCategory.SYSTEM
        elif any(term in error_str for term in ["config", "parameter", "setting"]):
            return ErrorCategory.CONFIGURATION
        else:
            return ErrorCategory.SYSTEM  # Default

    def _determine_severity(
        self, error: Exception, category: ErrorCategory
    ) -> ErrorSeverity:
        """Determine error severity"""
        error_str = str(error).lower()

        # Critical keywords
        if any(
            term in error_str for term in ["critical", "fatal", "emergency", "shutdown"]
        ):
            return ErrorSeverity.CRITICAL

        # Category-based severity
        if category == ErrorCategory.SAFETY:
            return ErrorSeverity.CRITICAL
        elif category == ErrorCategory.TRADING and "balance" in error_str:
            return ErrorSeverity.HIGH
        elif category == ErrorCategory.NETWORK:
            return ErrorSeverity.MEDIUM
        elif category == ErrorCategory.API and "rate limit" in error_str:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW

    def _determine_recovery_action(self, error_event: ErrorEvent) -> RecoveryAction:
        """Determine appropriate recovery action"""

        # Check for error patterns (too many similar errors)
        if self._is_error_pattern_critical(error_event):
            return RecoveryAction.EMERGENCY_STOP

        # Severity-based decisions
        if error_event.severity == ErrorSeverity.CRITICAL:
            return RecoveryAction.EMERGENCY_STOP
        elif error_event.severity == ErrorSeverity.HIGH:
            return RecoveryAction.STOP
        elif error_event.severity == ErrorSeverity.MEDIUM:
            if error_event.category in [
                ErrorCategory.NETWORK,
                ErrorCategory.API,
            ]:
                return RecoveryAction.RETRY
            else:
                return RecoveryAction.PAUSE
        else:
            return RecoveryAction.NOTIFY_ONLY

    async def _execute_recovery(self, error_event: ErrorEvent, action: RecoveryAction):
        """Execute recovery action"""
        try:
            if action == RecoveryAction.EMERGENCY_STOP:
                logger.critical(f"EMERGENCY STOP triggered by {error_event.title}")
                # TODO: Implement emergency stop logic

            elif action == RecoveryAction.STOP:
                logger.warning(
                    f"Stopping session {error_event.session_id} due to {error_event.title}"
                )
                # TODO: Implement session stop logic

            elif action == RecoveryAction.PAUSE:
                logger.warning(
                    f"Pausing session {error_event.session_id} due to {error_event.title}"
                )
                # TODO: Implement session pause logic

            elif action == RecoveryAction.RETRY:
                logger.info(f"Retry will be attempted for {error_event.title}")
                # Retry is handled by RetryManager

        except Exception as recovery_error:
            logger.error(f"Recovery action failed: {recovery_error}")

    def _track_error_pattern(self, error_event: ErrorEvent):
        """Track error patterns for detection of systemic issues"""
        pattern_key = f"{error_event.category.value}:{error_event.error_code}"

        with self._lock:
            if pattern_key not in self.error_patterns:
                self.error_patterns[pattern_key] = []

            self.error_patterns[pattern_key].append(error_event.timestamp)

            # Keep only last hour of errors for pattern analysis
            cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
            self.error_patterns[pattern_key] = [
                ts for ts in self.error_patterns[pattern_key] if ts >= cutoff
            ]

    def _is_error_pattern_critical(self, error_event: ErrorEvent) -> bool:
        """Check if error pattern indicates critical situation"""
        pattern_key = f"{error_event.category.value}:{error_event.error_code}"

        if pattern_key in self.error_patterns:
            recent_errors = len(self.error_patterns[pattern_key])

            # More than 10 similar errors in last hour = critical pattern
            if recent_errors > 10:
                return True

            # More than 5 errors in last 10 minutes = critical pattern
            ten_min_ago = datetime.now(timezone.utc) - timedelta(minutes=10)
            recent_ten_min = [
                ts for ts in self.error_patterns[pattern_key] if ts >= ten_min_ago
            ]
            if len(recent_ten_min) > 5:
                return True

        return False

    def _register_default_handlers(self):
        """Register default recovery handlers"""
        # This would be expanded with actual recovery implementations
        pass


# Global enhanced error handler instance
enhanced_error_handler = EnhancedErrorHandler()
