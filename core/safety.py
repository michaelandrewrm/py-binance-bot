"""
Safety - Circuit breakers, risk checks, panic close

This module implements safety mechanisms and risk management controls
to protect against adverse market conditions and system failures.
"""

from typing import Dict, List, Optional, Callable, Tuple
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class SafetyAction(Enum):
    NONE = "none"
    CONTINUE = "continue"
    PAUSE = "pause"
    FLATTEN = "flatten"
    CANCEL_ORDERS = "cancel_orders"
    CLOSE_POSITIONS = "close_positions"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class SafetyEvent:
    """Represents a safety event that was triggered"""

    timestamp: datetime
    rule_name: str
    alert_level: AlertLevel
    action: SafetyAction
    message: str
    data: Dict = None


@dataclass
class RiskLimits:
    """Risk management limits and thresholds"""

    max_position_size: Decimal = Decimal("1000")  # Maximum position value
    max_daily_loss: Decimal = Decimal("100")  # Maximum daily loss
    max_drawdown: Decimal = Decimal("0.1")  # 10% maximum drawdown
    max_order_size: Decimal = Decimal("100")  # Maximum single order size
    max_open_orders: int = 20  # Maximum open orders
    price_deviation_threshold: Decimal = Decimal("0.05")  # 5% price movement
    consecutive_losses_limit: int = 5  # Stop after N consecutive losses
    api_error_threshold: int = 10  # API errors before pause


@dataclass
class PerformanceMetrics:
    """Current performance metrics for risk assessment"""

    total_pnl: Decimal = Decimal("0")
    daily_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    consecutive_losses: int = 0
    api_errors: int = 0
    last_trade_time: Optional[datetime] = None


class CircuitBreaker:
    """Individual circuit breaker with configurable thresholds.

    A circuit breaker monitors a specific metric and triggers when a threshold
    is exceeded, then automatically resets after a specified time window.

    Attributes:
        name: Human-readable name for the circuit breaker
        threshold: Value that triggers the circuit breaker
        window: Time window after which the breaker resets
        triggered: Whether the breaker is currently triggered
        trigger_time: When the breaker was last triggered
        reset_time: When the breaker will reset
    """

    def __init__(self, name: str, threshold: Decimal, window_minutes: int = 60) -> None:
        """Initialize circuit breaker.

        Args:
            name: Human-readable name for the circuit breaker
            threshold: Value that triggers the circuit breaker
            window_minutes: Minutes after which the breaker resets
        """
        self.name = name
        self.threshold = threshold
        self.window = timedelta(minutes=window_minutes)
        self.triggered = False
        self.trigger_time: Optional[datetime] = None
        self.reset_time: Optional[datetime] = None

    def check(self, current_value: Decimal) -> bool:
        """Check if circuit breaker should trigger.

        Args:
            current_value: Current value to check against threshold

        Returns:
            bool: True if circuit breaker is triggered, False otherwise
        """
        now = datetime.now(timezone.utc)

        # Reset if enough time has passed
        if self.triggered and self.reset_time and now >= self.reset_time:
            self.reset()

        # Check threshold
        if not self.triggered and abs(current_value) >= self.threshold:
            self.trigger(now)
            return True

        return self.triggered

    def trigger(self, timestamp: datetime) -> None:
        """Trigger the circuit breaker.

        Args:
            timestamp: When the circuit breaker was triggered
        """
        self.triggered = True
        self.trigger_time = timestamp
        self.reset_time = timestamp + self.window
        logger.warning(f"Circuit breaker '{self.name}' triggered at {timestamp}")

    def reset(self) -> None:
        """Reset the circuit breaker to non-triggered state."""
        self.triggered = False
        self.trigger_time = None
        self.reset_time = None
        logger.info(f"Circuit breaker '{self.name}' reset")

    def force_reset(self):
        """Manually reset the circuit breaker"""
        self.reset()


class SafetyManager:
    """
    Central safety manager that monitors trading activity and triggers
    safety actions when risk thresholds are exceeded.
    """

    def __init__(self, risk_limits: RiskLimits):
        self.risk_limits = risk_limits
        self.metrics = PerformanceMetrics()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.safety_events: List[SafetyEvent] = []
        self.callbacks: Dict[SafetyAction, List[Callable]] = {
            action: [] for action in SafetyAction
        }

        self._initialize_circuit_breakers()

    def _initialize_circuit_breakers(self):
        """Initialize standard circuit breakers"""
        self.circuit_breakers = {
            "daily_loss": CircuitBreaker("daily_loss", self.risk_limits.max_daily_loss),
            "drawdown": CircuitBreaker("drawdown", self.risk_limits.max_drawdown),
            "price_deviation": CircuitBreaker(
                "price_deviation", self.risk_limits.price_deviation_threshold, 5
            ),
            "api_errors": CircuitBreaker(
                "api_errors", self.risk_limits.api_error_threshold, 30
            ),
        }

    def register_callback(self, action: SafetyAction, callback: Callable):
        """Register a callback for a specific safety action"""
        self.callbacks[action].append(callback)

    def update_metrics(self, **kwargs):
        """Update performance metrics"""
        for key, value in kwargs.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)

    def check_position_size(self, position_value: Decimal) -> SafetyAction:
        """Check if position size is within limits"""
        if position_value > self.risk_limits.max_position_size:
            self._trigger_safety_event(
                "position_size_exceeded",
                AlertLevel.CRITICAL,
                SafetyAction.FLATTEN,
                f"Position size {position_value} exceeds limit {self.risk_limits.max_position_size}",
            )
            return SafetyAction.FLATTEN
        return SafetyAction.CONTINUE

    def check_order_size(self, order_value: Decimal) -> SafetyAction:
        """Check if order size is within limits"""
        if order_value > self.risk_limits.max_order_size:
            self._trigger_safety_event(
                "order_size_exceeded",
                AlertLevel.WARNING,
                SafetyAction.CANCEL_ORDERS,
                f"Order size {order_value} exceeds limit {self.risk_limits.max_order_size}",
            )
            return SafetyAction.CANCEL_ORDERS
        return SafetyAction.CONTINUE

    def check_daily_loss(self) -> SafetyAction:
        """Check daily loss circuit breaker"""
        if self.circuit_breakers["daily_loss"].check(abs(self.metrics.daily_pnl)):
            self._trigger_safety_event(
                "daily_loss_limit",
                AlertLevel.CRITICAL,
                SafetyAction.FLATTEN,
                f"Daily loss {self.metrics.daily_pnl} exceeds limit {self.risk_limits.max_daily_loss}",
            )
            return SafetyAction.FLATTEN
        return SafetyAction.CONTINUE

    def check_drawdown(self) -> SafetyAction:
        """Check maximum drawdown circuit breaker"""
        if self.circuit_breakers["drawdown"].check(self.metrics.max_drawdown):
            self._trigger_safety_event(
                "max_drawdown",
                AlertLevel.CRITICAL,
                SafetyAction.FLATTEN,
                f"Drawdown {self.metrics.max_drawdown} exceeds limit {self.risk_limits.max_drawdown}",
            )
            return SafetyAction.FLATTEN
        return SafetyAction.CONTINUE

    def check_consecutive_losses(self) -> SafetyAction:
        """Check consecutive losses limit"""
        if self.metrics.consecutive_losses >= self.risk_limits.consecutive_losses_limit:
            self._trigger_safety_event(
                "consecutive_losses",
                AlertLevel.WARNING,
                SafetyAction.PAUSE,
                f"Consecutive losses {self.metrics.consecutive_losses} exceeds limit",
            )
            return SafetyAction.PAUSE
        return SafetyAction.CONTINUE

    def check_api_errors(self) -> SafetyAction:
        """Check API error rate"""
        if self.circuit_breakers["api_errors"].check(self.metrics.api_errors):
            self._trigger_safety_event(
                "api_error_threshold",
                AlertLevel.CRITICAL,
                SafetyAction.EMERGENCY_STOP,
                f"API errors {self.metrics.api_errors} exceeds threshold",
            )
            return SafetyAction.EMERGENCY_STOP
        return SafetyAction.CONTINUE

    def check_price_deviation(self, price_change_pct: Decimal) -> SafetyAction:
        """Check for extreme price movements"""
        if self.circuit_breakers["price_deviation"].check(abs(price_change_pct)):
            self._trigger_safety_event(
                "price_deviation",
                AlertLevel.WARNING,
                SafetyAction.CANCEL_ORDERS,
                f"Price deviation {price_change_pct} exceeds threshold",
            )
            return SafetyAction.CANCEL_ORDERS
        return SafetyAction.CONTINUE

    def emergency_stop(self, reason: str):
        """Trigger emergency stop"""
        self._trigger_safety_event(
            "manual_emergency_stop",
            AlertLevel.EMERGENCY,
            SafetyAction.EMERGENCY_STOP,
            f"Manual emergency stop: {reason}",
        )

    def evaluate_safety(self) -> SafetyAction:
        """
        Evaluate all safety conditions and return the most critical action required

        Returns:
            SafetyAction indicating what action should be taken
        """
        actions = []

        # Check all safety conditions
        actions.append(self.check_daily_loss())
        actions.append(self.check_drawdown())
        actions.append(self.check_consecutive_losses())
        actions.append(self.check_api_errors())

        # Return the most critical action (highest priority)
        action_priority = {
            SafetyAction.EMERGENCY_STOP: 5,
            SafetyAction.FLATTEN: 4,
            SafetyAction.PAUSE: 3,
            SafetyAction.CANCEL_ORDERS: 2,
            SafetyAction.CONTINUE: 1,
            SafetyAction.NONE: 0,
        }

        # Get the highest priority action
        critical_action = max(actions, key=lambda x: action_priority.get(x, 0))
        return critical_action

    def _trigger_safety_event(
        self,
        rule_name: str,
        alert_level: AlertLevel,
        action: SafetyAction,
        message: str,
        data: Dict = None,
    ):
        """Trigger a safety event and execute associated actions"""
        event = SafetyEvent(
            timestamp=datetime.now(timezone.utc),
            rule_name=rule_name,
            alert_level=alert_level,
            action=action,
            message=message,
            data=data or {},
        )

        self.safety_events.append(event)
        logger.log(
            self._get_log_level(alert_level), f"Safety event: {rule_name} - {message}"
        )

        # Execute callbacks
        for callback in self.callbacks[action]:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error executing safety callback: {e}")

    def _get_log_level(self, alert_level: AlertLevel) -> int:
        """Convert alert level to logging level"""
        mapping = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.CRITICAL: logging.CRITICAL,
            AlertLevel.EMERGENCY: logging.CRITICAL,
        }
        return mapping.get(alert_level, logging.INFO)

    def get_safety_status(self) -> Dict:
        """Get current safety status"""
        return {
            "active_breakers": [
                name
                for name, breaker in self.circuit_breakers.items()
                if breaker.triggered
            ],
            "recent_events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "rule": event.rule_name,
                    "level": event.alert_level.value,
                    "action": event.action.value,
                    "message": event.message,
                }
                for event in self.safety_events[-10:]  # Last 10 events
            ],
            "metrics": {
                "total_pnl": float(self.metrics.total_pnl),
                "daily_pnl": float(self.metrics.daily_pnl),
                "max_drawdown": float(self.metrics.max_drawdown),
                "consecutive_losses": self.metrics.consecutive_losses,
                "api_errors": self.metrics.api_errors,
            },
        }

    def reset_all_breakers(self):
        """Reset all circuit breakers (use with caution)"""
        for breaker in self.circuit_breakers.values():
            breaker.force_reset()
        logger.warning("All circuit breakers have been manually reset")

    def is_trading_allowed(self) -> Tuple[bool, str, SafetyAction]:
        """
        Check if trading is currently allowed

        Returns:
            Tuple of (allowed, reason_if_not, suggested_action)
        """
        # Evaluate current safety conditions
        action = self.evaluate_safety()

        if action in [SafetyAction.EMERGENCY_STOP, SafetyAction.FLATTEN]:
            return False, f"Critical safety condition requires {action.value}", action
        elif action == SafetyAction.PAUSE:
            return False, f"Safety condition requires pause", action
        elif action == SafetyAction.CANCEL_ORDERS:
            return True, f"Trading allowed but should cancel orders", action
        else:
            return True, "Trading allowed", SafetyAction.CONTINUE
