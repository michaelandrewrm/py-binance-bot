"""
Advanced Safety Manager - Phase 3 Enhanced Safety Features

This module provides advanced safety and risk management features including:
- Real-time risk monitoring
- Market condition detection
- Automatic emergency responses
- Position size enforcement
- Drawdown protection
"""

from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from enum import Enum
from dataclasses import dataclass
import logging
import threading
import time
import statistics

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classifications"""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"
    CRITICAL = "critical"


class MarketRegime(Enum):
    """Market volatility regime classifications"""

    CALM = "calm"  # Low volatility
    NORMAL = "normal"  # Normal volatility
    VOLATILE = "volatile"  # High volatility
    EXTREME = "extreme"  # Extreme volatility
    CRISIS = "crisis"  # Crisis conditions


class AlertType(Enum):
    """Types of alerts"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class RiskMetrics:
    """Real-time risk metrics"""

    current_drawdown: Decimal
    max_drawdown: Decimal
    position_exposure: Decimal
    leverage_ratio: float
    var_95: Decimal  # Value at Risk 95%
    sharpe_ratio: float
    volatility: float
    correlation_risk: float
    liquidity_risk: float

    def get_risk_level(self) -> RiskLevel:
        """Calculate overall risk level"""
        if self.current_drawdown > Decimal("0.25") or self.volatility > 0.8:
            return RiskLevel.CRITICAL
        elif self.current_drawdown > Decimal("0.15") or self.volatility > 0.6:
            return RiskLevel.EXTREME
        elif self.current_drawdown > Decimal("0.10") or self.volatility > 0.4:
            return RiskLevel.HIGH
        elif self.current_drawdown > Decimal("0.05") or self.volatility > 0.25:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW


@dataclass
class MarketConditions:
    """Current market condition assessment"""

    regime: MarketRegime
    trend_direction: str  # "up", "down", "sideways"
    trend_strength: float  # 0.0 to 1.0
    volatility_percentile: float  # Current vol vs historical
    volume_profile: str  # "low", "normal", "high"
    correlation_breakdown: bool  # High correlation periods

    def is_favorable_for_grid(self) -> bool:
        """Check if conditions favor grid trading"""
        return (
            self.regime in [MarketRegime.CALM, MarketRegime.NORMAL]
            and self.trend_direction == "sideways"
            and not self.correlation_breakdown
        )


@dataclass
class SafetyAlert:
    """Safety alert notification"""

    timestamp: datetime
    alert_type: AlertType
    category: str
    title: str
    message: str
    session_id: Optional[str] = None
    risk_level: Optional[RiskLevel] = None
    requires_action: bool = False
    auto_resolved: bool = False


class AdvancedSafetyManager:
    """Advanced safety and risk management system"""

    def __init__(self):
        self.risk_limits = {
            "max_drawdown": Decimal("0.15"),  # 15% max drawdown
            "max_position_size": Decimal("10000"),  # $10K max position
            "max_daily_loss": Decimal("500"),  # $500 max daily loss
            "min_liquidity_ratio": 0.2,  # 20% minimum liquidity
            "max_correlation": 0.8,  # 80% max correlation
            "volatility_threshold": 0.6,  # 60% volatility threshold
        }

        self.monitoring_active = False
        self.alerts: List[SafetyAlert] = []
        self._lock = threading.Lock()

        # Market data buffers for analysis
        self.price_history: Dict[str, List[Tuple[datetime, Decimal]]] = {}
        self.volume_history: Dict[str, List[Tuple[datetime, Decimal]]] = {}
        self.volatility_history: Dict[str, List[float]] = {}

        # Risk monitoring
        self.session_metrics: Dict[str, RiskMetrics] = {}
        self.market_conditions: Dict[str, MarketConditions] = {}

    def start_monitoring(self, session_id: str, symbol: str):
        """Start advanced monitoring for a trading session"""
        with self._lock:
            self.monitoring_active = True
            self._initialize_session_monitoring(session_id, symbol)

        logger.info(f"Advanced safety monitoring started for {session_id}")
        self._create_alert(
            AlertType.INFO,
            "Safety",
            "Monitoring Started",
            f"Advanced safety monitoring activated for {session_id}",
        )

    def stop_monitoring(self, session_id: str):
        """Stop monitoring for a session"""
        with self._lock:
            if session_id in self.session_metrics:
                del self.session_metrics[session_id]

        logger.info(f"Advanced safety monitoring stopped for {session_id}")
        self._create_alert(
            AlertType.INFO,
            "Safety",
            "Monitoring Stopped",
            f"Advanced safety monitoring deactivated for {session_id}",
        )

    def update_market_data(
        self, symbol: str, price: Decimal, volume: Decimal, timestamp: datetime
    ):
        """Update market data for analysis"""
        with self._lock:
            # Update price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            self.price_history[symbol].append((timestamp, price))

            # Keep last 1000 data points
            if len(self.price_history[symbol]) > 1000:
                self.price_history[symbol] = self.price_history[symbol][-1000:]

            # Update volume history
            if symbol not in self.volume_history:
                self.volume_history[symbol] = []
            self.volume_history[symbol].append((timestamp, volume))

            if len(self.volume_history[symbol]) > 1000:
                self.volume_history[symbol] = self.volume_history[symbol][-1000:]

            # Update market conditions
            self._update_market_conditions(symbol)

    def check_session_safety(
        self,
        session_id: str,
        current_pnl: Decimal,
        position_size: Decimal,
        trades_count: int,
    ) -> Tuple[bool, List[SafetyAlert]]:
        """Check session safety and return alerts"""
        alerts = []
        is_safe = True

        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(
            session_id, current_pnl, position_size
        )

        with self._lock:
            self.session_metrics[session_id] = risk_metrics

        # Check drawdown limits
        if risk_metrics.current_drawdown > self.risk_limits["max_drawdown"]:
            is_safe = False
            alert = self._create_alert(
                AlertType.CRITICAL,
                "Risk",
                "Drawdown Limit Exceeded",
                f"Session {session_id} exceeded max drawdown: {risk_metrics.current_drawdown:.1%}",
                session_id=session_id,
                risk_level=RiskLevel.CRITICAL,
                requires_action=True,
            )
            alerts.append(alert)

        # Check position size limits
        if position_size > self.risk_limits["max_position_size"]:
            is_safe = False
            alert = self._create_alert(
                AlertType.WARNING,
                "Risk",
                "Position Size Warning",
                f"Position size ${position_size} approaching limit",
                session_id=session_id,
                risk_level=RiskLevel.HIGH,
            )
            alerts.append(alert)

        # Check volatility regime
        risk_level = risk_metrics.get_risk_level()
        if risk_level in [RiskLevel.EXTREME, RiskLevel.CRITICAL]:
            alert = self._create_alert(
                AlertType.WARNING,
                "Market",
                "High Risk Conditions",
                f"Market conditions show {risk_level.value} risk level",
                session_id=session_id,
                risk_level=risk_level,
            )
            alerts.append(alert)

        return is_safe, alerts

    def get_emergency_action(self, session_id: str) -> Optional[str]:
        """Determine if emergency action is needed"""
        if session_id not in self.session_metrics:
            return None

        metrics = self.session_metrics[session_id]

        # Critical drawdown - emergency stop
        if metrics.current_drawdown > Decimal("0.20"):
            return "emergency_stop"

        # Extreme volatility - pause trading
        if metrics.volatility > 0.8:
            return "pause_trading"

        # High correlation risk - reduce exposure
        if metrics.correlation_risk > 0.9:
            return "reduce_exposure"

        return None

    def get_market_assessment(self, symbol: str) -> Optional[MarketConditions]:
        """Get current market condition assessment"""
        return self.market_conditions.get(symbol)

    def get_risk_report(self, session_id: str) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        if session_id not in self.session_metrics:
            return {"error": "Session not found"}

        metrics = self.session_metrics[session_id]
        recent_alerts = [a for a in self.alerts[-10:] if a.session_id == session_id]

        return {
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "risk_level": metrics.get_risk_level().value,
            "metrics": {
                "current_drawdown": float(metrics.current_drawdown),
                "max_drawdown": float(metrics.max_drawdown),
                "position_exposure": float(metrics.position_exposure),
                "volatility": metrics.volatility,
                "sharpe_ratio": metrics.sharpe_ratio,
                "var_95": float(metrics.var_95),
            },
            "recent_alerts": [
                {
                    "type": a.alert_type.value,
                    "title": a.title,
                    "message": a.message,
                    "timestamp": a.timestamp.isoformat(),
                }
                for a in recent_alerts
            ],
        }

    def _initialize_session_monitoring(self, session_id: str, symbol: str):
        """Initialize monitoring for a new session"""
        # Initialize empty metrics
        self.session_metrics[session_id] = RiskMetrics(
            current_drawdown=Decimal("0"),
            max_drawdown=Decimal("0"),
            position_exposure=Decimal("0"),
            leverage_ratio=1.0,
            var_95=Decimal("0"),
            sharpe_ratio=0.0,
            volatility=0.0,
            correlation_risk=0.0,
            liquidity_risk=0.0,
        )

        # Initialize market conditions if not exists
        if symbol not in self.market_conditions:
            self.market_conditions[symbol] = MarketConditions(
                regime=MarketRegime.NORMAL,
                trend_direction="sideways",
                trend_strength=0.5,
                volatility_percentile=0.5,
                volume_profile="normal",
                correlation_breakdown=False,
            )

    def _calculate_risk_metrics(
        self, session_id: str, current_pnl: Decimal, position_size: Decimal
    ) -> RiskMetrics:
        """Calculate current risk metrics"""
        # Get existing metrics or create new
        existing = self.session_metrics.get(session_id)
        if existing:
            max_drawdown = max(
                existing.max_drawdown, abs(min(Decimal("0"), current_pnl))
            )
        else:
            max_drawdown = abs(min(Decimal("0"), current_pnl))

        # Calculate current drawdown
        current_drawdown = abs(min(Decimal("0"), current_pnl)) / max(
            position_size, Decimal("1")
        )

        # Estimate volatility (simplified)
        volatility = min(0.8, float(current_drawdown) * 2)

        # Estimate other metrics (simplified for demo)
        return RiskMetrics(
            current_drawdown=current_drawdown,
            max_drawdown=max_drawdown,
            position_exposure=position_size,
            leverage_ratio=1.0,
            var_95=position_size * Decimal("0.05"),  # 5% VaR estimate
            sharpe_ratio=max(
                -2.0, min(3.0, float(current_pnl) / max(float(position_size), 1) * 10)
            ),
            volatility=volatility,
            correlation_risk=0.3,  # Default moderate correlation
            liquidity_risk=0.2,  # Default low liquidity risk
        )

    def _update_market_conditions(self, symbol: str):
        """Update market condition assessment"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 10:
            return

        prices = [
            float(p[1]) for p in self.price_history[symbol][-50:]
        ]  # Last 50 prices

        # Calculate volatility
        if len(prices) > 1:
            returns = [
                (prices[i] - prices[i - 1]) / prices[i - 1]
                for i in range(1, len(prices))
            ]
            volatility = statistics.stdev(returns) if len(returns) > 1 else 0.0
        else:
            volatility = 0.0

        # Classify regime based on volatility
        if volatility < 0.01:
            regime = MarketRegime.CALM
        elif volatility < 0.03:
            regime = MarketRegime.NORMAL
        elif volatility < 0.06:
            regime = MarketRegime.VOLATILE
        elif volatility < 0.10:
            regime = MarketRegime.EXTREME
        else:
            regime = MarketRegime.CRISIS

        # Trend analysis (simplified)
        if len(prices) >= 20:
            recent_avg = sum(prices[-10:]) / 10
            older_avg = sum(prices[-20:-10]) / 10

            if recent_avg > older_avg * 1.02:
                trend_direction = "up"
                trend_strength = min(1.0, (recent_avg - older_avg) / older_avg * 10)
            elif recent_avg < older_avg * 0.98:
                trend_direction = "down"
                trend_strength = min(1.0, (older_avg - recent_avg) / older_avg * 10)
            else:
                trend_direction = "sideways"
                trend_strength = 0.3
        else:
            trend_direction = "sideways"
            trend_strength = 0.5

        # Update conditions
        self.market_conditions[symbol] = MarketConditions(
            regime=regime,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            volatility_percentile=min(1.0, volatility * 20),
            volume_profile="normal",  # Simplified
            correlation_breakdown=volatility > 0.08,
        )

    def _create_alert(
        self,
        alert_type: AlertType,
        category: str,
        title: str,
        message: str,
        session_id: Optional[str] = None,
        risk_level: Optional[RiskLevel] = None,
        requires_action: bool = False,
    ) -> SafetyAlert:
        """Create and store a safety alert"""
        alert = SafetyAlert(
            timestamp=datetime.now(timezone.utc),
            alert_type=alert_type,
            category=category,
            title=title,
            message=message,
            session_id=session_id,
            risk_level=risk_level,
            requires_action=requires_action,
        )

        with self._lock:
            self.alerts.append(alert)
            # Keep last 1000 alerts
            if len(self.alerts) > 1000:
                self.alerts = self.alerts[-1000:]

        logger.warning(f"Safety Alert: {title} - {message}")
        return alert


# Global advanced safety manager instance
advanced_safety_manager = AdvancedSafetyManager()
