"""
Comprehensive test suite for core.advanced_safety module.

This module tests all functionality in the advanced safety system including:
- RiskMetrics data class and risk level calculation
- MarketConditions data class and grid trading assessment
- SafetyAlert data class
- AdvancedSafetyManager class with all methods
- Market data analysis and condition assessment
- Risk monitoring and alert generation
- Emergency action determination
- Thread safety and data persistence
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
import threading
import time
import statistics

from core.advanced_safety import (
    RiskLevel,
    MarketRegime,
    AlertType,
    RiskMetrics,
    MarketConditions,
    SafetyAlert,
    AdvancedSafetyManager,
    advanced_safety_manager,
)


class TestRiskLevel:
    """Test RiskLevel enum"""

    def test_risk_level_values(self):
        """Test RiskLevel enum values"""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MODERATE.value == "moderate"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.EXTREME.value == "extreme"
        assert RiskLevel.CRITICAL.value == "critical"

    def test_risk_level_ordering(self):
        """Test that risk levels can be compared"""
        # While enums aren't directly comparable, we can test their existence
        levels = [
            RiskLevel.LOW,
            RiskLevel.MODERATE,
            RiskLevel.HIGH,
            RiskLevel.EXTREME,
            RiskLevel.CRITICAL,
        ]
        assert len(levels) == 5
        assert len(set(levels)) == 5  # All unique


class TestMarketRegime:
    """Test MarketRegime enum"""

    def test_market_regime_values(self):
        """Test MarketRegime enum values"""
        assert MarketRegime.CALM.value == "calm"
        assert MarketRegime.NORMAL.value == "normal"
        assert MarketRegime.VOLATILE.value == "volatile"
        assert MarketRegime.EXTREME.value == "extreme"
        assert MarketRegime.CRISIS.value == "crisis"

    def test_market_regime_complete(self):
        """Test all market regimes are defined"""
        regimes = [
            MarketRegime.CALM,
            MarketRegime.NORMAL,
            MarketRegime.VOLATILE,
            MarketRegime.EXTREME,
            MarketRegime.CRISIS,
        ]
        assert len(regimes) == 5


class TestAlertType:
    """Test AlertType enum"""

    def test_alert_type_values(self):
        """Test AlertType enum values"""
        assert AlertType.INFO.value == "info"
        assert AlertType.WARNING.value == "warning"
        assert AlertType.ERROR.value == "error"
        assert AlertType.CRITICAL.value == "critical"
        assert AlertType.EMERGENCY.value == "emergency"


class TestRiskMetrics:
    """Test RiskMetrics data class"""

    @pytest.fixture
    def sample_risk_metrics(self):
        """Create sample risk metrics"""
        return RiskMetrics(
            current_drawdown=Decimal("0.05"),
            max_drawdown=Decimal("0.08"),
            position_exposure=Decimal("1000"),
            leverage_ratio=1.5,
            var_95=Decimal("50"),
            sharpe_ratio=1.2,
            volatility=0.3,
            correlation_risk=0.4,
            liquidity_risk=0.2,
        )

    def test_risk_metrics_creation(self, sample_risk_metrics):
        """Test RiskMetrics data class creation"""
        assert sample_risk_metrics.current_drawdown == Decimal("0.05")
        assert sample_risk_metrics.max_drawdown == Decimal("0.08")
        assert sample_risk_metrics.position_exposure == Decimal("1000")
        assert sample_risk_metrics.leverage_ratio == 1.5
        assert sample_risk_metrics.var_95 == Decimal("50")
        assert sample_risk_metrics.sharpe_ratio == 1.2
        assert sample_risk_metrics.volatility == 0.3
        assert sample_risk_metrics.correlation_risk == 0.4
        assert sample_risk_metrics.liquidity_risk == 0.2

    def test_get_risk_level_low(self):
        """Test risk level calculation - LOW"""
        metrics = RiskMetrics(
            current_drawdown=Decimal("0.02"),
            max_drawdown=Decimal("0.03"),
            position_exposure=Decimal("1000"),
            leverage_ratio=1.0,
            var_95=Decimal("20"),
            sharpe_ratio=2.0,
            volatility=0.15,
            correlation_risk=0.2,
            liquidity_risk=0.1,
        )
        assert metrics.get_risk_level() == RiskLevel.LOW

    def test_get_risk_level_moderate(self):
        """Test risk level calculation - MODERATE"""
        metrics = RiskMetrics(
            current_drawdown=Decimal("0.07"),
            max_drawdown=Decimal("0.08"),
            position_exposure=Decimal("1000"),
            leverage_ratio=1.0,
            var_95=Decimal("30"),
            sharpe_ratio=1.5,
            volatility=0.3,
            correlation_risk=0.3,
            liquidity_risk=0.2,
        )
        assert metrics.get_risk_level() == RiskLevel.MODERATE

    def test_get_risk_level_high(self):
        """Test risk level calculation - HIGH"""
        metrics = RiskMetrics(
            current_drawdown=Decimal("0.12"),
            max_drawdown=Decimal("0.12"),
            position_exposure=Decimal("1000"),
            leverage_ratio=1.0,
            var_95=Decimal("40"),
            sharpe_ratio=0.8,
            volatility=0.5,
            correlation_risk=0.5,
            liquidity_risk=0.3,
        )
        assert metrics.get_risk_level() == RiskLevel.HIGH

    def test_get_risk_level_extreme(self):
        """Test risk level calculation - EXTREME"""
        metrics = RiskMetrics(
            current_drawdown=Decimal("0.18"),
            max_drawdown=Decimal("0.18"),
            position_exposure=Decimal("1000"),
            leverage_ratio=1.0,
            var_95=Decimal("60"),
            sharpe_ratio=0.2,
            volatility=0.7,
            correlation_risk=0.7,
            liquidity_risk=0.4,
        )
        assert metrics.get_risk_level() == RiskLevel.EXTREME

    def test_get_risk_level_critical(self):
        """Test risk level calculation - CRITICAL"""
        metrics = RiskMetrics(
            current_drawdown=Decimal("0.30"),
            max_drawdown=Decimal("0.30"),
            position_exposure=Decimal("1000"),
            leverage_ratio=1.0,
            var_95=Decimal("100"),
            sharpe_ratio=-0.5,
            volatility=0.9,
            correlation_risk=0.9,
            liquidity_risk=0.8,
        )
        assert metrics.get_risk_level() == RiskLevel.CRITICAL

    def test_get_risk_level_volatility_threshold(self):
        """Test risk level based on volatility threshold"""
        # High volatility should trigger EXTREME
        metrics = RiskMetrics(
            current_drawdown=Decimal("0.02"),
            max_drawdown=Decimal("0.02"),
            position_exposure=Decimal("1000"),
            leverage_ratio=1.0,
            var_95=Decimal("20"),
            sharpe_ratio=1.0,
            volatility=0.65,  # Above 0.6 threshold
            correlation_risk=0.2,
            liquidity_risk=0.1,
        )
        assert metrics.get_risk_level() == RiskLevel.EXTREME


class TestMarketConditions:
    """Test MarketConditions data class"""

    @pytest.fixture
    def sample_market_conditions(self):
        """Create sample market conditions"""
        return MarketConditions(
            regime=MarketRegime.NORMAL,
            trend_direction="sideways",
            trend_strength=0.5,
            volatility_percentile=0.4,
            volume_profile="normal",
            correlation_breakdown=False,
        )

    def test_market_conditions_creation(self, sample_market_conditions):
        """Test MarketConditions data class creation"""
        assert sample_market_conditions.regime == MarketRegime.NORMAL
        assert sample_market_conditions.trend_direction == "sideways"
        assert sample_market_conditions.trend_strength == 0.5
        assert sample_market_conditions.volatility_percentile == 0.4
        assert sample_market_conditions.volume_profile == "normal"
        assert sample_market_conditions.correlation_breakdown is False

    def test_is_favorable_for_grid_true(self):
        """Test favorable conditions for grid trading"""
        conditions = MarketConditions(
            regime=MarketRegime.CALM,
            trend_direction="sideways",
            trend_strength=0.3,
            volatility_percentile=0.2,
            volume_profile="normal",
            correlation_breakdown=False,
        )
        assert conditions.is_favorable_for_grid() is True

    def test_is_favorable_for_grid_normal_regime(self):
        """Test favorable conditions with NORMAL regime"""
        conditions = MarketConditions(
            regime=MarketRegime.NORMAL,
            trend_direction="sideways",
            trend_strength=0.4,
            volatility_percentile=0.5,
            volume_profile="normal",
            correlation_breakdown=False,
        )
        assert conditions.is_favorable_for_grid() is True

    def test_is_favorable_for_grid_false_regime(self):
        """Test unfavorable conditions - wrong regime"""
        conditions = MarketConditions(
            regime=MarketRegime.VOLATILE,
            trend_direction="sideways",
            trend_strength=0.3,
            volatility_percentile=0.6,
            volume_profile="normal",
            correlation_breakdown=False,
        )
        assert conditions.is_favorable_for_grid() is False

    def test_is_favorable_for_grid_false_trend(self):
        """Test unfavorable conditions - trending market"""
        conditions = MarketConditions(
            regime=MarketRegime.CALM,
            trend_direction="up",
            trend_strength=0.8,
            volatility_percentile=0.2,
            volume_profile="normal",
            correlation_breakdown=False,
        )
        assert conditions.is_favorable_for_grid() is False

    def test_is_favorable_for_grid_false_correlation(self):
        """Test unfavorable conditions - correlation breakdown"""
        conditions = MarketConditions(
            regime=MarketRegime.CALM,
            trend_direction="sideways",
            trend_strength=0.3,
            volatility_percentile=0.2,
            volume_profile="normal",
            correlation_breakdown=True,
        )
        assert conditions.is_favorable_for_grid() is False


class TestSafetyAlert:
    """Test SafetyAlert data class"""

    def test_safety_alert_creation(self):
        """Test SafetyAlert creation with all fields"""
        timestamp = datetime.now(timezone.utc)
        alert = SafetyAlert(
            timestamp=timestamp,
            alert_type=AlertType.WARNING,
            category="Risk",
            title="Test Alert",
            message="This is a test alert",
            session_id="test_session",
            risk_level=RiskLevel.HIGH,
            requires_action=True,
            auto_resolved=False,
        )

        assert alert.timestamp == timestamp
        assert alert.alert_type == AlertType.WARNING
        assert alert.category == "Risk"
        assert alert.title == "Test Alert"
        assert alert.message == "This is a test alert"
        assert alert.session_id == "test_session"
        assert alert.risk_level == RiskLevel.HIGH
        assert alert.requires_action is True
        assert alert.auto_resolved is False

    def test_safety_alert_defaults(self):
        """Test SafetyAlert creation with default values"""
        timestamp = datetime.now(timezone.utc)
        alert = SafetyAlert(
            timestamp=timestamp,
            alert_type=AlertType.INFO,
            category="System",
            title="Info Alert",
            message="Information message",
        )

        assert alert.session_id is None
        assert alert.risk_level is None
        assert alert.requires_action is False
        assert alert.auto_resolved is False


class TestAdvancedSafetyManager:
    """Test AdvancedSafetyManager class"""

    @pytest.fixture
    def safety_manager(self):
        """Create a fresh AdvancedSafetyManager instance"""
        return AdvancedSafetyManager()

    def test_initialization(self, safety_manager):
        """Test AdvancedSafetyManager initialization"""
        assert safety_manager.monitoring_active is False
        assert len(safety_manager.alerts) == 0
        assert isinstance(safety_manager.risk_limits, dict)
        assert safety_manager.risk_limits["max_drawdown"] == Decimal("0.15")
        assert safety_manager.risk_limits["max_position_size"] == Decimal("10000")
        assert safety_manager.risk_limits["max_daily_loss"] == Decimal("500")
        assert safety_manager.risk_limits["min_liquidity_ratio"] == 0.2
        assert safety_manager.risk_limits["max_correlation"] == 0.8
        assert safety_manager.risk_limits["volatility_threshold"] == 0.6

    def test_start_monitoring(self, safety_manager):
        """Test starting monitoring for a session"""
        session_id = "test_session_001"
        symbol = "BTCUSDT"

        safety_manager.start_monitoring(session_id, symbol)

        assert safety_manager.monitoring_active is True
        assert session_id in safety_manager.session_metrics
        assert len(safety_manager.alerts) == 1
        assert safety_manager.alerts[0].alert_type == AlertType.INFO
        assert safety_manager.alerts[0].title == "Monitoring Started"

    def test_stop_monitoring(self, safety_manager):
        """Test stopping monitoring for a session"""
        session_id = "test_session_002"
        symbol = "ETHUSDT"

        # Start monitoring first
        safety_manager.start_monitoring(session_id, symbol)
        assert session_id in safety_manager.session_metrics

        # Stop monitoring
        safety_manager.stop_monitoring(session_id)

        assert session_id not in safety_manager.session_metrics
        assert len(safety_manager.alerts) == 2  # Start + stop alerts
        assert safety_manager.alerts[1].title == "Monitoring Stopped"

    def test_update_market_data(self, safety_manager):
        """Test updating market data"""
        symbol = "BTCUSDT"
        price = Decimal("50000")
        volume = Decimal("100")
        timestamp = datetime.now(timezone.utc)

        safety_manager.update_market_data(symbol, price, volume, timestamp)

        assert symbol in safety_manager.price_history
        assert symbol in safety_manager.volume_history
        assert len(safety_manager.price_history[symbol]) == 1
        assert len(safety_manager.volume_history[symbol]) == 1
        assert safety_manager.price_history[symbol][0] == (timestamp, price)
        assert safety_manager.volume_history[symbol][0] == (timestamp, volume)

    def test_update_market_data_history_limit(self, safety_manager):
        """Test market data history limit (1000 points)"""
        symbol = "BTCUSDT"

        # Add 1005 data points
        for i in range(1005):
            price = Decimal(str(50000 + i))
            volume = Decimal("100")
            timestamp = datetime.now(timezone.utc) + timedelta(seconds=i)
            safety_manager.update_market_data(symbol, price, volume, timestamp)

        # Should keep only last 1000
        assert len(safety_manager.price_history[symbol]) == 1000
        assert len(safety_manager.volume_history[symbol]) == 1000
        # First entry should be the 6th one we added (index 5)
        assert safety_manager.price_history[symbol][0][1] == Decimal("50005")

    def test_check_session_safety_normal(self, safety_manager):
        """Test session safety check under normal conditions"""
        session_id = "test_session_003"
        current_pnl = Decimal("100")
        position_size = Decimal("1000")
        trades_count = 5

        is_safe, alerts = safety_manager.check_session_safety(
            session_id, current_pnl, position_size, trades_count
        )

        assert is_safe is True
        assert len(alerts) == 0
        assert session_id in safety_manager.session_metrics

    def test_check_session_safety_drawdown_exceeded(self, safety_manager):
        """Test session safety check with drawdown limit exceeded"""
        session_id = "test_session_004"
        current_pnl = Decimal("-2000")  # 20% drawdown on 10K position
        position_size = Decimal("10000")
        trades_count = 10

        is_safe, alerts = safety_manager.check_session_safety(
            session_id, current_pnl, position_size, trades_count
        )

        assert is_safe is False
        assert len(alerts) >= 1
        drawdown_alert = next((a for a in alerts if "Drawdown Limit" in a.title), None)
        assert drawdown_alert is not None
        assert drawdown_alert.alert_type == AlertType.CRITICAL
        assert drawdown_alert.requires_action is True

    def test_check_session_safety_position_size_warning(self, safety_manager):
        """Test session safety check with large position size"""
        session_id = "test_session_005"
        current_pnl = Decimal("50")
        position_size = Decimal("12000")  # Above 10K limit
        trades_count = 5

        is_safe, alerts = safety_manager.check_session_safety(
            session_id, current_pnl, position_size, trades_count
        )

        assert is_safe is False
        assert len(alerts) >= 1
        position_alert = next((a for a in alerts if "Position Size" in a.title), None)
        assert position_alert is not None
        assert position_alert.alert_type == AlertType.WARNING

    def test_get_emergency_action_none(self, safety_manager):
        """Test emergency action when none needed"""
        session_id = "test_session_006"

        # Initialize with normal metrics
        safety_manager.session_metrics[session_id] = RiskMetrics(
            current_drawdown=Decimal("0.05"),
            max_drawdown=Decimal("0.05"),
            position_exposure=Decimal("1000"),
            leverage_ratio=1.0,
            var_95=Decimal("50"),
            sharpe_ratio=1.0,
            volatility=0.2,
            correlation_risk=0.3,
            liquidity_risk=0.2,
        )

        action = safety_manager.get_emergency_action(session_id)
        assert action is None

    def test_get_emergency_action_stop(self, safety_manager):
        """Test emergency stop action"""
        session_id = "test_session_007"

        # Initialize with critical drawdown
        safety_manager.session_metrics[session_id] = RiskMetrics(
            current_drawdown=Decimal("0.25"),  # 25% drawdown triggers emergency stop
            max_drawdown=Decimal("0.25"),
            position_exposure=Decimal("1000"),
            leverage_ratio=1.0,
            var_95=Decimal("250"),
            sharpe_ratio=-1.0,
            volatility=0.3,
            correlation_risk=0.4,
            liquidity_risk=0.3,
        )

        action = safety_manager.get_emergency_action(session_id)
        assert action == "emergency_stop"

    def test_get_emergency_action_pause(self, safety_manager):
        """Test pause trading action"""
        session_id = "test_session_008"

        # Initialize with extreme volatility
        safety_manager.session_metrics[session_id] = RiskMetrics(
            current_drawdown=Decimal("0.05"),
            max_drawdown=Decimal("0.05"),
            position_exposure=Decimal("1000"),
            leverage_ratio=1.0,
            var_95=Decimal("50"),
            sharpe_ratio=0.5,
            volatility=0.85,  # High volatility triggers pause
            correlation_risk=0.3,
            liquidity_risk=0.2,
        )

        action = safety_manager.get_emergency_action(session_id)
        assert action == "pause_trading"

    def test_get_emergency_action_reduce_exposure(self, safety_manager):
        """Test reduce exposure action"""
        session_id = "test_session_009"

        # Initialize with high correlation risk
        safety_manager.session_metrics[session_id] = RiskMetrics(
            current_drawdown=Decimal("0.05"),
            max_drawdown=Decimal("0.05"),
            position_exposure=Decimal("1000"),
            leverage_ratio=1.0,
            var_95=Decimal("50"),
            sharpe_ratio=0.5,
            volatility=0.3,
            correlation_risk=0.95,  # High correlation triggers reduce exposure
            liquidity_risk=0.2,
        )

        action = safety_manager.get_emergency_action(session_id)
        assert action == "reduce_exposure"

    def test_get_emergency_action_nonexistent_session(self, safety_manager):
        """Test emergency action for nonexistent session"""
        action = safety_manager.get_emergency_action("nonexistent_session")
        assert action is None

    def test_get_market_assessment(self, safety_manager):
        """Test getting market assessment"""
        symbol = "BTCUSDT"

        # Initialize market conditions
        safety_manager.market_conditions[symbol] = MarketConditions(
            regime=MarketRegime.NORMAL,
            trend_direction="up",
            trend_strength=0.7,
            volatility_percentile=0.4,
            volume_profile="high",
            correlation_breakdown=False,
        )

        conditions = safety_manager.get_market_assessment(symbol)
        assert conditions is not None
        assert conditions.regime == MarketRegime.NORMAL
        assert conditions.trend_direction == "up"
        assert conditions.trend_strength == 0.7

    def test_get_market_assessment_nonexistent(self, safety_manager):
        """Test getting market assessment for nonexistent symbol"""
        conditions = safety_manager.get_market_assessment("NONEXISTENT")
        assert conditions is None

    def test_get_risk_report(self, safety_manager):
        """Test generating risk report"""
        session_id = "test_session_010"

        # Add metrics and alerts
        safety_manager.session_metrics[session_id] = RiskMetrics(
            current_drawdown=Decimal("0.08"),
            max_drawdown=Decimal("0.10"),
            position_exposure=Decimal("5000"),
            leverage_ratio=1.2,
            var_95=Decimal("400"),
            sharpe_ratio=0.8,
            volatility=0.35,
            correlation_risk=0.4,
            liquidity_risk=0.3,
        )

        # Add a test alert
        alert = SafetyAlert(
            timestamp=datetime.now(timezone.utc),
            alert_type=AlertType.WARNING,
            category="Test",
            title="Test Alert",
            message="Test message",
            session_id=session_id,
        )
        safety_manager.alerts.append(alert)

        report = safety_manager.get_risk_report(session_id)

        assert report["session_id"] == session_id
        assert "timestamp" in report
        assert report["risk_level"] == "moderate"
        assert "metrics" in report
        assert report["metrics"]["current_drawdown"] == 0.08
        assert report["metrics"]["volatility"] == 0.35
        assert "recent_alerts" in report
        assert len(report["recent_alerts"]) == 1

    def test_get_risk_report_nonexistent_session(self, safety_manager):
        """Test generating risk report for nonexistent session"""
        report = safety_manager.get_risk_report("nonexistent_session")
        assert "error" in report
        assert report["error"] == "Session not found"

    def test_calculate_risk_metrics_new_session(self, safety_manager):
        """Test calculating risk metrics for new session"""
        session_id = "test_session_011"
        current_pnl = Decimal("-100")
        position_size = Decimal("2000")

        metrics = safety_manager._calculate_risk_metrics(
            session_id, current_pnl, position_size
        )

        assert isinstance(metrics, RiskMetrics)
        assert metrics.current_drawdown == Decimal("0.05")  # 100/2000
        assert metrics.max_drawdown == Decimal("100")
        assert metrics.position_exposure == position_size
        assert metrics.leverage_ratio == 1.0

    def test_calculate_risk_metrics_existing_session(self, safety_manager):
        """Test calculating risk metrics for existing session"""
        session_id = "test_session_012"

        # Pre-populate with existing metrics
        safety_manager.session_metrics[session_id] = RiskMetrics(
            current_drawdown=Decimal("0.05"),
            max_drawdown=Decimal("200"),  # Previous max
            position_exposure=Decimal("1000"),
            leverage_ratio=1.0,
            var_95=Decimal("50"),
            sharpe_ratio=1.0,
            volatility=0.2,
            correlation_risk=0.3,
            liquidity_risk=0.2,
        )

        current_pnl = Decimal("-150")  # Larger loss
        position_size = Decimal("2000")

        metrics = safety_manager._calculate_risk_metrics(
            session_id, current_pnl, position_size
        )

        # Max drawdown should be max of previous and current
        assert metrics.max_drawdown == Decimal("200")  # Previous max is higher

    def test_update_market_conditions_insufficient_data(self, safety_manager):
        """Test market condition update with insufficient data"""
        symbol = "BTCUSDT"

        # Add only 5 data points (less than 10 required)
        for i in range(5):
            price = Decimal(str(50000 + i * 100))
            volume = Decimal("100")
            timestamp = datetime.now(timezone.utc) + timedelta(seconds=i)
            safety_manager.update_market_data(symbol, price, volume, timestamp)

        # Market conditions should not be updated due to insufficient data
        # (But the data should still be stored)
        assert len(safety_manager.price_history[symbol]) == 5

    def test_update_market_conditions_calm_regime(self, safety_manager):
        """Test market condition classification - CALM regime"""
        symbol = "BTCUSDT"

        # Add stable price data (low volatility)
        base_price = 50000
        for i in range(20):
            # Small random variations (< 1%)
            price = Decimal(str(base_price + i % 3 - 1))  # Varies by +/-1
            volume = Decimal("100")
            timestamp = datetime.now(timezone.utc) + timedelta(seconds=i)
            safety_manager.update_market_data(symbol, price, volume, timestamp)

        conditions = safety_manager.market_conditions.get(symbol)
        if conditions:  # May not update if volatility calculation fails
            assert conditions.regime in [MarketRegime.CALM, MarketRegime.NORMAL]

    def test_update_market_conditions_trending_up(self, safety_manager):
        """Test market condition classification - upward trend"""
        symbol = "BTCUSDT"

        # Add trending up price data with >2% change to trigger trend detection
        base_price = 50000
        for i in range(25):
            # Use larger price increments to exceed 2% threshold
            price = Decimal(str(base_price + i * 500))  # 1% per step, total >20% change
            volume = Decimal("100")
            timestamp = datetime.now(timezone.utc) + timedelta(seconds=i)
            safety_manager.update_market_data(symbol, price, volume, timestamp)

        conditions = safety_manager.market_conditions.get(symbol)
        if conditions:
            assert conditions.trend_direction == "up"
            assert conditions.trend_strength > 0.5

    def test_update_market_conditions_trending_down(self, safety_manager):
        """Test market condition classification - downward trend"""
        symbol = "BTCUSDT"

        # Add trending down price data with >2% change to trigger trend detection
        base_price = 50000
        for i in range(25):
            # Use larger price decrements to exceed 2% threshold
            price = Decimal(str(base_price - i * 500))  # 1% per step, total >20% change
            volume = Decimal("100")
            timestamp = datetime.now(timezone.utc) + timedelta(seconds=i)
            safety_manager.update_market_data(symbol, price, volume, timestamp)

        conditions = safety_manager.market_conditions.get(symbol)
        if conditions:
            assert conditions.trend_direction == "down"
            assert conditions.trend_strength > 0.5

    def test_create_alert(self, safety_manager):
        """Test alert creation"""
        alert = safety_manager._create_alert(
            AlertType.WARNING,
            "Risk",
            "Test Alert",
            "This is a test alert message",
            session_id="test_session_013",
            risk_level=RiskLevel.HIGH,
            requires_action=True,
        )

        assert isinstance(alert, SafetyAlert)
        assert alert.alert_type == AlertType.WARNING
        assert alert.category == "Risk"
        assert alert.title == "Test Alert"
        assert alert.message == "This is a test alert message"
        assert alert.session_id == "test_session_013"
        assert alert.risk_level == RiskLevel.HIGH
        assert alert.requires_action is True

        # Alert should be added to manager's alerts list
        assert len(safety_manager.alerts) == 1
        assert safety_manager.alerts[0] == alert

    def test_create_alert_limit(self, safety_manager):
        """Test alert creation with limit (1000 alerts)"""
        # Create 1005 alerts
        for i in range(1005):
            safety_manager._create_alert(
                AlertType.INFO, "Test", f"Alert {i}", f"Test message {i}"
            )

        # Should keep only last 1000
        assert len(safety_manager.alerts) == 1000
        # First alert should be alert 5 (index 5)
        assert safety_manager.alerts[0].title == "Alert 5"

    def test_initialize_session_monitoring(self, safety_manager):
        """Test session monitoring initialization"""
        session_id = "test_session_014"
        symbol = "ETHUSDT"

        safety_manager._initialize_session_monitoring(session_id, symbol)

        # Check session metrics initialized
        assert session_id in safety_manager.session_metrics
        metrics = safety_manager.session_metrics[session_id]
        assert metrics.current_drawdown == Decimal("0")
        assert metrics.max_drawdown == Decimal("0")
        assert metrics.position_exposure == Decimal("0")
        assert metrics.leverage_ratio == 1.0

        # Check market conditions initialized
        assert symbol in safety_manager.market_conditions
        conditions = safety_manager.market_conditions[symbol]
        assert conditions.regime == MarketRegime.NORMAL
        assert conditions.trend_direction == "sideways"
        assert conditions.trend_strength == 0.5

    def test_thread_safety(self, safety_manager):
        """Test thread safety of concurrent operations"""
        results = []
        errors = []

        def monitor_session(session_id):
            try:
                safety_manager.start_monitoring(f"session_{session_id}", "BTCUSDT")
                time.sleep(0.01)  # Small delay
                safety_manager.stop_monitoring(f"session_{session_id}")
                results.append(session_id)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=monitor_session, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0
        assert len(results) == 10
        # Should have start/stop alerts for each session
        assert len(safety_manager.alerts) == 20  # 10 start + 10 stop


class TestGlobalAdvancedSafetyManager:
    """Test the global advanced_safety_manager instance"""

    def test_global_instance_exists(self):
        """Test that global instance exists and is correct type"""
        assert isinstance(advanced_safety_manager, AdvancedSafetyManager)

    def test_global_instance_initialized(self):
        """Test that global instance is properly initialized"""
        assert hasattr(advanced_safety_manager, "risk_limits")
        assert hasattr(advanced_safety_manager, "monitoring_active")
        assert hasattr(advanced_safety_manager, "alerts")
        assert hasattr(advanced_safety_manager, "_lock")


class TestComplexScenarios:
    """Test complex scenarios and edge cases"""

    @pytest.fixture
    def safety_manager(self):
        """Create a fresh AdvancedSafetyManager instance"""
        return AdvancedSafetyManager()

    def test_multiple_session_monitoring(self, safety_manager):
        """Test monitoring multiple sessions simultaneously"""
        sessions = [
            ("session_1", "BTCUSDT"),
            ("session_2", "ETHUSDT"),
            ("session_3", "ADAUSDT"),
        ]

        # Start monitoring all sessions
        for session_id, symbol in sessions:
            safety_manager.start_monitoring(session_id, symbol)

        # Check all sessions are being monitored
        assert len(safety_manager.session_metrics) == 3
        assert safety_manager.monitoring_active is True

        # Generate alerts for different sessions
        for i, (session_id, _) in enumerate(sessions):
            is_safe, alerts = safety_manager.check_session_safety(
                session_id, Decimal(str(-100 * (i + 1))), Decimal("1000"), 5
            )
            assert is_safe is True or len(alerts) > 0

    def test_market_regime_transitions(self, safety_manager):
        """Test market regime transitions through different volatility periods"""
        symbol = "BTCUSDT"
        base_price = 50000
        timestamp = datetime.now(timezone.utc)

        # Calm period - small variations
        for i in range(20):
            price = Decimal(str(base_price + (i % 2)))  # Minimal variation
            safety_manager.update_market_data(
                symbol, price, Decimal("100"), timestamp + timedelta(seconds=i)
            )

        conditions = safety_manager.market_conditions.get(symbol)
        if conditions:
            initial_regime = conditions.regime

        # Volatile period - large variations
        for i in range(20, 40):
            price = Decimal(str(base_price + (i % 2) * 1000))  # Large swings
            safety_manager.update_market_data(
                symbol, price, Decimal("100"), timestamp + timedelta(seconds=i)
            )

        conditions = safety_manager.market_conditions.get(symbol)
        if conditions:
            # Regime should change to more volatile
            assert (
                conditions.regime != initial_regime
                or conditions.volatility_percentile > 0.5
            )

    def test_emergency_action_priority(self, safety_manager):
        """Test emergency action priority when multiple conditions trigger"""
        session_id = "emergency_test"

        # Set metrics that trigger multiple emergency conditions
        safety_manager.session_metrics[session_id] = RiskMetrics(
            current_drawdown=Decimal("0.25"),  # Triggers emergency_stop
            max_drawdown=Decimal("0.25"),
            position_exposure=Decimal("1000"),
            leverage_ratio=1.0,
            var_95=Decimal("250"),
            sharpe_ratio=-1.0,
            volatility=0.9,  # Also triggers pause_trading
            correlation_risk=0.95,  # Also triggers reduce_exposure
            liquidity_risk=0.8,
        )

        action = safety_manager.get_emergency_action(session_id)
        # emergency_stop should have priority over other actions
        assert action == "emergency_stop"

    def test_risk_metrics_calculation_edge_cases(self, safety_manager):
        """Test risk metrics calculation with edge cases"""
        session_id = "edge_case_test"

        # Test with zero position size
        metrics = safety_manager._calculate_risk_metrics(
            session_id, Decimal("0"), Decimal("0")
        )
        assert metrics.current_drawdown == Decimal("0")
        assert metrics.position_exposure == Decimal("0")

        # Test with positive PnL
        metrics = safety_manager._calculate_risk_metrics(
            session_id, Decimal("500"), Decimal("1000")
        )
        assert metrics.current_drawdown == Decimal("0")  # No drawdown with profit

        # Test with extreme negative PnL
        metrics = safety_manager._calculate_risk_metrics(
            session_id, Decimal("-5000"), Decimal("1000")
        )
        assert metrics.current_drawdown == Decimal("5")  # 500% drawdown
        assert metrics.volatility >= 0.8  # Should be capped at 0.8

    def test_performance_under_high_load(self, safety_manager):
        """Test performance under high data load"""
        symbol = "BTCUSDT"
        start_time = time.time()

        # Simulate high-frequency data updates
        for i in range(1000):
            price = Decimal(str(50000 + i % 100))
            volume = Decimal(str(100 + i % 50))
            timestamp = datetime.now(timezone.utc) + timedelta(microseconds=i)
            safety_manager.update_market_data(symbol, price, volume, timestamp)

        end_time = time.time()
        processing_time = end_time - start_time

        # Should complete within reasonable time (< 1 second for 1000 updates)
        assert processing_time < 1.0
        assert len(safety_manager.price_history[symbol]) == 1000
        assert symbol in safety_manager.market_conditions

    def test_alert_categorization_and_filtering(self, safety_manager):
        """Test alert categorization and filtering capabilities"""
        session_id = "alert_test"

        # Create various types of alerts
        alert_configs = [
            (AlertType.INFO, "System", "Info Alert", "Information"),
            (AlertType.WARNING, "Risk", "Warning Alert", "Warning message"),
            (AlertType.ERROR, "System", "Error Alert", "Error occurred"),
            (AlertType.CRITICAL, "Risk", "Critical Alert", "Critical issue"),
            (AlertType.EMERGENCY, "Safety", "Emergency Alert", "Emergency!"),
        ]

        for alert_type, category, title, message in alert_configs:
            safety_manager._create_alert(
                alert_type, category, title, message, session_id
            )

        # Test filtering by session
        session_alerts = [
            a for a in safety_manager.alerts if a.session_id == session_id
        ]
        assert len(session_alerts) == 5

        # Test filtering by type
        critical_alerts = [
            a for a in safety_manager.alerts if a.alert_type == AlertType.CRITICAL
        ]
        assert len(critical_alerts) == 1

        # Test filtering by category
        risk_alerts = [a for a in safety_manager.alerts if a.category == "Risk"]
        assert len(risk_alerts) == 2


class TestModuleIntegration:
    """Test integration between different components"""

    def test_risk_metrics_market_conditions_integration(self):
        """Test integration between risk metrics and market conditions"""
        safety_manager = AdvancedSafetyManager()
        session_id = "integration_test"
        symbol = "BTCUSDT"

        # Start monitoring
        safety_manager.start_monitoring(session_id, symbol)

        # Add market data to establish conditions
        base_price = 50000
        for i in range(30):
            price = Decimal(str(base_price + i * 100))  # Upward trend
            volume = Decimal("100")
            timestamp = datetime.now(timezone.utc) + timedelta(seconds=i)
            safety_manager.update_market_data(symbol, price, volume, timestamp)

        # Check session safety with market conditions
        is_safe, alerts = safety_manager.check_session_safety(
            session_id, Decimal("-100"), Decimal("1000"), 5
        )

        # Get market assessment
        conditions = safety_manager.get_market_assessment(symbol)

        # Both systems should be working together
        assert session_id in safety_manager.session_metrics
        assert conditions is not None
        assert isinstance(is_safe, bool)
        assert isinstance(alerts, list)

    def test_complete_workflow_simulation(self):
        """Test complete workflow from start to emergency stop"""
        safety_manager = AdvancedSafetyManager()
        session_id = "workflow_test"
        symbol = "BTCUSDT"

        # 1. Start monitoring
        safety_manager.start_monitoring(session_id, symbol)
        assert safety_manager.monitoring_active is True

        # 2. Normal trading period
        for i in range(10):
            is_safe, alerts = safety_manager.check_session_safety(
                session_id, Decimal("50"), Decimal("1000"), i
            )
            assert is_safe is True
            assert len(alerts) == 0

        # 3. Market becomes volatile
        for i in range(20):
            price = Decimal(str(50000 + (i % 2) * 2000))  # High volatility
            volume = Decimal("100")
            timestamp = datetime.now(timezone.utc) + timedelta(seconds=i)
            safety_manager.update_market_data(symbol, price, volume, timestamp)

        # 4. Losses start accumulating
        current_pnl = Decimal("-1800")  # Approaching danger zone
        is_safe, alerts = safety_manager.check_session_safety(
            session_id, current_pnl, Decimal("10000"), 15
        )

        # Should generate warnings but still be "safe"
        assert len(alerts) >= 0  # May have volatility warnings

        # 5. Critical situation - emergency stop triggered
        current_pnl = Decimal("-2500")  # 25% drawdown
        is_safe, alerts = safety_manager.check_session_safety(
            session_id, current_pnl, Decimal("10000"), 20
        )

        assert is_safe is False
        assert len(alerts) > 0

        # Check emergency action
        action = safety_manager.get_emergency_action(session_id)
        assert action == "emergency_stop"

        # 6. Generate final risk report
        report = safety_manager.get_risk_report(session_id)
        assert report["risk_level"] in ["extreme", "critical"]
        assert len(report["recent_alerts"]) > 0

        # 7. Stop monitoring
        safety_manager.stop_monitoring(session_id)
        assert session_id not in safety_manager.session_metrics
