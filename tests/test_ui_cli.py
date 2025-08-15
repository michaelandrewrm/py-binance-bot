"""
Test Suite for UI Module - CLI Component (cli.py)

This module provides comprehensive unit tests for the CLI interface including:
- TradingMode enum functionality
- GridParameters data class and validation
- TradingSession lifecycle management
- SessionManager operations
- CLI command functions and utility methods
- Parameter collection and validation
- Duration parsing and formatting
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from io import StringIO
import json
import tempfile
import os
from typing import Optional

# Import the modules under test
from ui.cli import (
    TradingMode,
    GridParameters,
    TradingSession,
    TradingSessionStatus,
    SessionManager,
    get_market_context,
    collect_manual_parameters,
    get_ai_suggested_parameters,
    confirm_ai_parameters,
    parse_duration,
    format_duration,
    get_status_style,
    show_static_status,
    show_live_status,
    start_manual_grid,
    start_ai_grid,
    session_manager,
)

from core.grid_engine import GridConfig


class TestTradingMode:
    """Test TradingMode enum functionality"""

    def test_trading_mode_values(self):
        """Test that TradingMode enum has correct values"""
        assert TradingMode.MANUAL.value == "manual"
        assert TradingMode.AI.value == "ai"

    def test_trading_mode_membership(self):
        """Test membership operations on TradingMode enum"""
        assert TradingMode.MANUAL in TradingMode
        assert TradingMode.AI in TradingMode
        assert "invalid_mode" not in [mode.value for mode in TradingMode]


class TestGridParameters:
    """Test GridParameters data class functionality"""

    def test_grid_parameters_creation(self):
        """Test creating GridParameters with valid inputs"""
        params = GridParameters(
            symbol="BTCUSDC",
            lower_bound=Decimal("48000"),
            upper_bound=Decimal("52000"),
            grid_count=20,
            investment_per_grid=Decimal("50"),
            center_price=Decimal("50000"),
            mode=TradingMode.MANUAL,
            confidence=0.85,
        )

        assert params.symbol == "BTCUSDC"
        assert params.lower_bound == Decimal("48000")
        assert params.upper_bound == Decimal("52000")
        assert params.grid_count == 20
        assert params.investment_per_grid == Decimal("50")
        assert params.center_price == Decimal("50000")
        assert params.mode == TradingMode.MANUAL
        assert params.confidence == 0.85

    def test_grid_parameters_defaults(self):
        """Test GridParameters with default values"""
        params = GridParameters(
            symbol="ETHUSDC",
            lower_bound=Decimal("3000"),
            upper_bound=Decimal("4000"),
            grid_count=10,
            investment_per_grid=Decimal("100"),
        )

        assert params.center_price is None
        assert params.mode == TradingMode.MANUAL
        assert params.confidence is None

    def test_to_grid_config_with_center_price(self):
        """Test conversion to GridConfig with explicit center price"""
        params = GridParameters(
            symbol="BTCUSDC",
            lower_bound=Decimal("48000"),
            upper_bound=Decimal("52000"),
            grid_count=20,
            investment_per_grid=Decimal("50"),
            center_price=Decimal("50000"),
        )

        config = params.to_grid_config()

        assert isinstance(config, GridConfig)
        assert config.center_price == Decimal("50000")
        assert config.grid_spacing == Decimal("200")  # (52000-48000)/20
        assert config.num_levels_up == 10  # 20//2
        assert config.num_levels_down == 10  # 20//2
        assert config.order_amount == Decimal("50")
        assert config.max_position_size == Decimal("1000")  # 50*20

    def test_to_grid_config_without_center_price(self):
        """Test conversion to GridConfig with calculated center price"""
        params = GridParameters(
            symbol="ETHUSDC",
            lower_bound=Decimal("3000"),
            upper_bound=Decimal("4000"),
            grid_count=10,
            investment_per_grid=Decimal("100"),
        )

        config = params.to_grid_config()

        assert config.center_price == Decimal("3500")  # (3000+4000)/2
        assert config.grid_spacing == Decimal("100")  # (4000-3000)/10
        assert config.num_levels_up == 5
        assert config.num_levels_down == 5
        assert config.order_amount == Decimal("100")
        assert config.max_position_size == Decimal("1000")  # 100*10


class TestTradingSessionStatus:
    """Test TradingSessionStatus enum functionality"""

    def test_session_status_values(self):
        """Test that all session status values are correct"""
        assert TradingSessionStatus.STOPPED.value == "stopped"
        assert TradingSessionStatus.STARTING.value == "starting"
        assert TradingSessionStatus.RUNNING.value == "running"
        assert TradingSessionStatus.PAUSED.value == "paused"
        assert TradingSessionStatus.STOPPING.value == "stopping"
        assert TradingSessionStatus.ERROR.value == "error"
        assert TradingSessionStatus.COMPLETED.value == "completed"


class TestTradingSession:
    """Test TradingSession data class functionality"""

    def test_trading_session_creation(self):
        """Test creating a TradingSession with required parameters"""
        params = GridParameters(
            symbol="BTCUSDC",
            lower_bound=Decimal("48000"),
            upper_bound=Decimal("52000"),
            grid_count=20,
            investment_per_grid=Decimal("50"),
        )

        session = TradingSession(
            session_id="test-session-1", symbol="BTCUSDC", parameters=params
        )

        assert session.session_id == "test-session-1"
        assert session.symbol == "BTCUSDC"
        assert session.parameters == params
        assert session.status == TradingSessionStatus.STOPPED
        assert session.start_time is None
        assert session.end_time is None
        assert session.duration_limit is None
        assert session.pause_time is None
        assert session.total_paused_duration == timedelta()
        assert session.trades_count == 0
        assert session.profit_loss == Decimal("0")
        assert session.current_positions == 0

    def test_is_active_method(self):
        """Test is_active method for different statuses"""
        params = GridParameters(
            "BTCUSDC", Decimal("48000"), Decimal("52000"), 20, Decimal("50")
        )
        session = TradingSession("test-1", "BTCUSDC", params)

        # Test inactive states
        session.status = TradingSessionStatus.STOPPED
        assert not session.is_active()

        session.status = TradingSessionStatus.PAUSED
        assert not session.is_active()

        session.status = TradingSessionStatus.ERROR
        assert not session.is_active()

        session.status = TradingSessionStatus.COMPLETED
        assert not session.is_active()

        # Test active states
        session.status = TradingSessionStatus.RUNNING
        assert session.is_active()

        session.status = TradingSessionStatus.STARTING
        assert session.is_active()

    def test_is_time_expired_no_limit(self):
        """Test is_time_expired when no duration limit is set"""
        params = GridParameters(
            "BTCUSDC", Decimal("48000"), Decimal("52000"), 20, Decimal("50")
        )
        session = TradingSession("test-1", "BTCUSDC", params)
        session.start_time = datetime.now(timezone.utc)

        assert not session.is_time_expired()

    def test_is_time_expired_no_start_time(self):
        """Test is_time_expired when no start time is set"""
        params = GridParameters(
            "BTCUSDC", Decimal("48000"), Decimal("52000"), 20, Decimal("50")
        )
        session = TradingSession("test-1", "BTCUSDC", params)
        session.duration_limit = timedelta(hours=1)

        assert not session.is_time_expired()

    def test_is_time_expired_with_limit(self):
        """Test is_time_expired with duration limit"""
        params = GridParameters(
            "BTCUSDC", Decimal("48000"), Decimal("52000"), 20, Decimal("50")
        )
        session = TradingSession("test-1", "BTCUSDC", params)
        session.duration_limit = timedelta(hours=1)
        session.start_time = datetime.now(timezone.utc) - timedelta(
            hours=2
        )  # Started 2 hours ago

        assert session.is_time_expired()

    def test_is_time_expired_with_paused_time(self):
        """Test is_time_expired accounting for paused time"""
        params = GridParameters(
            "BTCUSDC", Decimal("48000"), Decimal("52000"), 20, Decimal("50")
        )
        session = TradingSession("test-1", "BTCUSDC", params)
        session.duration_limit = timedelta(hours=1)
        session.start_time = datetime.now(timezone.utc) - timedelta(hours=2)
        session.total_paused_duration = timedelta(
            hours=1, minutes=30
        )  # Paused for 1.5 hours

        assert not session.is_time_expired()  # Only 0.5 hours of actual trading time

    def test_get_elapsed_time_no_start(self):
        """Test get_elapsed_time when session hasn't started"""
        params = GridParameters(
            "BTCUSDC", Decimal("48000"), Decimal("52000"), 20, Decimal("50")
        )
        session = TradingSession("test-1", "BTCUSDC", params)

        elapsed = session.get_elapsed_time()
        assert elapsed == timedelta()

    def test_get_elapsed_time_with_start(self):
        """Test get_elapsed_time with start time set"""
        params = GridParameters(
            "BTCUSDC", Decimal("48000"), Decimal("52000"), 20, Decimal("50")
        )
        session = TradingSession("test-1", "BTCUSDC", params)
        session.start_time = datetime.now(timezone.utc) - timedelta(hours=1)

        elapsed = session.get_elapsed_time()
        assert abs(elapsed - timedelta(hours=1)) < timedelta(
            seconds=1
        )  # Allow small tolerance

    def test_get_elapsed_time_with_paused(self):
        """Test get_elapsed_time accounting for paused duration"""
        params = GridParameters(
            "BTCUSDC", Decimal("48000"), Decimal("52000"), 20, Decimal("50")
        )
        session = TradingSession("test-1", "BTCUSDC", params)
        session.start_time = datetime.now(timezone.utc) - timedelta(hours=2)
        session.total_paused_duration = timedelta(minutes=30)

        elapsed = session.get_elapsed_time()
        expected = timedelta(hours=1, minutes=30)  # 2 hours - 30 minutes paused
        assert abs(elapsed - expected) < timedelta(seconds=1)


class TestSessionManager:
    """Test SessionManager functionality"""

    def setup_method(self):
        """Setup fresh SessionManager for each test"""
        self.manager = SessionManager()

    def test_session_manager_initialization(self):
        """Test SessionManager initializes with empty state"""
        assert self.manager.sessions == {}
        assert self.manager.active_session_id is None

    def test_create_session(self):
        """Test creating a new trading session"""
        params = GridParameters(
            "BTCUSDC", Decimal("48000"), Decimal("52000"), 20, Decimal("50")
        )

        session_id = self.manager.create_session("BTCUSDC", params)

        assert session_id in self.manager.sessions
        session = self.manager.sessions[session_id]
        assert session.symbol == "BTCUSDC"
        assert session.parameters == params
        assert session.status == TradingSessionStatus.STOPPED

    def test_create_session_with_duration(self):
        """Test creating session with duration limit"""
        params = GridParameters(
            "BTCUSDC", Decimal("48000"), Decimal("52000"), 20, Decimal("50")
        )
        duration_limit = timedelta(hours=2)

        session_id = self.manager.create_session("BTCUSDC", params, duration_limit)

        session = self.manager.sessions[session_id]
        assert session.duration_limit == duration_limit

    def test_get_session_existing(self):
        """Test getting an existing session"""
        params = GridParameters(
            "BTCUSDC", Decimal("48000"), Decimal("52000"), 20, Decimal("50")
        )
        session_id = self.manager.create_session("BTCUSDC", params)

        retrieved_session = self.manager.get_session(session_id)

        assert retrieved_session is not None
        assert retrieved_session.session_id == session_id
        assert retrieved_session.symbol == "BTCUSDC"

    def test_get_session_nonexistent(self):
        """Test getting a non-existent session returns None"""
        result = self.manager.get_session("nonexistent-id")
        assert result is None

    def test_get_active_session_none(self):
        """Test getting active session when none is set"""
        assert self.manager.get_active_session() is None

    def test_set_active_session_valid(self):
        """Test setting active session with valid session ID"""
        params = GridParameters(
            "BTCUSDC", Decimal("48000"), Decimal("52000"), 20, Decimal("50")
        )
        session_id = self.manager.create_session("BTCUSDC", params)

        result = self.manager.set_active_session(session_id)

        assert result is True
        assert self.manager.active_session_id == session_id

        active_session = self.manager.get_active_session()
        assert active_session is not None
        assert active_session.session_id == session_id

    def test_set_active_session_invalid(self):
        """Test setting active session with invalid session ID"""
        result = self.manager.set_active_session("nonexistent-id")

        assert result is False
        assert self.manager.active_session_id is None


class TestUtilityFunctions:
    """Test utility functions in the CLI module"""

    def test_parse_duration_hours(self):
        """Test parsing duration strings with hours"""
        duration = parse_duration("2h")
        assert duration == timedelta(hours=2)

        duration = parse_duration("1h")
        assert duration == timedelta(hours=1)

        duration = parse_duration("24h")
        assert duration == timedelta(hours=24)

    def test_parse_duration_minutes(self):
        """Test parsing duration strings with minutes"""
        duration = parse_duration("30m")
        assert duration == timedelta(minutes=30)

        duration = parse_duration("15m")
        assert duration == timedelta(minutes=15)

        duration = parse_duration("120m")
        assert duration == timedelta(minutes=120)

    def test_parse_duration_days(self):
        """Test parsing duration strings with days"""
        duration = parse_duration("1d")
        assert duration == timedelta(days=1)

        duration = parse_duration("7d")
        assert duration == timedelta(days=7)

    def test_parse_duration_invalid_format(self):
        """Test parsing invalid duration strings raises ValueError"""
        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("invalid")

        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("2x")

        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("")

    def test_parse_duration_invalid_number(self):
        """Test parsing duration with invalid numbers raises ValueError"""
        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("ah")

        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("-5h")

    def test_format_duration_hours(self):
        """Test formatting timedelta to duration strings with hours"""
        duration = timedelta(hours=2)
        assert format_duration(duration) == "2h 0m"

        duration = timedelta(hours=1, minutes=30)
        assert format_duration(duration) == "1h 30m"

    def test_format_duration_minutes_only(self):
        """Test formatting timedelta with only minutes"""
        duration = timedelta(minutes=45)
        assert format_duration(duration) == "45m"

        duration = timedelta(minutes=5)
        assert format_duration(duration) == "5m"

    def test_format_duration_days(self):
        """Test formatting timedelta with days"""
        duration = timedelta(days=1, hours=2, minutes=30)
        assert format_duration(duration) == "1d 2h 30m"

        duration = timedelta(days=3)
        assert format_duration(duration) == "3d"

    def test_format_duration_zero(self):
        """Test formatting zero timedelta"""
        duration = timedelta()
        assert format_duration(duration) == "0m"

    def test_get_status_style(self):
        """Test getting Rich styles for different session statuses"""
        assert get_status_style(TradingSessionStatus.RUNNING) == "green"
        assert get_status_style(TradingSessionStatus.STOPPED) == "red"
        assert get_status_style(TradingSessionStatus.PAUSED) == "yellow"
        assert get_status_style(TradingSessionStatus.ERROR) == "red"
        assert get_status_style(TradingSessionStatus.STARTING) == "blue"
        assert get_status_style(TradingSessionStatus.STOPPING) == "yellow"
        assert get_status_style(TradingSessionStatus.COMPLETED) == "cyan"


class TestAsyncFunctions:
    """Test async functions in the CLI module"""

    @pytest.mark.asyncio
    async def test_get_market_context_success(self):
        """Test getting market context for a valid symbol"""
        with patch("ui.cli.create_binance_client") as mock_client_factory:
            mock_client = AsyncMock()
            mock_client.get_ticker_price.return_value = {"price": "50000.0"}
            mock_client.get_24hr_ticker.return_value = {
                "priceChangePercent": "2.5",
                "volume": "1000.0",
            }
            mock_client_factory.return_value = mock_client

            context = await get_market_context("BTCUSDC")

            assert "current_price" in context
            assert "price_change_24h" in context
            assert "volume_24h" in context
            assert context["current_price"] == 50000.0
            assert context["price_change_24h"] == 2.5
            assert context["volume_24h"] == 1000.0

    @pytest.mark.asyncio
    async def test_get_market_context_api_error(self):
        """Test get_market_context handling API errors"""
        with patch("ui.cli.create_binance_client") as mock_client_factory:
            mock_client = AsyncMock()
            mock_client.get_ticker_price.side_effect = Exception("API Error")
            mock_client_factory.return_value = mock_client

            context = await get_market_context("BTCUSDC")

            # Should return defaults when API fails
            assert "current_price" in context
            assert context["current_price"] is None

    @pytest.mark.asyncio
    async def test_get_ai_suggested_parameters_success(self):
        """Test AI parameter suggestion functionality"""
        with patch("ui.cli.get_market_context") as mock_context, patch(
            "ui.cli.BayesianOptimizer"
        ) as mock_optimizer:

            # Mock market context
            mock_context.return_value = {
                "current_price": 50000.0,
                "price_change_24h": 2.5,
                "volume_24h": 1000.0,
            }

            # Mock AI optimizer
            mock_opt_instance = AsyncMock()
            mock_opt_instance.suggest_grid_parameters.return_value = {
                "lower_bound": 48000.0,
                "upper_bound": 52000.0,
                "grid_count": 20,
                "investment_per_grid": 50.0,
                "confidence": 0.85,
            }
            mock_optimizer.return_value = mock_opt_instance

            params = await get_ai_suggested_parameters("BTCUSDC")

            assert params.symbol == "BTCUSDC"
            assert params.lower_bound == Decimal("48000.0")
            assert params.upper_bound == Decimal("52000.0")
            assert params.grid_count == 20
            assert params.investment_per_grid == Decimal("50.0")
            assert params.mode == TradingMode.AI
            assert params.confidence == 0.85

    @pytest.mark.asyncio
    async def test_get_ai_suggested_parameters_no_optimizer(self):
        """Test AI parameter suggestion when optimizer is not available"""
        with patch("ui.cli.get_market_context") as mock_context, patch(
            "ui.cli.BayesianOptimizer", side_effect=ImportError
        ):

            mock_context.return_value = {
                "current_price": 50000.0,
                "price_change_24h": 2.5,
                "volume_24h": 1000.0,
            }

            params = await get_ai_suggested_parameters("BTCUSDC")

            # Should fall back to simple heuristics
            assert params.symbol == "BTCUSDC"
            assert params.mode == TradingMode.AI
            assert params.confidence is not None
            assert 0 <= params.confidence <= 1


class TestCLICommands:
    """Test CLI command functions"""

    def test_collect_manual_parameters_mocked_input(self):
        """Test manual parameter collection with mocked user input"""
        with patch("ui.cli.Prompt.ask") as mock_prompt, patch(
            "ui.cli.FloatPrompt.ask"
        ) as mock_float_prompt, patch("ui.cli.IntPrompt.ask") as mock_int_prompt, patch(
            "ui.cli.Confirm.ask"
        ) as mock_confirm:

            # Mock user inputs
            mock_float_prompt.side_effect = [
                48000.0,
                52000.0,
                50.0,
            ]  # bounds, investment
            mock_int_prompt.return_value = 20  # grid count
            mock_confirm.return_value = True  # confirm parameters

            params = collect_manual_parameters("BTCUSDC")

            assert params.symbol == "BTCUSDC"
            assert params.lower_bound == Decimal("48000.0")
            assert params.upper_bound == Decimal("52000.0")
            assert params.grid_count == 20
            assert params.investment_per_grid == Decimal("50.0")
            assert params.mode == TradingMode.MANUAL

    def test_confirm_ai_parameters_accept(self):
        """Test confirming AI parameters without changes"""
        suggestion = GridParameters(
            symbol="BTCUSDC",
            lower_bound=Decimal("48000"),
            upper_bound=Decimal("52000"),
            grid_count=20,
            investment_per_grid=Decimal("50"),
            mode=TradingMode.AI,
            confidence=0.85,
        )

        with patch("ui.cli.Confirm.ask", return_value=True):
            result = confirm_ai_parameters(suggestion)

            assert result == suggestion

    def test_confirm_ai_parameters_modify(self):
        """Test modifying AI parameters after confirmation"""
        suggestion = GridParameters(
            symbol="BTCUSDC",
            lower_bound=Decimal("48000"),
            upper_bound=Decimal("52000"),
            grid_count=20,
            investment_per_grid=Decimal("50"),
            mode=TradingMode.AI,
            confidence=0.85,
        )

        with patch("ui.cli.Confirm.ask", return_value=False), patch(
            "ui.cli.collect_manual_parameters"
        ) as mock_collect:

            modified_params = GridParameters(
                symbol="BTCUSDC",
                lower_bound=Decimal("47000"),
                upper_bound=Decimal("53000"),
                grid_count=25,
                investment_per_grid=Decimal("40"),
                mode=TradingMode.MANUAL,
            )
            mock_collect.return_value = modified_params

            result = confirm_ai_parameters(suggestion)

            assert result == modified_params


class TestDisplayFunctions:
    """Test display and UI functions"""

    @patch("ui.cli.console")
    def test_show_static_status(self, mock_console):
        """Test displaying static session status"""
        params = GridParameters(
            "BTCUSDC", Decimal("48000"), Decimal("52000"), 20, Decimal("50")
        )
        session = TradingSession("test-1", "BTCUSDC", params)
        session.status = TradingSessionStatus.RUNNING
        session.start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        session.trades_count = 5
        session.profit_loss = Decimal("125.50")

        show_static_status(session)

        # Verify console.print was called (basic smoke test)
        assert mock_console.print.called

    @patch("ui.cli.Live")
    def test_show_live_status(self, mock_live):
        """Test displaying live session status"""
        params = GridParameters(
            "BTCUSDC", Decimal("48000"), Decimal("52000"), 20, Decimal("50")
        )
        session = TradingSession("test-1", "BTCUSDC", params)
        session.status = TradingSessionStatus.RUNNING

        # Mock the Live context manager
        mock_live_instance = MagicMock()
        mock_live.return_value.__enter__.return_value = mock_live_instance

        show_live_status(session)

        # Verify Live was instantiated and used
        mock_live.assert_called_once()


class TestIntegrationScenarios:
    """Test complex integration scenarios"""

    def test_session_lifecycle_complete(self):
        """Test complete session lifecycle from creation to completion"""
        manager = SessionManager()
        params = GridParameters(
            "BTCUSDC", Decimal("48000"), Decimal("52000"), 20, Decimal("50")
        )

        # Create session
        session_id = manager.create_session("BTCUSDC", params, timedelta(hours=1))
        session = manager.get_session(session_id)

        # Start session
        session.status = TradingSessionStatus.STARTING
        session.start_time = datetime.now(timezone.utc)
        assert session.is_active()

        # Run session
        session.status = TradingSessionStatus.RUNNING
        assert session.is_active()

        # Pause session
        session.status = TradingSessionStatus.PAUSED
        session.pause_time = datetime.now(timezone.utc)
        assert not session.is_active()

        # Resume session
        pause_duration = datetime.now(timezone.utc) - session.pause_time
        session.total_paused_duration += pause_duration
        session.pause_time = None
        session.status = TradingSessionStatus.RUNNING
        assert session.is_active()

        # Complete session
        session.status = TradingSessionStatus.COMPLETED
        session.end_time = datetime.now(timezone.utc)
        assert not session.is_active()

    def test_grid_parameters_validation_edge_cases(self):
        """Test edge cases in grid parameter validation"""
        # Test with very small price ranges
        params = GridParameters(
            symbol="BTCUSDC",
            lower_bound=Decimal("49999.99"),
            upper_bound=Decimal("50000.01"),
            grid_count=2,
            investment_per_grid=Decimal("0.01"),
        )

        config = params.to_grid_config()
        assert config.grid_spacing == Decimal("0.01")  # (50000.01-49999.99)/2

        # Test with large grid count
        params = GridParameters(
            symbol="BTCUSDC",
            lower_bound=Decimal("40000"),
            upper_bound=Decimal("60000"),
            grid_count=1000,
            investment_per_grid=Decimal("1"),
        )

        config = params.to_grid_config()
        assert config.grid_spacing == Decimal("20")  # (60000-40000)/1000
        assert config.num_levels_up == 500
        assert config.num_levels_down == 500

    def test_duration_parsing_edge_cases(self):
        """Test edge cases in duration parsing"""
        # Test zero durations
        duration = parse_duration("0h")
        assert duration == timedelta()

        duration = parse_duration("0m")
        assert duration == timedelta()

        # Test large durations
        duration = parse_duration("9999h")
        assert duration == timedelta(hours=9999)

        # Test format variations
        duration = parse_duration("1H")  # Case insensitive
        assert duration == timedelta(hours=1)


if __name__ == "__main__":
    pytest.main([__file__])
