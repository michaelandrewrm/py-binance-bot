"""
Test Suite for UI Module - Dashboard Component (dashboard.py)

This module provides comprehensive unit tests for the dashboard interface including:
- DashboardData class and its methods
- SimpleDashboard display functionality
- WebDashboard placeholder functionality
- Streamlit dashboard creation
- Data aggregation and calculation methods
- Chart data generation and formatting
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from io import StringIO
import sys
from typing import List, Dict, Optional

# Import the modules under test
from ui.dashboard import (
    DashboardData,
    SimpleDashboard,
    WebDashboard,
    create_streamlit_dashboard,
    quick_status_check,
    run_simple_dashboard,
    run_streamlit_dashboard,
)

# Mock data structures for testing
from data.schema import TradeRecord


class TestDashboardData:
    """Test DashboardData class functionality"""

    def setup_method(self):
        """Setup mock repository for each test"""
        self.mock_repo = Mock()
        self.dashboard_data = DashboardData(repository=self.mock_repo)

    @pytest.mark.asyncio
    async def test_get_portfolio_summary_no_trades(self):
        """Test portfolio summary when no trades exist"""
        self.mock_repo.load_trades.return_value = []

        summary = await self.dashboard_data.get_portfolio_summary()

        expected = {
            "total_value": 0,
            "pnl_24h": 0,
            "pnl_7d": 0,
            "pnl_30d": 0,
            "total_trades": 0,
            "win_rate": 0,
        }
        assert summary == expected
        self.mock_repo.load_trades.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_portfolio_summary_with_trades(self):
        """Test portfolio summary with trade data"""
        now = datetime.now(timezone.utc)

        # Create mock trades
        mock_trades = [
            Mock(pnl=Decimal("50.0"), timestamp=now - timedelta(hours=1)),  # Within 24h
            Mock(
                pnl=Decimal("-20.0"),
                timestamp=now - timedelta(days=2),  # Within 7d but not 24h
            ),
            Mock(
                pnl=Decimal("100.0"),
                timestamp=now - timedelta(days=10),  # Within 30d but not 7d
            ),
            Mock(
                pnl=Decimal("75.0"), timestamp=now - timedelta(hours=12)  # Within 24h
            ),
        ]

        self.mock_repo.load_trades.return_value = mock_trades

        summary = await self.dashboard_data.get_portfolio_summary()

        assert summary["total_value"] == 10205.0  # 10000 + 205 total PnL
        assert summary["pnl_24h"] == 125.0  # 50 + 75
        assert summary["pnl_7d"] == 105.0  # 50 + 75 - 20
        assert summary["pnl_30d"] == 205.0  # 50 + 75 - 20 + 100
        assert summary["total_trades"] == 4
        assert summary["win_rate"] == 0.75  # 3 winning trades out of 4

    @pytest.mark.asyncio
    async def test_get_portfolio_summary_all_losses(self):
        """Test portfolio summary with only losing trades"""
        now = datetime.now(timezone.utc)

        mock_trades = [
            Mock(pnl=Decimal("-50.0"), timestamp=now - timedelta(hours=1)),
            Mock(pnl=Decimal("-30.0"), timestamp=now - timedelta(hours=2)),
        ]

        self.mock_repo.load_trades.return_value = mock_trades

        summary = await self.dashboard_data.get_portfolio_summary()

        assert summary["win_rate"] == 0.0
        assert summary["pnl_24h"] == -80.0
        assert summary["total_value"] == 9920.0  # 10000 - 80

    @pytest.mark.asyncio
    async def test_get_portfolio_summary_none_pnl(self):
        """Test portfolio summary with None PnL values"""
        now = datetime.now(timezone.utc)

        mock_trades = [
            Mock(pnl=None, timestamp=now - timedelta(hours=1)),
            Mock(pnl=Decimal("50.0"), timestamp=now - timedelta(hours=2)),
        ]

        self.mock_repo.load_trades.return_value = mock_trades

        summary = await self.dashboard_data.get_portfolio_summary()

        assert summary["pnl_24h"] == 50.0  # None treated as 0
        assert summary["win_rate"] == 0.5  # 1 winning trade out of 2

    @pytest.mark.asyncio
    async def test_get_recent_trades_success(self):
        """Test getting recent trades with valid data"""
        now = datetime.now(timezone.utc)

        mock_trades = [
            Mock(
                timestamp=now - timedelta(hours=1),
                symbol="BTCUSDC",
                side="BUY",
                quantity=Decimal("0.001"),
                price=Decimal("50000.0"),
                pnl=Decimal("25.0"),
                strategy_name="Grid",
            ),
            Mock(
                timestamp=now - timedelta(hours=2),
                symbol="ETHUSDC",
                side="SELL",
                quantity=Decimal("0.1"),
                price=Decimal("3000.0"),
                pnl=Decimal("-10.0"),
                strategy_name=None,
            ),
        ]

        self.mock_repo.load_trades.return_value = mock_trades

        trades = await self.dashboard_data.get_recent_trades(10)

        assert len(trades) == 2
        assert trades[0]["symbol"] == "BTCUSDC"
        assert trades[0]["side"] == "BUY"
        assert trades[0]["quantity"] == 0.001
        assert trades[0]["price"] == 50000.0
        assert trades[0]["pnl"] == 25.0
        assert trades[0]["strategy"] == "Grid"

        assert trades[1]["symbol"] == "ETHUSDC"
        assert trades[1]["strategy"] == "Unknown"  # None becomes "Unknown"

    @pytest.mark.asyncio
    async def test_get_recent_trades_limit(self):
        """Test recent trades respects limit parameter"""
        now = datetime.now(timezone.utc)

        # Create more trades than limit
        mock_trades = [
            Mock(
                timestamp=now - timedelta(hours=i),
                symbol="BTCUSDC",
                side="BUY",
                quantity=Decimal("0.001"),
                price=Decimal("50000.0"),
                pnl=Decimal("25.0"),
                strategy_name="Grid",
            )
            for i in range(10)
        ]

        self.mock_repo.load_trades.return_value = mock_trades

        trades = await self.dashboard_data.get_recent_trades(5)

        assert len(trades) == 5
        # Should be sorted by timestamp descending (most recent first)
        for i in range(4):
            current_time = datetime.fromisoformat(trades[i]["timestamp"])
            next_time = datetime.fromisoformat(trades[i + 1]["timestamp"])
            assert current_time > next_time

    @pytest.mark.asyncio
    async def test_get_performance_chart_data_no_trades(self):
        """Test performance chart data with no trades"""
        self.mock_repo.load_trades.return_value = []

        chart_data = await self.dashboard_data.get_performance_chart_data(7)

        assert chart_data["dates"] == []
        assert chart_data["values"] == []
        assert chart_data["pnl"] == []

    @pytest.mark.asyncio
    async def test_get_performance_chart_data_with_trades(self):
        """Test performance chart data generation with trades"""
        base_date = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        mock_trades = [
            Mock(timestamp=base_date - timedelta(days=2), pnl=Decimal("50.0")),
            Mock(timestamp=base_date - timedelta(days=1), pnl=Decimal("30.0")),
            Mock(timestamp=base_date, pnl=Decimal("-20.0")),
        ]

        self.mock_repo.load_trades.return_value = mock_trades

        chart_data = await self.dashboard_data.get_performance_chart_data(3)

        assert len(chart_data["dates"]) == 4  # 3 days + today
        assert len(chart_data["values"]) == 4
        assert len(chart_data["pnl"]) == 4

        # Check cumulative PnL calculation
        assert chart_data["pnl"][1] == 50.0  # First trade
        assert chart_data["pnl"][2] == 80.0  # 50 + 30
        assert chart_data["pnl"][3] == 60.0  # 50 + 30 - 20

        # Check portfolio values
        assert chart_data["values"][0] == 10000  # Initial value
        assert chart_data["values"][3] == 10060  # 10000 + 60 PnL

    @pytest.mark.asyncio
    async def test_get_strategy_performance_success(self):
        """Test strategy performance breakdown"""
        now = datetime.now(timezone.utc)

        mock_trades = [
            Mock(
                strategy_name="Grid",
                pnl=Decimal("50.0"),
                quantity=Decimal("0.001"),
                price=Decimal("50000.0"),
            ),
            Mock(
                strategy_name="Grid",
                pnl=Decimal("-20.0"),
                quantity=Decimal("0.002"),
                price=Decimal("45000.0"),
            ),
            Mock(
                strategy_name="Mean Reversion",
                pnl=Decimal("100.0"),
                quantity=Decimal("0.005"),
                price=Decimal("40000.0"),
            ),
            Mock(
                strategy_name=None,  # Unknown strategy
                pnl=Decimal("25.0"),
                quantity=Decimal("0.001"),
                price=Decimal("30000.0"),
            ),
        ]

        self.mock_repo.load_trades.return_value = mock_trades

        performance = await self.dashboard_data.get_strategy_performance()

        assert len(performance) == 3  # Grid, Mean Reversion, Unknown

        # Find Grid strategy performance
        grid_perf = next(p for p in performance if p["strategy"] == "Grid")
        assert grid_perf["total_trades"] == 2
        assert grid_perf["winning_trades"] == 1
        assert grid_perf["win_rate"] == 0.5
        assert grid_perf["total_pnl"] == 30.0  # 50 - 20
        assert grid_perf["avg_pnl"] == 15.0  # 30 / 2
        assert grid_perf["volume"] == 140.0  # (0.001*50000) + (0.002*45000)

        # Should be sorted by total_pnl descending
        assert performance[0]["total_pnl"] >= performance[1]["total_pnl"]

    @pytest.mark.asyncio
    async def test_get_market_data_status_success(self):
        """Test market data status retrieval"""
        # Mock database connection and cursor
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            {
                "symbol": "BTCUSDC",
                "interval": "5m",
                "count": 1000,
                "start_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "end_date": datetime.now(timezone.utc) - timedelta(hours=1),
                "last_updated": datetime.now(timezone.utc),
            },
            {
                "symbol": "ETHUSDC",
                "interval": "1h",
                "count": 500,
                "start_date": datetime(2024, 1, 15, tzinfo=timezone.utc),
                "end_date": None,  # Test None handling
                "last_updated": None,
            },
        ]

        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=None)

        self.mock_repo.db_repo.get_connection.return_value = mock_conn

        status = await self.dashboard_data.get_market_data_status()

        assert len(status) == 2

        btc_status = status[0]
        assert btc_status["symbol"] == "BTCUSDC"
        assert btc_status["interval"] == "5m"
        assert btc_status["record_count"] == 1000
        assert btc_status["data_age_hours"] == pytest.approx(1.0, abs=0.1)

        eth_status = status[1]
        assert eth_status["symbol"] == "ETHUSDC"
        assert eth_status["end_date"] is None
        assert eth_status["data_age_hours"] is None


class TestSimpleDashboard:
    """Test SimpleDashboard class functionality"""

    def setup_method(self):
        """Setup SimpleDashboard with mocked data provider"""
        with patch("ui.dashboard.DashboardData") as mock_data_class:
            self.mock_data_provider = Mock()
            mock_data_class.return_value = self.mock_data_provider
            self.dashboard = SimpleDashboard()

    @pytest.mark.asyncio
    async def test_display_summary(self):
        """Test portfolio summary display"""
        self.mock_data_provider.get_portfolio_summary.return_value = {
            "total_value": 10250.0,
            "pnl_24h": 125.0,
            "pnl_7d": 250.0,
            "pnl_30d": 250.0,
            "total_trades": 15,
            "win_rate": 0.67,
        }

        # Capture printed output
        with patch("builtins.print") as mock_print:
            await self.dashboard.display_summary()

            # Verify some key information was printed
            mock_print.assert_called()
            printed_text = " ".join(
                str(call.args[0]) for call in mock_print.call_args_list
            )
            assert "TRADING BOT DASHBOARD" in printed_text
            assert "$10,250.00" in printed_text
            assert "+$125.00" in printed_text
            assert "67.0%" in printed_text

    @pytest.mark.asyncio
    async def test_display_recent_trades(self):
        """Test recent trades display"""
        self.mock_data_provider.get_recent_trades.return_value = [
            {
                "timestamp": "2024-01-01T10:00:00+00:00",
                "symbol": "BTCUSDC",
                "side": "BUY",
                "quantity": 0.001,
                "price": 50000.0,
                "pnl": 25.0,
                "strategy": "Grid",
            },
            {
                "timestamp": "2024-01-01T09:00:00+00:00",
                "symbol": "ETHUSDC",
                "side": "SELL",
                "quantity": 0.1,
                "price": 3000.0,
                "pnl": -10.0,
                "strategy": "Mean Reversion",
            },
        ]

        with patch("builtins.print") as mock_print:
            await self.dashboard.display_recent_trades(5)

            mock_print.assert_called()
            printed_text = " ".join(
                str(call.args[0]) for call in mock_print.call_args_list
            )
            assert "RECENT TRADES" in printed_text
            assert "BTCUSDC" in printed_text
            assert "ETHUSDC" in printed_text

    @pytest.mark.asyncio
    async def test_display_strategy_performance(self):
        """Test strategy performance display"""
        self.mock_data_provider.get_strategy_performance.return_value = [
            {
                "strategy": "Grid",
                "total_trades": 10,
                "win_rate": 0.7,
                "total_pnl": 150.0,
                "avg_pnl": 15.0,
                "volume": 1000.0,
            },
            {
                "strategy": "Mean Reversion",
                "total_trades": 5,
                "win_rate": 0.6,
                "total_pnl": 75.0,
                "avg_pnl": 15.0,
                "volume": 500.0,
            },
        ]

        with patch("builtins.print") as mock_print:
            await self.dashboard.display_strategy_performance()

            mock_print.assert_called()
            printed_text = " ".join(
                str(call.args[0]) for call in mock_print.call_args_list
            )
            assert "STRATEGY PERFORMANCE" in printed_text
            assert "Grid" in printed_text
            assert "Mean Reversion" in printed_text

    @pytest.mark.asyncio
    async def test_display_data_status(self):
        """Test market data status display"""
        self.mock_data_provider.get_market_data_status.return_value = [
            {
                "symbol": "BTCUSDC",
                "interval": "5m",
                "record_count": 1000,
                "start_date": "2024-01-01T00:00:00+00:00",
                "end_date": "2024-01-01T10:00:00+00:00",
                "last_updated": "2024-01-01T10:00:00+00:00",
                "data_age_hours": 1.0,
            }
        ]

        with patch("builtins.print") as mock_print:
            await self.dashboard.display_data_status()

            mock_print.assert_called()
            printed_text = " ".join(
                str(call.args[0]) for call in mock_print.call_args_list
            )
            assert "MARKET DATA STATUS" in printed_text
            assert "BTCUSDC" in printed_text

    @pytest.mark.asyncio
    async def test_run_full_dashboard(self):
        """Test running the complete dashboard"""
        # Mock all data provider methods
        self.mock_data_provider.get_portfolio_summary.return_value = {
            "total_value": 10000,
            "pnl_24h": 0,
            "pnl_7d": 0,
            "pnl_30d": 0,
            "total_trades": 0,
            "win_rate": 0,
        }
        self.mock_data_provider.get_recent_trades.return_value = []
        self.mock_data_provider.get_strategy_performance.return_value = []
        self.mock_data_provider.get_market_data_status.return_value = []

        with patch("builtins.print"):
            await self.dashboard.run_full_dashboard()

            # Verify all methods were called
            self.mock_data_provider.get_portfolio_summary.assert_called_once()
            self.mock_data_provider.get_recent_trades.assert_called_once()
            self.mock_data_provider.get_strategy_performance.assert_called_once()
            self.mock_data_provider.get_market_data_status.assert_called_once()


class TestWebDashboard:
    """Test WebDashboard class functionality"""

    def test_web_dashboard_initialization(self):
        """Test WebDashboard initialization with default and custom parameters"""
        # Default initialization
        dashboard = WebDashboard()
        assert dashboard.host == "localhost"
        assert dashboard.port == 8000
        assert dashboard.data_provider is not None

        # Custom initialization
        dashboard = WebDashboard("0.0.0.0", 3000)
        assert dashboard.host == "0.0.0.0"
        assert dashboard.port == 3000

    @pytest.mark.asyncio
    async def test_start_server_placeholder(self):
        """Test that start_server method is a placeholder"""
        dashboard = WebDashboard()

        with patch("builtins.print") as mock_print:
            await dashboard.start_server()

            mock_print.assert_called()
            printed_text = " ".join(
                str(call.args[0]) for call in mock_print.call_args_list
            )
            assert "Web dashboard would start at http://localhost:8000" in printed_text
            assert "not yet implemented" in printed_text


class TestStreamlitDashboard:
    """Test Streamlit dashboard functionality"""

    def test_create_streamlit_dashboard_success(self):
        """Test successful Streamlit dashboard creation"""
        with patch("streamlit.set_page_config") as mock_config, patch(
            "streamlit.title"
        ) as mock_title, patch("streamlit.sidebar") as mock_sidebar, patch(
            "streamlit.columns"
        ) as mock_columns, patch(
            "streamlit.metric"
        ) as mock_metric, patch(
            "streamlit.subheader"
        ) as mock_subheader, patch(
            "streamlit.plotly_chart"
        ) as mock_chart, patch(
            "streamlit.dataframe"
        ) as mock_dataframe, patch(
            "pandas.DataFrame"
        ) as mock_df, patch(
            "plotly.graph_objects.Figure"
        ) as mock_fig:

            # Mock column layout
            mock_columns.return_value = [Mock(), Mock(), Mock(), Mock()]

            result = create_streamlit_dashboard()

            assert result is True
            mock_config.assert_called_once()
            mock_title.assert_called_once_with("🤖 Trading Bot Dashboard")

    def test_create_streamlit_dashboard_import_error(self):
        """Test Streamlit dashboard creation when imports fail"""
        with patch("builtins.print") as mock_print:
            # Patch the import to raise ImportError
            with patch.dict("sys.modules", {"streamlit": None}):
                result = create_streamlit_dashboard()

                assert result is False
                mock_print.assert_called()
                printed_text = str(mock_print.call_args[0][0])
                assert "Streamlit not available" in printed_text


class TestUtilityFunctions:
    """Test utility functions in dashboard module"""

    @pytest.mark.asyncio
    async def test_quick_status_check(self):
        """Test quick status check function"""
        with patch("ui.dashboard.SimpleDashboard") as mock_dashboard_class:
            mock_dashboard = Mock()
            mock_dashboard.display_summary = AsyncMock()
            mock_dashboard_class.return_value = mock_dashboard

            await quick_status_check()

            mock_dashboard.display_summary.assert_called_once()

    def test_run_simple_dashboard(self):
        """Test running simple dashboard"""
        with patch("ui.dashboard.SimpleDashboard") as mock_dashboard_class, patch(
            "asyncio.run"
        ) as mock_run:

            mock_dashboard = Mock()
            mock_dashboard.run_full_dashboard = AsyncMock()
            mock_dashboard_class.return_value = mock_dashboard

            run_simple_dashboard()

            mock_run.assert_called_once()

    def test_run_streamlit_dashboard_success(self):
        """Test running Streamlit dashboard when available"""
        with patch(
            "ui.dashboard.create_streamlit_dashboard", return_value=True
        ) as mock_create, patch("builtins.print") as mock_print:

            run_streamlit_dashboard()

            mock_create.assert_called_once()
            printed_args = [str(call.args[0]) for call in mock_print.call_args_list]
            assert any("Streamlit dashboard created" in arg for arg in printed_args)

    def test_run_streamlit_dashboard_fallback(self):
        """Test running Streamlit dashboard falls back to simple dashboard"""
        with patch(
            "ui.dashboard.create_streamlit_dashboard", return_value=False
        ) as mock_create, patch(
            "ui.dashboard.run_simple_dashboard"
        ) as mock_run_simple, patch(
            "builtins.print"
        ) as mock_print:

            run_streamlit_dashboard()

            mock_create.assert_called_once()
            mock_run_simple.assert_called_once()
            printed_args = [str(call.args[0]) for call in mock_print.call_args_list]
            assert any("Streamlit not available" in arg for arg in printed_args)


class TestErrorHandling:
    """Test error handling in dashboard components"""

    @pytest.mark.asyncio
    async def test_dashboard_data_repository_error(self):
        """Test DashboardData handling repository errors"""
        mock_repo = Mock()
        mock_repo.load_trades.side_effect = Exception("Database error")

        dashboard_data = DashboardData(repository=mock_repo)

        # Should handle database errors gracefully
        with pytest.raises(Exception, match="Database error"):
            await dashboard_data.get_portfolio_summary()

    @pytest.mark.asyncio
    async def test_dashboard_data_calculation_edge_cases(self):
        """Test edge cases in dashboard calculations"""
        mock_repo = Mock()
        dashboard_data = DashboardData(repository=mock_repo)

        # Test with trades having extreme values
        extreme_trades = [
            Mock(pnl=Decimal("999999999.99"), timestamp=datetime.now(timezone.utc)),
            Mock(pnl=Decimal("-999999999.99"), timestamp=datetime.now(timezone.utc)),
        ]

        mock_repo.load_trades.return_value = extreme_trades

        summary = await dashboard_data.get_portfolio_summary()

        assert summary["total_trades"] == 2
        assert summary["win_rate"] == 0.5
        assert summary["pnl_30d"] == 0.0  # Net zero

    def test_format_edge_cases(self):
        """Test formatting edge cases in dashboard displays"""
        # Test with very small and very large numbers
        dashboard = SimpleDashboard()

        # This is more of a smoke test to ensure no crashes occur
        # with extreme values in the display methods
        assert dashboard is not None


class TestIntegrationScenarios:
    """Test complex integration scenarios"""

    @pytest.mark.asyncio
    async def test_complete_dashboard_workflow(self):
        """Test complete dashboard data flow from repository to display"""
        # Create realistic test data
        now = datetime.now(timezone.utc)
        mock_trades = [
            Mock(
                timestamp=now - timedelta(hours=1),
                symbol="BTCUSDC",
                side="BUY",
                quantity=Decimal("0.001"),
                price=Decimal("50000.0"),
                pnl=Decimal("25.0"),
                strategy_name="Grid",
            ),
            Mock(
                timestamp=now - timedelta(hours=3),
                symbol="ETHUSDC",
                side="SELL",
                quantity=Decimal("0.1"),
                price=Decimal("3000.0"),
                pnl=Decimal("-10.0"),
                strategy_name="Mean Reversion",
            ),
        ]

        mock_repo = Mock()
        mock_repo.load_trades.return_value = mock_trades

        # Mock database queries for market data status
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=None)
        mock_repo.db_repo.get_connection.return_value = mock_conn

        # Test data provider
        dashboard_data = DashboardData(repository=mock_repo)

        # Test all major data aggregation methods
        summary = await dashboard_data.get_portfolio_summary()
        recent_trades = await dashboard_data.get_recent_trades(10)
        performance = await dashboard_data.get_strategy_performance()
        chart_data = await dashboard_data.get_performance_chart_data(7)
        market_status = await dashboard_data.get_market_data_status()

        # Verify consistent data across methods
        assert summary["total_trades"] == len(recent_trades)
        assert len(performance) >= 1  # Should have at least one strategy
        assert isinstance(chart_data, dict)
        assert isinstance(market_status, list)

    @pytest.mark.asyncio
    async def test_dashboard_with_large_dataset(self):
        """Test dashboard performance with large datasets"""
        # Simulate large number of trades
        now = datetime.now(timezone.utc)
        large_trade_set = [
            Mock(
                timestamp=now - timedelta(hours=i),
                symbol=f"SYM{i % 10}USDC",
                side="BUY" if i % 2 == 0 else "SELL",
                quantity=Decimal(str(0.001 * (i + 1))),
                price=Decimal(str(1000.0 + i)),
                pnl=Decimal(str((i % 10) - 5)),  # Mix of positive and negative
                strategy_name=f"Strategy{i % 3}",
            )
            for i in range(1000)  # 1000 trades
        ]

        mock_repo = Mock()
        mock_repo.load_trades.return_value = large_trade_set

        dashboard_data = DashboardData(repository=mock_repo)

        # Test that operations complete without errors
        summary = await dashboard_data.get_portfolio_summary()
        recent_trades = await dashboard_data.get_recent_trades(50)
        performance = await dashboard_data.get_strategy_performance()

        assert summary["total_trades"] == 1000
        assert len(recent_trades) == 50  # Respects limit
        assert len(performance) == 3  # Three different strategies


if __name__ == "__main__":
    pytest.main([__file__])
