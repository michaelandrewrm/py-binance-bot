"""
Dashboard Module - Web-based Monitoring Interface

This module provides a web-based dashboard for monitoring the trading bot's
performance, viewing real-time data, and managing trading operations.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
import json
import logging
from decimal import Decimal

# Note: This is a basic structure. For a full implementation, you would use
# a web framework like FastAPI, Streamlit, or Dash.

logger = logging.getLogger(__name__)


class DashboardData:
    """
    Manages data aggregation for the dashboard
    """

    def __init__(self, repository=None):
        from storage.repo import get_default_repository

        self.repo = repository or get_default_repository()

    async def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        # Load recent trades
        recent_trades = self.repo.load_trades(
            start_time=datetime.now(timezone.utc) - timedelta(days=30)
        )

        if not recent_trades:
            return {
                "total_value": 0,
                "pnl_24h": 0,
                "pnl_7d": 0,
                "pnl_30d": 0,
                "total_trades": 0,
                "win_rate": 0,
            }

        # Calculate metrics
        total_pnl = sum(float(trade.pnl or 0) for trade in recent_trades)
        winning_trades = sum(1 for trade in recent_trades if (trade.pnl or 0) > 0)
        win_rate = winning_trades / len(recent_trades) if recent_trades else 0

        # Time-based PnL
        now = datetime.now(timezone.utc)
        pnl_24h = sum(
            float(trade.pnl or 0)
            for trade in recent_trades
            if trade.timestamp >= now - timedelta(hours=24)
        )
        pnl_7d = sum(
            float(trade.pnl or 0)
            for trade in recent_trades
            if trade.timestamp >= now - timedelta(days=7)
        )

        return {
            "total_value": 10000 + total_pnl,  # Assuming 10k initial
            "pnl_24h": pnl_24h,
            "pnl_7d": pnl_7d,
            "pnl_30d": total_pnl,
            "total_trades": len(recent_trades),
            "win_rate": win_rate,
        }

    async def get_recent_trades(self, limit: int = 50) -> List[Dict]:
        """Get recent trades for display"""
        trades = self.repo.load_trades(
            start_time=datetime.now(timezone.utc) - timedelta(days=7)
        )

        # Sort by timestamp and limit
        trades.sort(key=lambda x: x.timestamp, reverse=True)
        trades = trades[:limit]

        return [
            {
                "timestamp": trade.timestamp.isoformat(),
                "symbol": trade.symbol,
                "side": trade.side,
                "quantity": float(trade.quantity),
                "price": float(trade.price),
                "pnl": float(trade.pnl or 0),
                "strategy": trade.strategy_name or "Unknown",
            }
            for trade in trades
        ]

    async def get_performance_chart_data(self, days: int = 30) -> Dict:
        """Get data for performance charts"""
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        trades = self.repo.load_trades(start_time=start_date)

        if not trades:
            return {"dates": [], "values": [], "pnl": []}

        # Group trades by day and calculate cumulative PnL
        daily_pnl = {}

        for trade in trades:
            date_key = trade.timestamp.date().isoformat()
            if date_key not in daily_pnl:
                daily_pnl[date_key] = 0
            daily_pnl[date_key] += float(trade.pnl or 0)

        # Generate time series
        dates = []
        cumulative_pnl = []
        portfolio_values = []

        current_date = start_date.date()
        end_date = datetime.now(timezone.utc).date()
        running_pnl = 0
        initial_value = 10000  # Assumed initial portfolio value

        while current_date <= end_date:
            date_str = current_date.isoformat()
            dates.append(date_str)

            day_pnl = daily_pnl.get(date_str, 0)
            running_pnl += day_pnl

            cumulative_pnl.append(running_pnl)
            portfolio_values.append(initial_value + running_pnl)

            current_date += timedelta(days=1)

        return {"dates": dates, "values": portfolio_values, "pnl": cumulative_pnl}

    async def get_strategy_performance(self) -> List[Dict]:
        """Get performance breakdown by strategy"""
        trades = self.repo.load_trades(
            start_time=datetime.now(timezone.utc) - timedelta(days=30)
        )

        strategy_stats = {}

        for trade in trades:
            strategy = trade.strategy_name or "Unknown"
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "total_pnl": 0,
                    "volume": 0,
                }

            stats = strategy_stats[strategy]
            stats["total_trades"] += 1
            if (trade.pnl or 0) > 0:
                stats["winning_trades"] += 1
            stats["total_pnl"] += float(trade.pnl or 0)
            stats["volume"] += float(trade.quantity) * float(trade.price)

        # Calculate metrics
        result = []
        for strategy, stats in strategy_stats.items():
            win_rate = (
                stats["winning_trades"] / stats["total_trades"]
                if stats["total_trades"] > 0
                else 0
            )
            avg_pnl = (
                stats["total_pnl"] / stats["total_trades"]
                if stats["total_trades"] > 0
                else 0
            )

            result.append(
                {
                    "strategy": strategy,
                    "total_trades": stats["total_trades"],
                    "win_rate": win_rate,
                    "total_pnl": stats["total_pnl"],
                    "avg_pnl": avg_pnl,
                    "volume": stats["volume"],
                }
            )

        return sorted(result, key=lambda x: x["total_pnl"], reverse=True)

    async def get_market_data_status(self) -> List[Dict]:
        """Get status of available market data"""
        with self.repo.db_repo.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT symbol, interval, COUNT(*) as count,
                       MIN(open_time) as start_date,
                       MAX(open_time) as end_date,
                       MAX(created_at) as last_updated
                FROM klines
                GROUP BY symbol, interval
                ORDER BY symbol, interval
            """
            )

            rows = cursor.fetchall()

        return [
            {
                "symbol": row["symbol"],
                "interval": row["interval"],
                "record_count": row["count"],
                "start_date": (
                    row["start_date"].isoformat() if row["start_date"] else None
                ),
                "end_date": row["end_date"].isoformat() if row["end_date"] else None,
                "last_updated": (
                    row["last_updated"].isoformat() if row["last_updated"] else None
                ),
                "data_age_hours": (
                    (datetime.now(timezone.utc) - row["end_date"]).total_seconds()
                    / 3600
                    if row["end_date"]
                    else None
                ),
            }
            for row in rows
        ]


class SimpleDashboard:
    """
    Simple text-based dashboard for console output
    """

    def __init__(self):
        self.data_provider = DashboardData()

    async def display_summary(self):
        """Display portfolio summary"""
        summary = await self.data_provider.get_portfolio_summary()

        print("\n" + "=" * 50)
        print("           TRADING BOT DASHBOARD")
        print("=" * 50)
        print(f"Portfolio Value: ${summary['total_value']:,.2f}")
        print(f"24h P&L:        ${summary['pnl_24h']:+,.2f}")
        print(f"7d P&L:         ${summary['pnl_7d']:+,.2f}")
        print(f"30d P&L:        ${summary['pnl_30d']:+,.2f}")
        print(f"Total Trades:   {summary['total_trades']}")
        print(f"Win Rate:       {summary['win_rate']:.1%}")
        print("=" * 50)

    async def display_recent_trades(self, limit: int = 10):
        """Display recent trades"""
        trades = await self.data_provider.get_recent_trades(limit)

        print(f"\nRECENT TRADES (Last {limit})")
        print("-" * 80)
        print(
            f"{'Time':<20} {'Symbol':<10} {'Side':<5} {'Qty':<12} {'Price':<12} {'P&L':<10}"
        )
        print("-" * 80)

        for trade in trades:
            timestamp = datetime.fromisoformat(trade["timestamp"]).strftime(
                "%m-%d %H:%M"
            )
            pnl_str = f"{trade['pnl']:+.4f}" if trade["pnl"] != 0 else "0.0000"

            print(
                f"{timestamp:<20} {trade['symbol']:<10} {trade['side']:<5} "
                f"{trade['quantity']:<12.4f} {trade['price']:<12.4f} {pnl_str:<10}"
            )

    async def display_strategy_performance(self):
        """Display strategy performance breakdown"""
        strategies = await self.data_provider.get_strategy_performance()

        print(f"\nSTRATEGY PERFORMANCE")
        print("-" * 70)
        print(
            f"{'Strategy':<15} {'Trades':<8} {'Win Rate':<10} {'Total P&L':<12} {'Avg P&L':<10}"
        )
        print("-" * 70)

        for strategy in strategies:
            print(
                f"{strategy['strategy']:<15} {strategy['total_trades']:<8} "
                f"{strategy['win_rate']:<10.1%} {strategy['total_pnl']:<12.4f} "
                f"{strategy['avg_pnl']:<10.4f}"
            )

    async def display_data_status(self):
        """Display market data status"""
        data_status = await self.data_provider.get_market_data_status()

        print(f"\nMARKET DATA STATUS")
        print("-" * 80)
        print(
            f"{'Symbol':<10} {'Interval':<10} {'Records':<10} {'Last Update':<20} {'Age (h)':<8}"
        )
        print("-" * 80)

        for status in data_status:
            last_update = (
                datetime.fromisoformat(status["end_date"]).strftime("%m-%d %H:%M")
                if status["end_date"]
                else "N/A"
            )
            age = (
                f"{status['data_age_hours']:.1f}" if status["data_age_hours"] else "N/A"
            )

            print(
                f"{status['symbol']:<10} {status['interval']:<10} {status['record_count']:<10} "
                f"{last_update:<20} {age:<8}"
            )

    async def run_full_dashboard(self):
        """Run complete dashboard display"""
        await self.display_summary()
        await self.display_recent_trades()
        await self.display_strategy_performance()
        await self.display_data_status()
        print("\n")


# Web Dashboard (Placeholder for future implementation)
class WebDashboard:
    """
    Web-based dashboard using FastAPI (to be implemented)
    """

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.data_provider = DashboardData()

    async def start_server(self):
        """Start web dashboard server"""
        print(f"Web dashboard would start at http://{self.host}:{self.port}")
        print("Web dashboard not yet implemented. Use SimpleDashboard for now.")

        # TODO: Implement FastAPI server with:
        # - Real-time portfolio updates
        # - Interactive charts
        # - Trade management interface
        # - Strategy configuration
        # - Risk monitoring alerts


# Streamlit Dashboard (Alternative implementation)
def create_streamlit_dashboard():
    """
    Create Streamlit dashboard (requires streamlit package)
    """
    try:
        import streamlit as st
        import plotly.graph_objects as go
        import plotly.express as px
        import pandas as pd

        st.set_page_config(
            page_title="Trading Bot Dashboard", page_icon="📈", layout="wide"
        )

        st.title("🤖 Trading Bot Dashboard")

        # Initialize data provider
        data_provider = DashboardData()

        # Sidebar
        st.sidebar.header("Controls")
        refresh_data = st.sidebar.button("Refresh Data")

        # Main content
        col1, col2, col3, col4 = st.columns(4)

        # Portfolio summary (would need to be async in real implementation)
        with col1:
            st.metric("Portfolio Value", "$10,000", "0.0%")

        with col2:
            st.metric("24h P&L", "$0", "0.0%")

        with col3:
            st.metric("Total Trades", "0")

        with col4:
            st.metric("Win Rate", "0%")

        # Charts
        st.subheader("Portfolio Performance")

        # Placeholder chart
        dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
        values = [10000] * 30  # Placeholder data

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=dates, y=values, mode="lines", name="Portfolio Value")
        )
        fig.update_layout(title="Portfolio Value Over Time")
        st.plotly_chart(fig, use_container_width=True)

        # Recent trades table
        st.subheader("Recent Trades")

        # Placeholder data
        trade_data = pd.DataFrame(
            {
                "Timestamp": ["2024-01-01 10:00:00"],
                "Symbol": ["BTCUSDC"],
                "Side": ["BUY"],
                "Quantity": [0.001],
                "Price": [45000.0],
                "P&L": [0.0],
            }
        )

        st.dataframe(trade_data, use_container_width=True)

        return True

    except ImportError:
        print("Streamlit not available. Install with: pip install streamlit plotly")
        return False


# Utility functions
async def quick_status_check():
    """Quick status check for monitoring"""
    dashboard = SimpleDashboard()
    await dashboard.display_summary()


def run_simple_dashboard():
    """Run the simple console dashboard"""
    dashboard = SimpleDashboard()
    asyncio.run(dashboard.run_full_dashboard())


def run_streamlit_dashboard():
    """Run Streamlit dashboard"""
    if create_streamlit_dashboard():
        print("Streamlit dashboard created. Run with: streamlit run dashboard.py")
    else:
        print("Streamlit not available. Using simple dashboard instead.")
        run_simple_dashboard()


if __name__ == "__main__":
    # Default to simple dashboard
    run_simple_dashboard()
