"""
End of Run Reports Module

This module generates comprehensive reports at the end of trading sessions,
backtests, or optimization runs. Reports can be exported as HTML, PDF, or PNG.
"""

import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import json
import logging
from pathlib import Path
import base64
from io import BytesIO

from data.schema import BacktestResult, TradeRecord, PerformanceMetrics
from sim.evaluation import PerformanceEvaluator

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates comprehensive trading reports
    """

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.evaluator = PerformanceEvaluator()

    def generate_backtest_report(
        self,
        result: BacktestResult,
        comparison_results: Optional[List[BacktestResult]] = None,
    ) -> str:
        """Generate comprehensive backtest report"""

        # Generate filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_{result.strategy_name}_{result.symbol}_{timestamp}.html"
        filepath = self.output_dir / filename

        # Generate charts
        equity_chart = self._create_equity_curve_chart(result.equity_curve)
        trades_chart = self._create_trades_chart(result.trades)
        monthly_returns_chart = self._create_monthly_returns_chart(result.equity_curve)

        # Create HTML content
        html_content = self._create_backtest_html(
            result,
            equity_chart,
            trades_chart,
            monthly_returns_chart,
            comparison_results,
        )

        # Save report
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Backtest report saved to {filepath}")
        return str(filepath)

    def generate_optimization_report(
        self, optimization_results: Dict, strategy_name: str, symbol: str
    ) -> str:
        """Generate hyperparameter optimization report"""

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_{strategy_name}_{symbol}_{timestamp}.html"
        filepath = self.output_dir / filename

        # Generate charts
        optimization_chart = self._create_optimization_chart(optimization_results)
        parameter_importance_chart = self._create_parameter_importance_chart(
            optimization_results
        )

        # Create HTML content
        html_content = self._create_optimization_html(
            optimization_results,
            strategy_name,
            symbol,
            optimization_chart,
            parameter_importance_chart,
        )

        # Save report
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Optimization report saved to {filepath}")
        return str(filepath)

    def generate_trading_session_report(
        self,
        trades: List[TradeRecord],
        session_start: datetime,
        session_end: datetime,
    ) -> str:
        """Generate trading session report"""

        if not trades:
            logger.warning("No trades provided for session report")
            return ""

        # Calculate session metrics
        session_metrics = self._calculate_session_metrics(
            trades, session_start, session_end
        )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"session_{session_start.strftime('%Y%m%d')}_{timestamp}.html"
        filepath = self.output_dir / filename

        # Generate charts
        pnl_chart = self._create_session_pnl_chart(trades)
        volume_chart = self._create_session_volume_chart(trades)

        # Create HTML content
        html_content = self._create_session_html(
            trades,
            session_metrics,
            session_start,
            session_end,
            pnl_chart,
            volume_chart,
        )

        # Save report
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Session report saved to {filepath}")
        return str(filepath)

    def _calculate_session_metrics(
        self,
        trades: List[TradeRecord],
        session_start: datetime,
        session_end: datetime,
    ) -> Dict:
        """Calculate metrics for a trading session"""

        # Filter trades to session period
        session_trades = [
            t for t in trades if session_start <= t.timestamp <= session_end
        ]

        if not session_trades:
            return self._empty_session_metrics()

        # Basic calculations
        total_pnl = sum(float(trade.pnl or 0) for trade in session_trades)
        total_volume = sum(
            float(trade.quantity) * float(trade.price) for trade in session_trades
        )
        total_fees = sum(float(trade.commission or 0) for trade in session_trades)

        winning_trades = [t for t in session_trades if (t.pnl or 0) > 0]
        losing_trades = [t for t in session_trades if (t.pnl or 0) < 0]

        # Performance metrics
        win_rate = len(winning_trades) / len(session_trades) if session_trades else 0
        avg_win = (
            sum(float(t.pnl or 0) for t in winning_trades) / len(winning_trades)
            if winning_trades
            else 0
        )
        avg_loss = (
            sum(float(t.pnl or 0) for t in losing_trades) / len(losing_trades)
            if losing_trades
            else 0
        )
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        # Trading frequency
        session_duration = (session_end - session_start).total_seconds() / 3600  # hours
        trades_per_hour = (
            len(session_trades) / session_duration if session_duration > 0 else 0
        )

        return {
            "session_start": session_start,
            "session_end": session_end,
            "session_duration_hours": session_duration,
            "total_trades": len(session_trades),
            "total_pnl": total_pnl,
            "total_volume": total_volume,
            "total_fees": total_fees,
            "net_pnl": total_pnl - total_fees,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "trades_per_hour": trades_per_hour,
            "largest_win": max((float(t.pnl or 0) for t in session_trades), default=0),
            "largest_loss": min((float(t.pnl or 0) for t in session_trades), default=0),
        }

    def _empty_session_metrics(self) -> Dict:
        """Return empty session metrics"""
        return {
            "total_trades": 0,
            "total_pnl": 0,
            "total_volume": 0,
            "total_fees": 0,
            "net_pnl": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "trades_per_hour": 0,
            "largest_win": 0,
            "largest_loss": 0,
        }

    def _create_equity_curve_chart(self, equity_curve) -> str:
        """Create equity curve chart as base64 encoded image"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            fig, ax = plt.subplots(figsize=(12, 6))

            # Extract data
            timestamps = [point.timestamp for point in equity_curve]
            values = [float(point.value) for point in equity_curve]

            # Plot equity curve
            ax.plot(timestamps, values, linewidth=2, color="#2E86AB")
            ax.fill_between(timestamps, values, alpha=0.3, color="#2E86AB")

            # Formatting
            ax.set_title("Portfolio Equity Curve", fontsize=16, fontweight="bold")
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Portfolio Value ($)", fontsize=12)
            ax.grid(True, alpha=0.3)

            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(
                mdates.DayLocator(interval=max(1, len(timestamps) // 10))
            )
            plt.xticks(rotation=45)

            plt.tight_layout()

            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)

            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)

            return f"data:image/png;base64,{chart_base64}"

        except ImportError:
            logger.warning("Matplotlib not available for chart generation")
            return ""
        except Exception as e:
            logger.error(f"Error creating equity curve chart: {e}")
            return ""

    def _create_trades_chart(self, trades: List[TradeRecord]) -> str:
        """Create trades scatter plot"""
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 6))

            # Separate winning and losing trades
            timestamps = [trade.timestamp for trade in trades]
            pnls = [float(trade.pnl or 0) for trade in trades]

            winning_times = [t for t, pnl in zip(timestamps, pnls) if pnl > 0]
            winning_pnls = [pnl for pnl in pnls if pnl > 0]

            losing_times = [t for t, pnl in zip(timestamps, pnls) if pnl < 0]
            losing_pnls = [pnl for pnl in pnls if pnl < 0]

            # Plot trades
            if winning_times:
                ax.scatter(
                    winning_times,
                    winning_pnls,
                    color="green",
                    alpha=0.6,
                    s=50,
                    label="Winning Trades",
                )
            if losing_times:
                ax.scatter(
                    losing_times,
                    losing_pnls,
                    color="red",
                    alpha=0.6,
                    s=50,
                    label="Losing Trades",
                )

            # Add zero line
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

            # Formatting
            ax.set_title("Individual Trade P&L", fontsize=16, fontweight="bold")
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("P&L ($)", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()

            plt.xticks(rotation=45)
            plt.tight_layout()

            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)

            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)

            return f"data:image/png;base64,{chart_base64}"

        except ImportError:
            logger.warning("Matplotlib not available for chart generation")
            return ""
        except Exception as e:
            logger.error(f"Error creating trades chart: {e}")
            return ""

    def _create_monthly_returns_chart(self, equity_curve) -> str:
        """Create monthly returns heatmap"""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np

            # Convert equity curve to pandas DataFrame
            data = []
            for point in equity_curve:
                data.append({"timestamp": point.timestamp, "value": float(point.value)})

            df = pd.DataFrame(data)
            df.set_index("timestamp", inplace=True)

            # Calculate monthly returns
            monthly_returns = (
                df.groupby(pd.Grouper(freq="M"))["value"].last().pct_change().dropna()
            )

            if len(monthly_returns) == 0:
                return ""

            # Create simple bar chart instead of heatmap for simplicity
            fig, ax = plt.subplots(figsize=(12, 6))

            colors = ["green" if x > 0 else "red" for x in monthly_returns.values]
            bars = ax.bar(
                range(len(monthly_returns)),
                monthly_returns.values * 100,
                color=colors,
                alpha=0.7,
            )

            # Formatting
            ax.set_title("Monthly Returns (%)", fontsize=16, fontweight="bold")
            ax.set_xlabel("Month", fontsize=12)
            ax.set_ylabel("Return (%)", fontsize=12)
            ax.grid(True, alpha=0.3, axis="y")
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)

            # Set x-axis labels
            month_labels = [date.strftime("%Y-%m") for date in monthly_returns.index]
            ax.set_xticks(range(len(monthly_returns)))
            ax.set_xticklabels(month_labels, rotation=45)

            plt.tight_layout()

            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)

            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)

            return f"data:image/png;base64,{chart_base64}"

        except ImportError:
            logger.warning("Matplotlib/Pandas not available for chart generation")
            return ""
        except Exception as e:
            logger.error(f"Error creating monthly returns chart: {e}")
            return ""

    def _create_optimization_chart(self, optimization_results: Dict) -> str:
        """Create optimization progress chart"""
        try:
            import matplotlib.pyplot as plt

            # Extract optimization history
            if "optimization_history" not in optimization_results:
                return ""

            history = optimization_results["optimization_history"]
            trials = [h["trial"] for h in history]
            values = [h["value"] for h in history]

            # Calculate best value so far
            best_values = []
            current_best = float("-inf")
            for value in values:
                if value > current_best:
                    current_best = value
                best_values.append(current_best)

            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot optimization progress
            ax.plot(
                trials,
                values,
                "o-",
                alpha=0.6,
                label="Trial Values",
                color="lightblue",
            )
            ax.plot(
                trials,
                best_values,
                "-",
                linewidth=2,
                label="Best Value So Far",
                color="darkblue",
            )

            # Formatting
            ax.set_title(
                "Hyperparameter Optimization Progress",
                fontsize=16,
                fontweight="bold",
            )
            ax.set_xlabel("Trial", fontsize=12)
            ax.set_ylabel("Objective Value", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()

            plt.tight_layout()

            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)

            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)

            return f"data:image/png;base64,{chart_base64}"

        except ImportError:
            logger.warning("Matplotlib not available for chart generation")
            return ""
        except Exception as e:
            logger.error(f"Error creating optimization chart: {e}")
            return ""

    def _create_parameter_importance_chart(self, optimization_results: Dict) -> str:
        """Create parameter importance chart"""
        # Simplified version - would need more sophisticated analysis for real implementation
        return ""

    def _create_session_pnl_chart(self, trades: List[TradeRecord]) -> str:
        """Create session P&L chart"""
        try:
            import matplotlib.pyplot as plt

            # Calculate cumulative P&L
            cumulative_pnl = []
            running_total = 0
            timestamps = []

            for trade in sorted(trades, key=lambda x: x.timestamp):
                running_total += float(trade.pnl or 0)
                cumulative_pnl.append(running_total)
                timestamps.append(trade.timestamp)

            if not timestamps:
                return ""

            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot cumulative P&L
            ax.plot(timestamps, cumulative_pnl, linewidth=2, color="#2E86AB")
            ax.fill_between(timestamps, cumulative_pnl, alpha=0.3, color="#2E86AB")
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

            # Formatting
            ax.set_title("Session Cumulative P&L", fontsize=16, fontweight="bold")
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Cumulative P&L ($)", fontsize=12)
            ax.grid(True, alpha=0.3)

            plt.xticks(rotation=45)
            plt.tight_layout()

            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)

            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)

            return f"data:image/png;base64,{chart_base64}"

        except ImportError:
            logger.warning("Matplotlib not available for chart generation")
            return ""
        except Exception as e:
            logger.error(f"Error creating session P&L chart: {e}")
            return ""

    def _create_session_volume_chart(self, trades: List[TradeRecord]) -> str:
        """Create session volume chart"""
        try:
            import matplotlib.pyplot as plt

            # Group trades by hour
            hourly_volume = {}
            for trade in trades:
                hour_key = trade.timestamp.replace(minute=0, second=0, microsecond=0)
                if hour_key not in hourly_volume:
                    hourly_volume[hour_key] = 0
                hourly_volume[hour_key] += float(trade.quantity) * float(trade.price)

            if not hourly_volume:
                return ""

            hours = sorted(hourly_volume.keys())
            volumes = [hourly_volume[hour] for hour in hours]

            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot volume bars
            ax.bar(hours, volumes, alpha=0.7, color="orange")

            # Formatting
            ax.set_title("Trading Volume by Hour", fontsize=16, fontweight="bold")
            ax.set_xlabel("Hour", fontsize=12)
            ax.set_ylabel("Volume ($)", fontsize=12)
            ax.grid(True, alpha=0.3, axis="y")

            plt.xticks(rotation=45)
            plt.tight_layout()

            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)

            chart_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)

            return f"data:image/png;base64,{chart_base64}"

        except ImportError:
            logger.warning("Matplotlib not available for chart generation")
            return ""
        except Exception as e:
            logger.error(f"Error creating session volume chart: {e}")
            return ""

    def _create_backtest_html(
        self,
        result: BacktestResult,
        equity_chart: str,
        trades_chart: str,
        monthly_chart: str,
        comparison_results: Optional[List[BacktestResult]] = None,
    ) -> str:
        """Create HTML content for backtest report"""

        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report - {strategy_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2E86AB; }}
        .metric-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
        .chart-container {{ margin-bottom: 30px; text-align: center; }}
        .chart-title {{ font-size: 18px; font-weight: bold; margin-bottom: 15px; }}
        .chart img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        .trades-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        .trades-table th, .trades-table td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        .trades-table th {{ background-color: #f8f9fa; font-weight: bold; }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
        .summary {{ background-color: #e9ecef; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Backtest Report</h1>
            <h2>{strategy_name} - {symbol}</h2>
            <p>Period: {start_date} to {end_date}</p>
            <p>Generated: {report_date}</p>
        </div>
        
        <div class="summary">
            <h3>Executive Summary</h3>
            <p>The {strategy_name} strategy was backtested on {symbol} from {start_date} to {end_date}. 
            The strategy achieved a total return of {total_return:.2%} with {total_trades} trades 
            and a win rate of {win_rate:.1%}.</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">${initial_capital:,.2f}</div>
                <div class="metric-label">Initial Capital</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${final_capital:,.2f}</div>
                <div class="metric-label">Final Capital</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {return_class}">{total_return:+.2%}</div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{sharpe_ratio:.3f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">{max_drawdown:.2%}</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{total_trades}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{win_rate:.1%}</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{profit_factor:.2f}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
        </div>
        
        {equity_chart_section}
        {trades_chart_section}
        {monthly_chart_section}
        
        <div>
            <h3>Recent Trades</h3>
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Quantity</th>
                        <th>Price</th>
                        <th>P&L</th>
                    </tr>
                </thead>
                <tbody>
                    {trades_rows}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
        """

        # Prepare data
        return_class = "positive" if result.total_return > 0 else "negative"

        # Prepare chart sections
        equity_chart_section = (
            f"""
        <div class="chart-container">
            <div class="chart-title">Portfolio Equity Curve</div>
            <div class="chart"><img src="{equity_chart}" alt="Equity Curve"></div>
        </div>
        """
            if equity_chart
            else ""
        )

        trades_chart_section = (
            f"""
        <div class="chart-container">
            <div class="chart-title">Individual Trade P&L</div>
            <div class="chart"><img src="{trades_chart}" alt="Trades Chart"></div>
        </div>
        """
            if trades_chart
            else ""
        )

        monthly_chart_section = (
            f"""
        <div class="chart-container">
            <div class="chart-title">Monthly Returns</div>
            <div class="chart"><img src="{monthly_chart}" alt="Monthly Returns"></div>
        </div>
        """
            if monthly_chart
            else ""
        )

        # Prepare trades table (last 20 trades)
        recent_trades = sorted(result.trades, key=lambda x: x.timestamp, reverse=True)[
            :20
        ]
        trades_rows = ""
        for trade in recent_trades:
            pnl_class = (
                "positive"
                if (trade.pnl or 0) > 0
                else "negative" if (trade.pnl or 0) < 0 else ""
            )
            trades_rows += f"""
            <tr>
                <td>{trade.timestamp.strftime('%Y-%m-%d %H:%M')}</td>
                <td>{trade.symbol}</td>
                <td>{trade.side}</td>
                <td>{trade.quantity:.6f}</td>
                <td>${trade.price:.2f}</td>
                <td class="{pnl_class}">${trade.pnl or 0:+.4f}</td>
            </tr>
            """

        # Fill template
        return html_template.format(
            strategy_name=result.strategy_name,
            symbol=result.symbol,
            start_date=result.start_date.strftime("%Y-%m-%d"),
            end_date=result.end_date.strftime("%Y-%m-%d"),
            report_date=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            initial_capital=float(result.initial_capital),
            final_capital=float(result.final_capital),
            total_return=float(result.total_return),
            return_class=return_class,
            sharpe_ratio=float(result.metrics.sharpe_ratio),
            max_drawdown=float(result.metrics.max_drawdown),
            total_trades=result.metrics.total_trades,
            win_rate=float(result.metrics.win_rate),
            profit_factor=float(result.metrics.profit_factor),
            equity_chart_section=equity_chart_section,
            trades_chart_section=trades_chart_section,
            monthly_chart_section=monthly_chart_section,
            trades_rows=trades_rows,
        )

    def _create_optimization_html(
        self,
        optimization_results: Dict,
        strategy_name: str,
        symbol: str,
        optimization_chart: str,
        importance_chart: str,
    ) -> str:
        """Create HTML content for optimization report"""

        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Optimization Report - {strategy_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .best-params {{ background-color: #e9ecef; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .param-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px; }}
        .param-card {{ background-color: white; padding: 15px; border-radius: 8px; text-align: center; }}
        .param-value {{ font-size: 18px; font-weight: bold; color: #2E86AB; }}
        .param-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
        .chart-container {{ margin-bottom: 30px; text-align: center; }}
        .chart-title {{ font-size: 18px; font-weight: bold; margin-bottom: 15px; }}
        .chart img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Hyperparameter Optimization Report</h1>
            <h2>{strategy_name} - {symbol}</h2>
            <p>Generated: {report_date}</p>
        </div>
        
        <div class="best-params">
            <h3>Best Parameters Found</h3>
            <p><strong>Best Objective Value:</strong> {best_value:.6f}</p>
            <p><strong>Total Trials:</strong> {n_trials}</p>
            
            <div class="param-grid">
                {best_params_cards}
            </div>
        </div>
        
        {optimization_chart_section}
    </div>
</body>
</html>
        """

        # Prepare best parameters cards
        best_params = optimization_results.get("best_params", {})
        best_params_cards = ""
        for param_name, param_value in best_params.items():
            best_params_cards += f"""
            <div class="param-card">
                <div class="param-value">{param_value}</div>
                <div class="param-label">{param_name}</div>
            </div>
            """

        # Prepare chart section
        optimization_chart_section = (
            f"""
        <div class="chart-container">
            <div class="chart-title">Optimization Progress</div>
            <div class="chart"><img src="{optimization_chart}" alt="Optimization Progress"></div>
        </div>
        """
            if optimization_chart
            else ""
        )

        return html_template.format(
            strategy_name=strategy_name,
            symbol=symbol,
            report_date=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            best_value=optimization_results.get("best_value", 0),
            n_trials=optimization_results.get("n_trials", 0),
            best_params_cards=best_params_cards,
            optimization_chart_section=optimization_chart_section,
        )

    def _create_session_html(
        self,
        trades: List[TradeRecord],
        metrics: Dict,
        session_start: datetime,
        session_end: datetime,
        pnl_chart: str,
        volume_chart: str,
    ) -> str:
        """Create HTML content for session report"""

        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Session Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2E86AB; }}
        .metric-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
        .chart-container {{ margin-bottom: 30px; text-align: center; }}
        .chart-title {{ font-size: 18px; font-weight: bold; margin-bottom: 15px; }}
        .chart img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Trading Session Report</h1>
            <p>Session: {session_start} to {session_end}</p>
            <p>Duration: {duration:.1f} hours</p>
            <p>Generated: {report_date}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{total_trades}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {pnl_class}">${total_pnl:+.2f}</div>
                <div class="metric-label">Total P&L</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${total_volume:,.0f}</div>
                <div class="metric-label">Total Volume</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{win_rate:.1%}</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{profit_factor:.2f}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{trades_per_hour:.1f}</div>
                <div class="metric-label">Trades/Hour</div>
            </div>
        </div>
        
        {pnl_chart_section}
        {volume_chart_section}
    </div>
</body>
</html>
        """

        # Prepare data
        pnl_class = "positive" if metrics["total_pnl"] > 0 else "negative"

        # Prepare chart sections
        pnl_chart_section = (
            f"""
        <div class="chart-container">
            <div class="chart-title">Cumulative P&L</div>
            <div class="chart"><img src="{pnl_chart}" alt="P&L Chart"></div>
        </div>
        """
            if pnl_chart
            else ""
        )

        volume_chart_section = (
            f"""
        <div class="chart-container">
            <div class="chart-title">Trading Volume</div>
            <div class="chart"><img src="{volume_chart}" alt="Volume Chart"></div>
        </div>
        """
            if volume_chart
            else ""
        )

        return html_template.format(
            session_start=session_start.strftime("%Y-%m-%d %H:%M"),
            session_end=session_end.strftime("%Y-%m-%d %H:%M"),
            duration=metrics["session_duration_hours"],
            report_date=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            total_trades=metrics["total_trades"],
            total_pnl=metrics["total_pnl"],
            pnl_class=pnl_class,
            total_volume=metrics["total_volume"],
            win_rate=metrics["win_rate"],
            profit_factor=metrics["profit_factor"],
            trades_per_hour=metrics["trades_per_hour"],
            pnl_chart_section=pnl_chart_section,
            volume_chart_section=volume_chart_section,
        )


# Convenience functions
def generate_quick_backtest_report(result: BacktestResult) -> str:
    """Generate a quick backtest report"""
    generator = ReportGenerator()
    return generator.generate_backtest_report(result)


def generate_quick_session_report(trades: List[TradeRecord]) -> str:
    """Generate a quick session report"""
    if not trades:
        return ""

    generator = ReportGenerator()
    session_start = min(trade.timestamp for trade in trades)
    session_end = max(trade.timestamp for trade in trades)

    return generator.generate_trading_session_report(trades, session_start, session_end)
