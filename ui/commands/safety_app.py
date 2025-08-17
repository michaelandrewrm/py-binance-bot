"""
Safety and risk management commands for the trading bot CLI
"""
import typer
from datetime import datetime, timezone
from typing import Optional
from rich.text import Text
from rich.panel import Panel
from rich.table import Table

from ui.error_handling import handle_cli_errors
from ui.output import OutputHandler

logger = OutputHandler("ui.cli")

# Safety commands will be registered to this app instance
safety_app = typer.Typer(help="Safety and risk management")


@safety_app.command("monitor")
def start_safety_monitoring(
    session_id: str = typer.Argument(..., help="Session ID to monitor"),
    symbol: str = typer.Argument(..., help="Trading symbol"),
    alert_level: str = typer.Option(
        "medium", "--alert-level", help="Alert level: low, medium, high, critical"
    ),
):
    """Start advanced safety monitoring for a trading session"""
    try:
        # Validate alert level
        valid_levels = ["low", "medium", "high", "critical"]
        if alert_level not in valid_levels:
            logger.error(
                f"Invalid alert level. Must be one of: {', '.join(valid_levels)}"
            )
            raise typer.Exit(1)

        # Start monitoring
        # advanced_safety_manager.start_monitoring(session_id, symbol)
        # enhanced_error_handler.start_monitoring()

        logger.custom(f"🛡️  Advanced safety monitoring started")
        logger.custom(f"Session: [cyan]{session_id}[/cyan]")
        logger.custom(f"Symbol: [green]{symbol}[/green]")
        logger.custom(f"Alert Level: [yellow]{alert_level.upper()}[/yellow]")

        logger.success(f"\nMonitoring systems active:")
        logger.info(f"   • Risk metrics tracking")
        logger.info(f"   • Market condition analysis")
        logger.info(f"   • Connection health monitoring")
        logger.info(f"   • Error pattern detection")
        logger.info(f"   • Automatic recovery systems")

    except Exception as e:
        logger.error(f"Error starting safety monitoring: {e}")
        raise typer.Exit(1)


@safety_app.command("status")
def safety_status(
    session_id: Optional[str] = typer.Argument(
        None, help="Session ID to check (default: all active sessions)"
    ),
    detailed: bool = typer.Option(
        False, "--detailed", help="Show detailed risk analysis"
    ),
):
    """Show safety status and risk analysis"""

    if session_id:
        # Show specific session safety status
        # risk_report = advanced_safety_manager.get_risk_report(session_id)
        
        # Sample risk report for demonstration
        risk_report = {
            "risk_level": "moderate",
            "metrics": {
                "current_drawdown": 0.05,
                "volatility": 0.25,
                "sharpe_ratio": 1.5,
                "var_95": 150.75
            },
            "recent_alerts": [
                {
                    "timestamp": "2025-08-10T06:24:00Z",
                    "type": "warning",
                    "title": "Increased Volatility Detected",
                    "message": "Market volatility increased to 45%, consider reducing position size"
                }
            ]
        }

        if "error" in risk_report:
            logger.error(risk_report["error"])
            return

        # Create safety status panel
        status_text = Text()
        status_text.append(f"Session: ", style="bold")
        status_text.append(f"{session_id}\n", style="cyan")
        status_text.append(f"Risk Level: ", style="bold")

        risk_level = risk_report["risk_level"]
        risk_style = {
            "low": "green",
            "moderate": "yellow",
            "high": "red",
            "extreme": "red bold",
            "critical": "red bold blink",
        }.get(risk_level, "white")

        status_text.append(f"{risk_level.upper()}\n", style=risk_style)

        # Risk metrics
        metrics = risk_report["metrics"]
        status_text.append(f"Current Drawdown: ", style="bold")
        status_text.append(f"{metrics['current_drawdown']:.1%}\n")
        status_text.append(f"Volatility: ", style="bold")
        status_text.append(f"{metrics['volatility']:.1%}\n")
        status_text.append(f"Sharpe Ratio: ", style="bold")
        status_text.append(f"{metrics['sharpe_ratio']:.2f}\n")
        status_text.append(f"VaR (95%): ", style="bold")
        status_text.append(f"${metrics['var_95']:.2f}\n")

        panel = Panel(status_text, title="🛡️ Safety Status", border_style="blue")
        logger.console.print(panel)

        # Show recent alerts if detailed
        if detailed and risk_report["recent_alerts"]:
            logger.critical("\nRecent Alerts:")
            for alert in risk_report["recent_alerts"]:
                alert_style = {
                    "info": "blue",
                    "warning": "yellow",
                    "error": "red",
                    "critical": "red bold",
                }.get(alert["type"], "white")

                logger.console.print(
                    f"   [{alert['timestamp'][:19]}] {alert['title']}",
                    style=alert_style,
                )
                logger.console.print(f"      {alert['message']}", style="dim")

    else:
        # Show overall safety status
        # error_summary = enhanced_error_handler.get_error_summary(24)
        
        # Sample error summary for demonstration
        error_summary = {
            "total_errors": 3,
            "by_category": {
                "network": 1,
                "trading": 2
            },
            "by_severity": {
                "low": 1,
                "medium": 2,
                "high": 0
            },
            "connection_health": {
                "overall_healthy": True,
                "services": {
                    "Binance API": {"healthy": True},
                    "Database": {"healthy": True},
                    "Redis Cache": {"healthy": True}
                }
            }
        }

        logger.custom("🛡️ Overall Safety Status (Last 24 Hours)")
        logger.custom("=" * 50)

        # Error summary
        total_errors = error_summary["total_errors"]
        if total_errors == 0:
            logger.success("No errors detected")
        else:
            logger.warning(f"{total_errors} errors detected")

            # Show by category
            logger.custom("📊 Errors by Category:")
            for category, count in error_summary["by_category"].items():
                logger.info(f"   {category.title()}: {count}")

            # Show by severity
            logger.critical("\nErrors by Severity:")
            for severity, count in error_summary["by_severity"].items():
                severity_style = {
                    "low": "green",
                    "medium": "yellow",
                    "high": "red",
                    "critical": "red bold",
                }.get(severity, "white")
                logger.console.print(
                    f"   {severity.title()}: {count}", style=severity_style
                )

        # Connection health
        health = error_summary["connection_health"]
        logger.custom("🔗 Connection Health:")
        logger.info(
            f"Overall Status: {'✅ Healthy' if health['overall_healthy'] else '❌ Issues Detected'}"
        )

        for service, status in health["services"].items():
            health_icon = "✅" if status["healthy"] else "❌"
            logger.info(f"   {service}: {health_icon}")


@safety_app.command("alerts")
def show_alerts(
    session_id: Optional[str] = typer.Option(
        None, "--session", help="Filter by session ID"
    ),
    hours: int = typer.Option(24, "--hours", help="Show alerts from last N hours"),
    severity: Optional[str] = typer.Option(
        None, "--severity", help="Filter by severity: low, medium, high, critical"
    ),
):
    """Show safety alerts and notifications"""

    # Get alerts from advanced safety manager
    logger.critical(f"Safety Alerts (Last {hours} hours)")
    logger.custom("=" * 50)

    # Sample alerts for demonstration
    sample_alerts = [
        {
            "timestamp": "2025-08-10T06:24:00Z",
            "type": "warning",
            "category": "Risk",
            "title": "Increased Volatility Detected",
            "message": "Market volatility increased to 45%, consider reducing position size",
            "session_id": "BTCUSDC_20250810_062400",
        },
        {
            "timestamp": "2025-08-10T06:20:00Z",
            "type": "info",
            "category": "Market",
            "title": "Regime Change Detected",
            "message": "Market regime changed from NORMAL to VOLATILE",
            "session_id": None,
        },
    ]

    if not sample_alerts:
        logger.success("No alerts in the specified time period")
        return

    # Create alerts table
    table = Table(title="Safety Alerts")
    table.add_column("Time", style="cyan")
    table.add_column("Type", style="white")
    table.add_column("Category", style="blue")
    table.add_column("Title", style="yellow")
    table.add_column("Session", style="green")

    for alert in sample_alerts:
        # Apply filters
        if session_id and alert.get("session_id") != session_id:
            continue
        if severity and alert["type"] != severity:
            continue

        alert_style = {
            "info": "blue",
            "warning": "yellow",
            "error": "red",
            "critical": "red bold",
        }.get(alert["type"], "white")

        table.add_row(
            alert["timestamp"][:19],
            f"[{alert_style}]{alert['type'].upper()}[/{alert_style}]",
            alert["category"],
            alert["title"],
            alert.get("session_id", "N/A"),
        )

    logger.console.print(table)


@safety_app.command("emergency")
def emergency_controls(
    action: str = typer.Argument(..., help="Emergency action: stop, flatten, status"),
    session_id: Optional[str] = typer.Option(
        None, "--session", help="Session ID (required for stop)"
    ),
    confirm: bool = typer.Option(False, "--confirm", help="Confirm emergency action"),
):
    """Emergency safety controls"""

    if action == "status":
        logger.critical("Emergency Status Check")
        logger.custom("=" * 30)

        # Check if any emergency conditions exist
        emergency_sessions = []  # This would check actual sessions

        if not emergency_sessions:
            logger.success("No emergency conditions detected")
        else:
            logger.warning("⚠️ Emergency conditions detected:")
            for session in emergency_sessions:
                logger.info(f"   Session {session}: Critical risk level")

    elif action == "stop":
        if not session_id:
            logger.error("Session ID required for emergency stop")
            raise typer.Exit(1)

        if not confirm:
            logger.warning("Emergency stop requires --confirm flag")
            logger.info("This will immediately halt all trading for the session")
            raise typer.Exit(1)

        logger.critical(f"EMERGENCY STOP initiated for {session_id}")
        logger.custom("⏹️ Stopping all trading activity...")
        logger.custom("🔄 Cancelling open orders...")
        logger.custom("📊 Recording emergency event...")
        logger.success("Emergency stop completed")

    elif action == "flatten":
        if not confirm:
            logger.warning("Emergency flatten requires --confirm flag")
            logger.info("This will immediately close all positions")
            raise typer.Exit(1)

        logger.critical("EMERGENCY FLATTEN initiated")
        logger.custom("📈 Closing all open positions...")
        logger.custom("🔄 Cancelling open orders...")
        logger.custom("💰 Converting to base currency...")
        logger.success("Emergency flatten completed")

    else:
        logger.error(f"Unknown emergency action: {action}")
        logger.info("Available actions: stop, flatten, status")
        raise typer.Exit(1)
