"""
Main CLI interface for the Cryptocurrency Trading Bot

This file serves as the main entry point and registers command apps from
separate modules organized by functionality.
"""

import typer
from pathlib import Path

from rich.table import Table

from ui.output import OutputHandler


logger = OutputHandler("ui.cli")

app = typer.Typer(help="Cryptocurrency Trading Bot CLI")

# Import command apps from the commands package
from ui.commands import (
    config_app,
    data_app,
    backtest_app,
    model_app,
    grid_app,
    trade_app,
    safety_app,
    monitor_app,
)

# Register command apps
app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")
app.add_typer(backtest_app, name="backtest")
app.add_typer(model_app, name="model")
app.add_typer(grid_app, name="grid")
app.add_typer(trade_app, name="trade")
app.add_typer(safety_app, name="safety")
app.add_typer(monitor_app, name="monitor")


@app.command("status")
def show_status():
    """Show bot status"""
    ok_status = "✅ OK"
    missing_status = "❌ Missing"
    status_info = {
        "Configuration": missing_status,
        "Data": missing_status,
        "Database": missing_status,
        "Models": missing_status
    }

    # Check configuration
    paths = {
        "Configuration": "config.json",
        "Data": "data", 
        "Database": "trading_bot.db",
        "Models": "artifacts"
    }

    # Check all paths and update status
    for key, path in paths.items():
        if Path(path).exists():
            status_info[key] = ok_status

    table = Table(title="🤖 Trading Bot Status")
    table.add_column("Directory", style="cyan")
    table.add_column("Status", style="cyan")

    for file, status in status_info.items():
        status_style = "green" if status == ok_status else "red"
        table.add_row(file, f"[{status_style}]{status}[/{status_style}]")
    
    logger.console.print(table)


@app.command("version")
def show_version():
    """Show bot version information"""
    version_info = {"Bot Version": "1.0.0", "Python": "3.10+", "Created": "2024"}

    table = Table(title="Version Information")
    table.add_column("Component", style="cyan")
    table.add_column("Version", style="green")

    for component, version in version_info.items():
        table.add_row(component, version)

    logger.console.print(table)


if __name__ == "__main__":
    app()
