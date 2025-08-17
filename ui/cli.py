"""
Main CLI interface for the Cryptocurrency Trading Bot

This file serves as the main entry point and registers command apps from
separate modules organized by functionality.
"""

import typer
from pathlib import Path

from rich.table import Table
from rich.console import Console

from constants.icons import Icon
from rich.panel import Panel
from rich.text import Text

console = Console()

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
    ok_status = f"{Icon.SUCCESS} OK"
    missing_status = f"{Icon.ERROR} Missing"
    status_info = {
        "Configuration": missing_status,
        "Data": missing_status,
        "Database": missing_status,
        "Models": missing_status,
    }

    # Check configuration
    paths = {
        "Configuration": "config.json",
        "Data": "data",
        "Database": "trading_bot.db",
        "Models": "artifacts",
    }

    # Check all paths and update status
    for key, path in paths.items():
        try:
            if Path(path).exists():
                status_info[key] = ok_status
        except (PermissionError, OSError):
            # Handle permission errors or other OS-level errors gracefully
            # Status remains as missing_status
            pass

    version_text = Text()
    for file, status in status_info.items():
        color = "green" if status == ok_status else "red"
        version_text.append(f"{file}: ", style="cyan")
        version_text.append(f"{status}\n", style=color)

    panel = Panel(
        version_text, title=f"{Icon.ROBOT} Trading Bot Status", border_style="green"
    )

    console.print(panel)


@app.command("version")
def show_version():
    """Show bot version information"""
    version_text = Text()
    version_text.append("Bot Version: ", style="cyan")
    version_text.append("1.0.0\n", style="green")
    version_text.append("Python: ", style="cyan")
    version_text.append("3.12.8\n", style="green")
    version_text.append("Created: ", style="cyan")
    version_text.append("2025", style="green")

    panel = Panel(version_text, title="Version Information", border_style="blue")

    console.print(panel)


if __name__ == "__main__":
    app()
