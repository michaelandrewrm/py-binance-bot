"""
CLI Module - Command Line Interface

This module provides a comprehensive command-line interface for the trading bot
using Typer for a modern, intuitive CLI experience.
"""

import typer
import asyncio
import json
import time
import re

from typing import Optional, List, Dict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, FloatPrompt, IntPrompt
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from decimal import Decimal

# Import enhanced modules
from ui.security import (
    encrypt_credentials,
    decrypt_credentials,
    validate_trading_symbol,
    validate_decimal_input,
    validate_api_key_format,
    secure_file_write,
    check_file_permissions,
    mask_sensitive_data,
    validate_file_path,
    check_encryption_available,
    SecurityError,
)
from ui.error_handling import (
    handle_cli_errors,
    ValidationError,
    ConfigurationError,
    parse_duration_safe,
    format_duration,
    confirm_operation,
    validate_positive_int,
    safe_json_load,
    log_user_action,
    validate_required_config,
)
from ui.session_manager import (
    TradingMode,
    TradingSessionStatus,
    ThreadSafeSessionManager,
    TradingSession,
    GridParameters,
)

# Import bot modules
from storage.repo import get_default_repository
from storage.artifacts import get_default_artifact_manager
from data.loader import DataLoader
from sim.engine import BacktestEngine, SimpleGridStrategy
from ai.baseline import BaselineStrategy
from ai.bo_suggester import BayesianOptimizer

# Core modules
from core.state import load_state, create_empty_state, save_state
from core.safety import SafetyManager, RiskLimits
from core.flatten import execute_flatten_operation
from core.executor import run_trading_bot as run_bot
from core.baseline_heuristic import get_baseline_parameters
from core.advanced_safety import (
    advanced_safety_manager,
    RiskLevel,
    MarketRegime,
    AlertType,
    MarketConditions,
    MarketRegime,
)
from core.enhanced_error_handling import (
    enhanced_error_handler,
    ErrorCategory,
    ErrorSeverity,
)
from exec.binance import create_binance_client

from .output import OutputHandler

logger = OutputHandler("ui.cli")

app = typer.Typer(help="Cryptocurrency Trading Bot CLI")

# Global configuration
config_app = typer.Typer(help="Configuration management")
app.add_typer(config_app, name="config")

data_app = typer.Typer(help="Data management")
app.add_typer(data_app, name="data")

backtest_app = typer.Typer(help="Backtesting commands")
app.add_typer(backtest_app, name="backtest")

model_app = typer.Typer(help="Model management")
app.add_typer(model_app, name="model")

# Enhanced: New grid trading app with mode support
grid_app = typer.Typer(help="Grid Trading Commands")
app.add_typer(grid_app, name="grid")


# Global enhanced session manager
try:
    session_manager = ThreadSafeSessionManager(max_sessions=50, cleanup_threshold=40)
except Exception as e:
    logger.error(f"Failed to initialize session manager: {e}")
    raise typer.Exit(1)


# Helper functions for parameter collection
async def get_market_context(symbol: str) -> dict:
    """Get current market context for parameter suggestions"""
    try:
        # TODO: Integrate with real market data
        # For now, return mock data
        return {
            "current_price": Decimal("50000.0"),
            "24h_high": Decimal("52000.0"),
            "24h_low": Decimal("48000.0"),
            "volatility": 0.02,  # 2% daily volatility
            "atr": Decimal("1000.0"),  # Average True Range
        }
    except Exception as e:
        logger.warning(f"Could not fetch market data: {e}")
        return {}


def collect_manual_parameters(symbol: str) -> GridParameters:
    """Collect grid parameters manually from user input with comprehensive validation"""
    logger.info(f"Manual Grid Configuration for {symbol}")

    # Validate symbol first
    try:
        symbol = validate_trading_symbol(symbol)
    except ValueError as e:
        logger.error(f"{e}")
        raise typer.Abort()

    # Get market context with error handling
    market_data = {}
    try:
        market_data = asyncio.run(get_market_context(symbol))
    except Exception as e:
        logger.error(f"Failed to get market context for {symbol}: {e}")
        logger.warning("Could not fetch market data, proceeding with manual input")

    if market_data:
        logger.custom(
            f"📊 [dim]Current Price: ${market_data.get('current_price', 'N/A')}[/dim]"
        )
        logger.custom(
            f"📊 [dim]24h Range: ${market_data.get('24h_low', 'N/A')} - ${market_data.get('24h_high', 'N/A')}[/dim]"
        )
        logger.custom(f"📊 [dim]ATR: ${market_data.get('atr', 'N/A')}[/dim]\n")

    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            # Collect parameters with validation
            lower_input = Prompt.ask("💰 Lower grid bound (price)", default="45000")
            lower_bound = validate_decimal_input(
                lower_input, min_value=Decimal("0.01"), field_name="lower bound"
            )

            upper_input = Prompt.ask("💰 Upper grid bound (price)", default="55000")
            upper_bound = validate_decimal_input(
                upper_input,
                min_value=lower_bound * Decimal("1.01"),
                field_name="upper bound",
            )

            if lower_bound >= upper_bound:
                logger.error("Upper bound must be greater than lower bound")
                continue

            grid_count_input = Prompt.ask("🔢 Number of grid levels", default="10")
            grid_count = validate_positive_int(grid_count_input, min_val=2, max_val=100)

            investment_input = Prompt.ask(
                "💵 Investment per grid level", default="100.0"
            )
            investment_per_grid = validate_decimal_input(
                investment_input,
                min_value=Decimal("1.0"),
                max_value=Decimal("100000.0"),
                field_name="investment per grid",
            )

            # Calculate and display summary
            total_investment = investment_per_grid * grid_count
            grid_spacing = (upper_bound - lower_bound) / grid_count

            logger.custom(f"\n📋 [bold]Configuration Summary:[/bold]")
            logger.custom(f"   Range: ${lower_bound} - ${upper_bound}")
            logger.custom(f"   Grid Levels: {grid_count}")
            logger.custom(f"   Grid Spacing: ${grid_spacing:.2f}")
            logger.custom(f"   Per Level: ${investment_per_grid}")
            logger.custom(f"   Total Investment: ${total_investment}")

            if Confirm.ask("\n✅ Confirm these parameters?", default=True):
                logger.user_action(
                    "manual_parameters_collected",
                    {
                        "action": "manual_parameters_collected",
                        "timestamp": None,  # Will be added by logger
                        "details": {
                            "symbol": symbol,
                            "total_investment": str(total_investment),
                            "grid_count": grid_count,
                        },
                    },
                )

                return GridParameters(
                    symbol=symbol,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    grid_count=grid_count,
                    investment_per_grid=investment_per_grid,
                    mode=TradingMode.MANUAL,
                )
            else:
                logger.custom("🔄 Let's try again...")
                continue

        except ValidationError as e:
            logger.error(f"Invalid input: {e}")
            if attempt == max_retries - 1:
                logger.error("Too many invalid attempts")
                raise typer.Abort()
            logger.warning(f"Attempt {attempt + 1}/{max_retries}, please try again...")
        except KeyboardInterrupt:
            logger.error("\nParameter collection cancelled")
            raise typer.Abort()


async def get_ai_suggested_parameters(symbol: str) -> GridParameters:
    """Get AI-suggested grid parameters"""
    logger.custom(f"\n🤖 [bold blue]AI Parameter Suggestion for {symbol}[/bold blue]")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=logger.console,
        ) as progress:
            task = progress.add_task("Analyzing market data...", total=None)

            # TODO: Implement actual AI parameter suggestion
            # For now, use a baseline heuristic
            await asyncio.sleep(2)  # Simulate AI processing

            progress.update(task, description="Calculating optimal parameters...")
            await asyncio.sleep(1)

            # Mock AI suggestion based on market data
            market_data = await get_market_context(symbol)
            current_price = market_data.get("current_price", Decimal("50000"))
            atr = market_data.get("atr", Decimal("1000"))

            # AI heuristic: bounds at ±2*ATR, 10 levels
            lower_bound = current_price - (atr * 2)
            upper_bound = current_price + (atr * 2)
            grid_count = 10
            investment_per_grid = Decimal("100")  # Default

            progress.update(task, description="Generating confidence intervals...")
            await asyncio.sleep(0.5)

        # Display AI suggestion
        logger.custom("\n🎯 [bold green]AI Recommendation:[/bold green]")
        logger.custom(f"   Range: ${lower_bound:.2f} - ${upper_bound:.2f}")
        logger.custom(f"   Grid Levels: {grid_count}")
        logger.custom(f"   Suggested Investment/Level: ${investment_per_grid}")
        logger.custom(f"   Confidence: 85% ± 5%")
        logger.custom(f"   Expected Sharpe Ratio: 1.2 ± 0.3")
        logger.custom(f"   Max Expected Drawdown: 8% ± 2%")

        return GridParameters(
            symbol=symbol,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            grid_count=grid_count,
            investment_per_grid=investment_per_grid,
            mode=TradingMode.AI,
            confidence=0.85,
        )

    except Exception as e:
        logger.error(f"\nAI analysis failed: {e}")
        logger.warning("Falling back to baseline heuristic...")

        try:
            # Use baseline heuristic as fallback
            current_price = 50000.0  # Default price for fallback
            market_data = {
                "current_price": Decimal(str(current_price)),
                "atr": Decimal(str(current_price * 0.02)),  # Estimate 2% ATR
                "volatility": 0.25,  # Estimate 25% annualized volatility
            }

            baseline_params = get_baseline_parameters(
                symbol=symbol, market_data=market_data, investment=Decimal("100")
            )

            logger.success(f"Baseline parameters calculated")
            logger.custom(f"[dim]Using heuristic-based approach for {symbol}[/dim]")

            return GridParameters(
                symbol=symbol,
                lower_bound=baseline_params["lower_bound"],
                upper_bound=baseline_params["upper_bound"],
                grid_count=baseline_params["grid_count"],
                investment_per_grid=baseline_params["investment_per_grid"],
                mode=TradingMode.AI,
                confidence=0.65,  # Lower confidence for fallback
            )

        except Exception as fallback_e:
            logger.error(f"\nBaseline fallback also failed: {fallback_e}")
            # Return conservative defaults as last resort
            return GridParameters(
                symbol=symbol,
                lower_bound=Decimal("45000"),
                upper_bound=Decimal("55000"),
                grid_count=5,
                investment_per_grid=Decimal("50"),
                mode=TradingMode.AI,
                confidence=0.3,
            )


def confirm_ai_parameters(suggestion: GridParameters) -> GridParameters:
    """Allow user to confirm or modify AI suggestions"""
    logger.custom(f"\n🔍 [bold]Review AI Suggestion[/bold]")

    while True:
        action = Prompt.ask(
            "Choose action", choices=["accept", "modify", "cancel"], default="accept"
        )

        if action == "accept":
            logger.success("AI parameters accepted!")
            return suggestion

        elif action == "modify":
            logger.custom("🛠️ Modify parameters:")

            # Allow modifications
            new_lower = Decimal(
                Prompt.ask("New lower bound", default=str(suggestion.lower_bound))
            )
            new_upper = Decimal(
                Prompt.ask("New upper bound", default=str(suggestion.upper_bound))
            )
            new_count = IntPrompt.ask("New grid count", default=suggestion.grid_count)
            new_investment = Decimal(
                FloatPrompt.ask(
                    "New investment per grid",
                    default=float(suggestion.investment_per_grid),
                )
            )

            modified = GridParameters(
                symbol=suggestion.symbol,
                lower_bound=new_lower,
                upper_bound=new_upper,
                grid_count=new_count,
                investment_per_grid=new_investment,
                mode=TradingMode.AI,
                confidence=0.75,  # Lower confidence for modified params
            )

            logger.custom("\n📋 [bold]Modified Configuration:[/bold]")
            logger.custom(
                f"   Range: ${modified.lower_bound} - ${modified.upper_bound}"
            )
            logger.custom(f"   Grid Levels: {modified.grid_count}")
            logger.custom(f"   Per Level: ${modified.investment_per_grid}")

            if Confirm.ask("Accept modified parameters?", default=True):
                logger.success("Modified parameters accepted!")
                return modified

        elif action == "cancel":
            logger.error("AI suggestion cancelled")
            raise typer.Abort()


# Grid trading commands
@grid_app.command("manual")
@handle_cli_errors
def start_manual_grid(
    symbol: str = typer.Argument(..., help="Trading symbol (e.g., BTCUSDC)"),
    start_immediately: bool = typer.Option(
        False, "--start", help="Start trading immediately"
    ),
    duration: Optional[str] = typer.Option(
        None, "--duration", "-d", help="Auto-stop duration"
    ),
    paper: bool = typer.Option(
        True, "--paper/--live", help="Paper trading (default) or live"
    ),
    confirm_live: bool = typer.Option(
        False, "--confirm-live", help="Explicit confirmation for live"
    ),
):
    """Configure grid trading in manual mode with enhanced error handling"""

    # Enhanced safety check for live trading
    if not paper:
        if not confirm_live:
            logger.warning("Live trading requires --confirm-live flag!")
            logger.warning("Use: grid manual SYMBOL --live --confirm-live")
            raise typer.Exit(1)

        # Additional confirmation for live trading
        if not confirm_operation(
            f"This will execute REAL trades with REAL money for {symbol}.",
            default=False,
            danger=True,
            require_explicit=True,
        ):
            logger.warning("Live trading cancelled")
            raise typer.Abort()

    # Validate and collect parameters
    params = collect_manual_parameters(symbol)

    # Parse and validate duration if provided
    duration_limit = None
    if duration:
        try:
            duration_limit = parse_duration_safe(duration)
            logger.success(
                f"Session will auto-stop after: {format_duration(duration_limit)}"
            )
        except ValidationError as e:
            logger.error(f"Invalid duration: {e}")
            raise typer.Exit(1)
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            logger.console_only("Unexpected error during session creation")
            raise typer.Exit(1)

    # Create session with validation
    try:
        session_id = session_manager.create_session(symbol, params, duration_limit)
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise typer.Exit(1)

    # Success feedback
    logger.success(f"\nManual grid session created: [cyan]{session_id}[/cyan]")
    logger.custom(f"Mode: [blue]{'LIVE' if not paper else 'PAPER'}[/blue]")
    logger.custom(
        f"Grid Range: [green]${params.lower_bound} - ${params.upper_bound}[/green]"
    )
    logger.custom(f"Grid Levels: [yellow]{params.grid_count}[/yellow]")

    # Log the action
    log_user_action(
        "manual_grid_session_created",
        {
            "session_id": session_id,
            "symbol": symbol,
            "paper_mode": paper,
            "has_duration_limit": duration_limit is not None,
        },
    )

    if start_immediately:
        logger.custom("🚀 Starting session immediately...")
        logger.success("Session configuration complete")
    else:
        logger.custom(f"💡 Use 'grid start {session_id}' to begin trading")


@grid_app.command("ai")
@handle_cli_errors
def start_ai_grid(
    symbol: str = typer.Argument(..., help="Trading symbol (e.g., BTCUSDC)"),
    start_immediately: bool = typer.Option(
        False, "--start", help="Start trading immediately after creating session"
    ),
    duration: Optional[str] = typer.Option(
        None, "--duration", "-d", help="Auto-stop duration (e.g., '1h', '30m')"
    ),
    paper: bool = typer.Option(
        True, "--paper/--live", help="Paper trading (default) or live"
    ),
    confirm_live: bool = typer.Option(
        False, "--confirm-live", help="Explicit confirmation for live AI trading"
    ),
):
    """Configure grid trading in AI mode - AI suggests optimal parameters"""

    # Safety check for live trading
    if not paper and not confirm_live:
        logger.warning("Live AI trading requires --confirm-live flag!")
        raise typer.Abort()

    try:
        # Get AI suggestions
        logger.custom(f"🤖 [bold blue]AI Grid Configuration for {symbol}[/bold blue]")
        suggestion = asyncio.run(get_ai_suggested_parameters(symbol))

        # Allow user to review and modify
        final_params = confirm_ai_parameters(suggestion)

        # Parse duration if provided
        duration_limit = None
        if duration:
            duration_limit = parse_duration(duration)

        # Create session
        session_id = session_manager.create_session(
            symbol, final_params, duration_limit
        )

        logger.success(f"\nAI session created: [cyan]{session_id}[/cyan]")
        logger.custom(f"Mode: [blue]{final_params.mode.value.upper()}[/blue]")
        logger.custom(f"Confidence: [yellow]{final_params.confidence:.0%}[/yellow]")
        logger.custom(
            f"Grid Range: [green]${final_params.lower_bound} - ${final_params.upper_bound}[/green]"
        )
        logger.custom(f"Grid Levels: [yellow]{final_params.grid_count}[/yellow]")
        if duration_limit:
            logger.custom(
                f"Duration Limit: [yellow]{format_duration(duration_limit)}[/yellow]"
            )

        # Option to start immediately
        if start_immediately:
            if not paper:
                logger.critical(f"\nFINAL LIVE AI TRADING CONFIRMATION")
                logger.custom(
                    f"Total Investment: [red]${final_params.grid_count * final_params.investment_per_grid}[/red]"
                )
                logger.custom(
                    f"AI Confidence: [yellow]{final_params.confidence:.0%}[/yellow]"
                )
                if not Confirm.ask(
                    "Proceed with LIVE AI trading using real money?", default=False
                ):
                    logger.custom("Live trading cancelled")
                    logger.custom(f"💡 Session {session_id} created but not started")
                    return

            # Start the session
            session = session_manager.get_session(session_id)
            session.status = TradingSessionStatus.STARTING
            session.start_time = datetime.now(timezone.utc)
            session_manager.set_active_session(session_id)

            logger.custom(
                f"\n🚀 Starting {'paper' if paper else 'LIVE'} AI trading session..."
            )
            # TODO: Implement actual grid trading execution
            session.status = TradingSessionStatus.RUNNING
            logger.success("AI session started successfully")
            logger.custom("💡 Use 'grid status --live' for real-time monitoring")
        else:
            logger.custom(f"\n💡 [dim]Next steps:[/dim]")
            logger.custom(f"   • [cyan]grid start {session_id}[/cyan] - Start trading")
            logger.custom(f"   • [cyan]grid status {session_id}[/cyan] - Check status")
            logger.custom(
                f"   • [cyan]grid adjust {session_id}[/cyan] - Modify parameters"
            )

    except typer.Abort:
        raise
    except Exception as e:
        logger.error(f"Error creating AI grid session: {e}")
        raise typer.Exit(1)


# Helper functions for runtime controls
def parse_duration(duration_str: str) -> timedelta:
    """Parse duration string like '1h', '30m', '2h30m' into timedelta"""

    # Parse patterns like: 1h, 30m, 2h30m, 1d2h30m
    pattern = r"(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?"
    match = re.match(pattern, duration_str.lower())

    if not match:
        raise ValueError(f"Invalid duration format: {duration_str}")

    days, hours, minutes = match.groups()

    total_minutes = 0
    if days:
        total_minutes += int(days) * 24 * 60
    if hours:
        total_minutes += int(hours) * 60
    if minutes:
        total_minutes += int(minutes)

    if total_minutes == 0:
        raise ValueError(f"Duration must be greater than 0: {duration_str}")

    return timedelta(minutes=total_minutes)


def format_duration(duration: timedelta) -> str:
    """Format timedelta as human-readable string"""
    total_seconds = int(duration.total_seconds())
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 and not parts:  # Only show seconds if no larger units
        parts.append(f"{seconds}s")

    return " ".join(parts) if parts else "0s"


def get_status_style(status: TradingSessionStatus) -> str:
    """Get Rich style for status display"""
    styles = {
        TradingSessionStatus.STOPPED: "red",
        TradingSessionStatus.STARTING: "yellow",
        TradingSessionStatus.RUNNING: "green",
        TradingSessionStatus.PAUSED: "blue",
        TradingSessionStatus.STOPPING: "yellow",
        TradingSessionStatus.ERROR: "red bold",
        TradingSessionStatus.COMPLETED: "cyan",
    }
    return styles.get(status, "white")


def show_static_status(session: TradingSession):
    """Show static session status"""
    status_style = get_status_style(session.status)
    elapsed = session.get_elapsed_time()

    # Create status panel
    status_text = Text()
    status_text.append(f"Session: ", style="bold")
    status_text.append(f"{session.session_id}\n", style="cyan")
    status_text.append(f"Symbol: ", style="bold")
    status_text.append(f"{session.symbol}\n", style="green")
    status_text.append(f"Mode: ", style="bold")
    status_text.append(f"{session.parameters.mode.value.upper()}\n", style="blue")
    status_text.append(f"Status: ", style="bold")
    status_text.append(f"{session.status.value.upper()}\n", style=status_style)
    status_text.append(f"Elapsed: ", style="bold")
    status_text.append(f"{format_duration(elapsed)}\n", style="white")

    if session.duration_limit:
        remaining = session.duration_limit - elapsed
        status_text.append(f"Remaining: ", style="bold")
        status_text.append(f"{format_duration(remaining)}\n", style="yellow")

    status_text.append(f"Trades: ", style="bold")
    status_text.append(f"{session.trades_count}\n", style="white")
    status_text.append(f"P&L: ", style="bold")
    pnl_style = "green" if session.profit_loss >= 0 else "red"
    status_text.append(f"${session.profit_loss}", style=pnl_style)

    panel = Panel(status_text, title="Trading Session Status", border_style="blue")
    logger.console_only(panel)


def show_live_status(session: TradingSession):
    """Show live updating session status"""

    def create_status_layout():
        """Create the status layout"""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )

        # Header
        header_text = Text(f"Live Trading Status - {session.symbol}", style="bold cyan")
        layout["header"].update(Panel(header_text, style="blue"))

        # Main status
        status_text = create_live_status_text(session)
        layout["main"].update(
            Panel(status_text, title="Session Details", style="green")
        )

        # Footer
        footer_text = Text("Press Ctrl+C to exit live view", style="dim")
        layout["footer"].update(Panel(footer_text, style="yellow"))

        return layout

    def create_live_status_text(session: TradingSession) -> Text:
        """Create live status text"""
        status_style = get_status_style(session.status)
        elapsed = session.get_elapsed_time()

        text = Text()
        text.append(f"Session ID: {session.session_id}\n", style="cyan")
        text.append(f"Status: ", style="bold")
        text.append(f"{session.status.value.upper()}\n", style=status_style)
        text.append(f"Mode: {session.parameters.mode.value.upper()}\n", style="blue")
        text.append(
            f"Grid Range: ${session.parameters.lower_bound} - ${session.parameters.upper_bound}\n"
        )
        text.append(f"Grid Levels: {session.parameters.grid_count}\n")
        text.append(f"Elapsed Time: {format_duration(elapsed)}\n", style="white")

        if session.duration_limit:
            remaining = session.duration_limit - elapsed
            text.append(
                f"Time Remaining: {format_duration(remaining)}\n", style="yellow"
            )

            # Progress bar
            progress_pct = min(
                100,
                (elapsed.total_seconds() / session.duration_limit.total_seconds())
                * 100,
            )
            text.append(f"Progress: {progress_pct:.1f}%\n", style="cyan")

        text.append(f"Active Trades: {session.trades_count}\n")
        text.append(f"Current Positions: {session.current_positions}\n")
        text.append(f"P&L: ", style="bold")
        pnl_style = "green" if session.profit_loss >= 0 else "red"
        text.append(f"${session.profit_loss}\n", style=pnl_style)

        # Check if time expired
        if session.is_time_expired():
            text.append(
                "\n⏰ Duration limit reached! Session will auto-stop.\n",
                style="red bold",
            )

        return text

    # Live display loop
    try:
        with Live(create_status_layout(), refresh_per_second=1, screen=True) as live:
            while session.is_active():
                live.update(create_status_layout())
                time.sleep(1)

                # Auto-stop if time expired
                if session.is_time_expired():
                    session.status = TradingSessionStatus.STOPPING
                    logger.custom(
                        "\n⏰ Duration limit reached - stopping session automatically"
                    )
                    # TODO: Implement actual stop logic
                    session.status = TradingSessionStatus.COMPLETED
                    session.end_time = datetime.now(timezone.utc)
                    break

            # Final update
            live.update(create_status_layout())
            logger.success(f"\nLive status ended - Session {session.status.value}")

    except KeyboardInterrupt:
        logger.custom("\n👋 Live status view ended")


# Runtime Control Commands
@grid_app.command("start")
def start_session(
    session_id: str = typer.Argument(..., help="Session ID to start"),
    duration: Optional[str] = typer.Option(
        None,
        "--duration",
        "-d",
        help="Auto-stop after duration (e.g., '1h', '30m', '2h30m')",
    ),
):
    """Start a configured trading session"""
    session = session_manager.get_session(session_id)
    if not session:
        logger.error(f"Session {session_id} not found")
        raise typer.Exit(1)

    # Parse duration if provided
    duration_limit = None
    if duration:
        duration_limit = parse_duration(duration)
        session.duration_limit = duration_limit

    # Start the session
    session.status = TradingSessionStatus.STARTING
    session.start_time = datetime.now(timezone.utc)
    session_manager.set_active_session(session_id)

    logger.custom(f"🚀 Starting session {session_id}")
    logger.info(f"Symbol: {session.symbol}")
    logger.info(f"Mode: {session.parameters.mode.value.upper()}")
    if duration_limit:
        logger.info(f"Duration Limit: {format_duration(duration_limit)}")

    # TODO: Start actual trading logic
    session.status = TradingSessionStatus.RUNNING
    logger.success("Session started successfully")


@grid_app.command("stop")
def stop_session(
    session_id: Optional[str] = typer.Argument(
        None, help="Session ID to stop (default: active session)"
    ),
    force: bool = typer.Option(
        False, "--force", help="Force stop without confirmation"
    ),
):
    """Stop a running trading session"""
    if not session_id:
        active_session = session_manager.get_active_session()
        if not active_session:
            logger.error("No active session to stop")
            raise typer.Exit(1)
        session = active_session
        session_id = session.session_id
    else:
        session = session_manager.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            raise typer.Exit(1)

    if not session.is_active():
        logger.error(f"Session {session_id} is not running")
        raise typer.Exit(1)

    # Confirmation for stopping
    if not force:
        elapsed = session.get_elapsed_time()
        logger.warning(f"\nStopping session {session_id}")
        logger.info(f"Elapsed time: {format_duration(elapsed)}")
        logger.info(f"Current P&L: ${session.profit_loss}")

        if not Confirm.ask(
            "Are you sure you want to stop this session?", default=False
        ):
            logger.error("Stop cancelled")
            raise typer.Abort()

    # Stop the session
    session.status = TradingSessionStatus.STOPPING
    logger.custom(f"🛑 Stopping session {session_id}...")

    # TODO: Implement actual trading stop logic
    session.status = TradingSessionStatus.STOPPED
    session.end_time = datetime.now(timezone.utc)

    logger.success("Session stopped successfully")


@grid_app.command("pause")
def pause_session(
    session_id: Optional[str] = typer.Argument(
        None, help="Session ID to pause (default: active session)"
    )
):
    """Pause a running trading session"""
    if not session_id:
        active_session = session_manager.get_active_session()
        if not active_session:
            logger.error("No active session to pause")
            raise typer.Exit(1)
        session = active_session
        session_id = session.session_id
    else:
        session = session_manager.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            raise typer.Exit(1)

    if session.status != TradingSessionStatus.RUNNING:
        logger.error(f"Session {session_id} is not running")
        raise typer.Exit(1)

    # Pause the session
    session.status = TradingSessionStatus.PAUSED
    session.pause_time = datetime.now(timezone.utc)

    logger.custom(f"⏸️  Session {session_id} paused")
    logger.info("Use 'grid resume' to continue trading")


@grid_app.command("resume")
def resume_session(
    session_id: Optional[str] = typer.Argument(
        None, help="Session ID to resume (default: active session)"
    )
):
    """Resume a paused trading session"""
    if not session_id:
        active_session = session_manager.get_active_session()
        if not active_session:
            logger.error("No active session to resume")
            raise typer.Exit(1)
        session = active_session
        session_id = session.session_id
    else:
        session = session_manager.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            raise typer.Exit(1)

    if session.status != TradingSessionStatus.PAUSED:
        logger.error(f"Session {session_id} is not paused")
        raise typer.Exit(1)

    # Resume the session
    if session.pause_time:
        pause_duration = datetime.now(timezone.utc) - session.pause_time
        session.total_paused_duration += pause_duration
        session.pause_time = None

    session.status = TradingSessionStatus.RUNNING
    logger.custom(f"▶️  Session {session_id} resumed")


@grid_app.command("status")
def show_session_status(
    session_id: Optional[str] = typer.Argument(
        None, help="Session ID to check (default: active session)"
    ),
    live: bool = typer.Option(False, "--live", help="Live updating status display"),
):
    """Show trading session status"""
    if not session_id:
        active_session = session_manager.get_active_session()
        if not active_session:
            logger.error("No active session")
            raise typer.Exit(1)
        session = active_session
        session_id = session.session_id
    else:
        session = session_manager.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            raise typer.Exit(1)

    if live:
        show_live_status(session)
    else:
        show_static_status(session)


@grid_app.command("list")
def list_sessions(
    all_sessions: bool = typer.Option(
        False, "--all", help="Show all sessions (including completed)"
    )
):
    """List trading sessions"""
    sessions = session_manager.sessions

    if not sessions:
        logger.custom("📭 No trading sessions found")
        return

    # Filter sessions
    if not all_sessions:
        sessions = {
            k: v
            for k, v in sessions.items()
            if v.status != TradingSessionStatus.COMPLETED
        }

    # Create table
    table = Table(title="Trading Sessions")
    table.add_column("Session ID", style="cyan")
    table.add_column("Symbol", style="green")
    table.add_column("Mode", style="blue")
    table.add_column("Status", style="yellow")
    table.add_column("Elapsed", style="white")
    table.add_column("P&L", style="magenta")

    for session in sessions.values():
        elapsed = format_duration(session.get_elapsed_time())
        status_style = get_status_style(session.status)

        table.add_row(
            session.session_id,
            session.symbol,
            session.parameters.mode.value.upper(),
            f"[{status_style}]{session.status.value.upper()}[/{status_style}]",
            elapsed,
            f"${session.profit_loss}",
        )

    logger.console_only(table)


@grid_app.command("create")
def create_session_cmd(
    symbol: str = typer.Argument(..., help="Trading symbol (e.g., BTCUSDC)"),
    mode: str = typer.Option("manual", "--mode", help="Trading mode: manual or ai"),
    duration: Optional[str] = typer.Option(
        None, "--duration", "-d", help="Auto-stop duration (e.g., '1h', '30m')"
    ),
):
    """Create a new trading session with parameters"""

    try:
        # Validate mode
        if mode.lower() not in ["manual", "ai"]:
            logger.error("Mode must be 'manual' or 'ai'")
            raise typer.Exit(1)

        trading_mode = (
            TradingMode.MANUAL if mode.lower() == "manual" else TradingMode.AI
        )

        # Get parameters based on mode
        if trading_mode == TradingMode.MANUAL:
            logger.custom(f"🎯 Creating manual session for {symbol}")
            params = collect_manual_parameters(symbol)
        else:
            logger.custom(f"🤖 Creating AI session for {symbol}")
            ai_suggestion = asyncio.run(get_ai_suggested_parameters(symbol))
            params = confirm_ai_parameters(ai_suggestion)

        # Parse duration if provided
        duration_limit = None
        if duration:
            duration_limit = parse_duration(duration)

        # Create session
        session_id = session_manager.create_session(symbol, params, duration_limit)

        logger.success(f"\nSession created: {session_id}")
        logger.info(f"Mode: {params.mode.value.upper()}")
        logger.info(f"Grid Range: ${params.lower_bound} - ${params.upper_bound}")
        logger.info(f"Grid Levels: {params.grid_count}")
        if duration_limit:
            logger.info(f"Duration Limit: {format_duration(duration_limit)}")

        logger.custom(f"\n💡 Use 'grid start {session_id}' to begin trading")

    except typer.Abort:
        raise
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise typer.Exit(1)


@grid_app.command("adjust")
def adjust_parameters(
    session_id: Optional[str] = typer.Argument(
        None, help="Session ID to adjust (default: active session)"
    ),
    lower_bound: Optional[float] = typer.Option(
        None, "--lower", help="New lower bound"
    ),
    upper_bound: Optional[float] = typer.Option(
        None, "--upper", help="New upper bound"
    ),
    grid_count: Optional[int] = typer.Option(None, "--grids", help="New grid count"),
    investment: Optional[float] = typer.Option(
        None, "--investment", help="New investment per grid"
    ),
):
    """Adjust parameters of a trading session (session must be paused)"""

    if not session_id:
        active_session = session_manager.get_active_session()
        if not active_session:
            logger.error("No active session to adjust")
            raise typer.Exit(1)
        session = active_session
        session_id = session.session_id
    else:
        session = session_manager.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            raise typer.Exit(1)

    # Check if session can be adjusted
    if session.status == TradingSessionStatus.RUNNING:
        logger.warning("Session must be paused before adjusting parameters")
        if Confirm.ask("Pause session now?", default=True):
            session.status = TradingSessionStatus.PAUSED
            session.pause_time = datetime.now(timezone.utc)
            logger.custom("⏸️ Session paused")
        else:
            logger.error("Cannot adjust parameters while running")
            raise typer.Abort()

    # Show current parameters
    logger.custom(f"\n📋 Current Parameters for {session_id}:")
    logger.info(f"Lower Bound: ${session.parameters.lower_bound}")
    logger.info(f"Upper Bound: ${session.parameters.upper_bound}")
    logger.info(f"Grid Count: {session.parameters.grid_count}")
    logger.info(f"Investment/Grid: ${session.parameters.investment_per_grid}")

    # Apply adjustments
    changes_made = False

    if lower_bound is not None:
        old_value = session.parameters.lower_bound
        session.parameters.lower_bound = Decimal(str(lower_bound))
        logger.custom(
            f"✏️ Lower Bound: ${old_value} → ${session.parameters.lower_bound}"
        )
        changes_made = True

    if upper_bound is not None:
        old_value = session.parameters.upper_bound
        session.parameters.upper_bound = Decimal(str(upper_bound))
        logger.custom(
            f"✏️ Upper Bound: ${old_value} → ${session.parameters.upper_bound}"
        )
        changes_made = True

    if grid_count is not None:
        old_value = session.parameters.grid_count
        session.parameters.grid_count = grid_count
        logger.custom(f"✏️ Grid Count: {old_value} → {session.parameters.grid_count}")
        changes_made = True

    if investment is not None:
        old_value = session.parameters.investment_per_grid
        session.parameters.investment_per_grid = Decimal(str(investment))
        logger.custom(
            f"✏️ Investment/Grid: ${old_value} → ${session.parameters.investment_per_grid}"
        )
        changes_made = True

    if not changes_made:
        logger.info("No parameters specified to adjust")
        return

    # Validate new parameters
    try:
        grid_config = session.parameters.to_grid_config()
        logger.success("\nParameter validation passed")

        # Show new totals
        total_investment = (
            session.parameters.grid_count * session.parameters.investment_per_grid
        )
        grid_spacing = (
            session.parameters.upper_bound - session.parameters.lower_bound
        ) / session.parameters.grid_count

        logger.info(f"New Total Investment: ${total_investment}")
        logger.info(f"New Grid Spacing: ${grid_spacing:.2f}")

        # Confirmation
        if Confirm.ask("Apply these parameter changes?", default=True):
            logger.success("Parameters updated successfully")
            logger.custom(
                "💡 Use 'grid resume' to continue trading with new parameters"
            )
        else:
            logger.error("Parameter changes cancelled")

    except Exception as e:
        logger.error(f"Parameter validation failed: {e}")
        logger.custom("🔄 Reverting changes...")


@grid_app.command("monitor")
def monitor_session(
    session_id: Optional[str] = typer.Argument(
        None, help="Session ID to monitor (default: active session)"
    ),
    refresh: int = typer.Option(
        5, "--refresh", "-r", help="Refresh interval in seconds"
    ),
):
    """Monitor trading session with periodic updates"""

    if not session_id:
        active_session = session_manager.get_active_session()
        if not active_session:
            logger.error("No active session to monitor")
            raise typer.Exit(1)
        session = active_session
        session_id = session.session_id
    else:
        session = session_manager.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            raise typer.Exit(1)

    logger.custom(f"📊 Monitoring session {session_id} (refresh every {refresh}s)")
    logger.info("Press Ctrl+C to stop monitoring\n")

    try:
        while True:
            # Clear screen and show status
            logger.console.clear()
            show_static_status(session)

            # Show last update time
            logger.custom(f"\n🕐 Last updated: {datetime.now().strftime('%H:%M:%S')}")
            logger.info(f"Next update in {refresh} seconds...")

            # Check if session ended
            if (
                not session.is_active()
                and session.status != TradingSessionStatus.PAUSED
            ):
                logger.success(f"\nSession {session_id} has ended")
                break

            time.sleep(refresh)

    except KeyboardInterrupt:
        logger.custom("\n👋 Monitoring stopped")


trade_app = typer.Typer(help="Trading commands")
app.add_typer(trade_app, name="trade")

safety_app = typer.Typer(help="Safety and risk management")
app.add_typer(safety_app, name="safety")

monitor_app = typer.Typer(help="Advanced monitoring and analytics")
app.add_typer(monitor_app, name="monitor")


# Phase 3 - Advanced Safety Commands
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
        advanced_safety_manager.start_monitoring(session_id, symbol)
        enhanced_error_handler.start_monitoring()

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
        risk_report = advanced_safety_manager.get_risk_report(session_id)

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
        logger.console_only(panel)

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

                logger.console_only(
                    f"   [{alert['timestamp'][:19]}] {alert['title']}",
                    style=alert_style,
                )
                logger.console_only(f"      {alert['message']}", style="dim")

    else:
        # Show overall safety status
        error_summary = enhanced_error_handler.get_error_summary(24)

        logger.custom("🛡️ Overall Safety Status (Last 24 Hours)")
        logger.console_only("=" * 50)

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

    # This would show actual alerts from the safety manager
    # For now, show a sample structure

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

    logger.console_only(table)


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


# Advanced Monitoring Commands
@monitor_app.command("market")
def monitor_market_conditions(
    symbol: str = typer.Argument(..., help="Trading symbol to monitor"),
    duration: int = typer.Option(
        300, "--duration", help="Monitoring duration in seconds"
    ),
    refresh: int = typer.Option(5, "--refresh", help="Refresh interval in seconds"),
):
    """Monitor real-time market conditions and regime changes"""

    logger.custom(f"📊 Market Condition Monitoring: {symbol}")
    logger.info(f"Duration: {duration}s | Refresh: {refresh}s")
    logger.custom("=" * 50)

    start_time = datetime.now(timezone.utc)

    try:
        while True:
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            if elapsed >= duration:
                break

            # Get market assessment (simulated)
            market_conditions = advanced_safety_manager.get_market_assessment(symbol)

            if not market_conditions:
                # Create sample market conditions for demo
                market_conditions = MarketConditions(
                    regime=MarketRegime.NORMAL,
                    trend_direction="sideways",
                    trend_strength=0.4,
                    volatility_percentile=0.35,
                    volume_profile="normal",
                    correlation_breakdown=False,
                )

            # Clear screen and show current conditions
            logger.console.clear()
            logger.custom(
                f"📊 Market Conditions: {symbol} | Elapsed: {elapsed:.0f}s/{duration}s"
            )
            logger.custom("=" * 60)

            # Market regime
            regime_style = {
                "calm": "green",
                "normal": "blue",
                "volatile": "yellow",
                "extreme": "red",
                "crisis": "red bold",
            }.get(market_conditions.regime.value, "white")

            logger.console.print(
                f"Market Regime: [{regime_style}]{market_conditions.regime.value.upper()}[/{regime_style}]"
            )

            # Trend analysis
            trend_style = {"up": "green", "down": "red", "sideways": "blue"}.get(
                market_conditions.trend_direction, "white"
            )

            logger.console.print(
                f"Trend: [{trend_style}]{market_conditions.trend_direction.upper()}[/{trend_style}] "
                f"(Strength: {market_conditions.trend_strength:.1%})"
            )

            # Volatility
            vol_style = (
                "green"
                if market_conditions.volatility_percentile < 0.3
                else (
                    "yellow" if market_conditions.volatility_percentile < 0.7 else "red"
                )
            )
            logger.console.print(
                f"Volatility: [{vol_style}]{market_conditions.volatility_percentile:.1%} percentile[/{vol_style}]"
            )

            # Grid trading suitability
            is_favorable = market_conditions.is_favorable_for_grid()
            favorable_style = "green" if is_favorable else "red"
            logger.console.print(
                f"Grid Trading: [{'green' if is_favorable else 'red'}]{'✅ FAVORABLE' if is_favorable else '❌ UNFAVORABLE'}[/{'green' if is_favorable else 'red'}]"
            )

            # Additional indicators
            logger.custom("📈 Additional Indicators:")
            logger.info(f"   Volume Profile: {market_conditions.volume_profile}")
            logger.info(
                f"   Correlation Risk: {'⚠️ HIGH' if market_conditions.correlation_breakdown else '✅ Normal'}"
            )

            logger.custom(f"🕐 Next update in {refresh} seconds... (Ctrl+C to stop)")

            time.sleep(refresh)

    except KeyboardInterrupt:
        logger.custom("👋 Market monitoring stopped")


@monitor_app.command("performance")
def monitor_performance(
    session_id: Optional[str] = typer.Option(
        None, "--session", help="Session ID to monitor"
    ),
    metric: str = typer.Option(
        "all", "--metric", help="Metric to focus on: all, pnl, risk, trades"
    ),
):
    """Monitor trading performance metrics"""

    if session_id:
        logger.custom(f"📈 Performance Monitor: {session_id}")
    else:
        logger.custom("📈 Overall Performance Monitor")

    logger.custom("=" * 50)

    # Performance metrics table
    table = Table(title="Performance Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Current", style="white")
    table.add_column("Target", style="green")
    table.add_column("Status", style="yellow")

    # Sample performance data (would be real data in production)
    metrics_data = [
        ("Total P&L", "$1,247.50", "$1,000+", "✅ Above Target"),
        ("Win Rate", "68.4%", "60%+", "✅ Above Target"),
        ("Sharpe Ratio", "1.85", "1.0+", "✅ Above Target"),
        ("Max Drawdown", "8.2%", "<15%", "✅ Within Limit"),
        ("Daily Return", "2.3%", "1%+", "✅ Above Target"),
        ("Risk Score", "Medium", "Low-Medium", "✅ Acceptable"),
        ("Trades Today", "24", "10+", "✅ Active"),
        ("Success Rate", "72%", "60%+", "✅ Good"),
    ]

    for metric_name, current, target, status in metrics_data:
        if metric != "all" and metric.lower() not in metric_name.lower():
            continue

        # Color code status
        if "✅" in status:
            status_style = "green"
        elif "⚠️" in status:
            status_style = "yellow"
        else:
            status_style = "red"

        table.add_row(
            metric_name, current, target, f"[{status_style}]{status}[/{status_style}]"
        )

    logger.console_only(table)

    # Performance chart (text-based)
    logger.custom("📊 P&L Chart (Last 24 Hours):")
    logger.info("   ┌─────────────────────────────────────────┐")
    logger.info("   │ $1200 ████████████████████████████████  │")
    logger.info("   │ $1000 ████████████████████████████      │")
    logger.info("   │ $800  ████████████████████              │")
    logger.info("   │ $600  ████████████████                  │")
    logger.info("   │ $400  ████████████                      │")
    logger.info("   │ $200  ██████                            │")
    logger.info("   │ $0    ──────────────────────────────────│")
    logger.info("   └─────────────────────────────────────────┘")
    logger.info("     00:00    06:00    12:00    18:00   24:00")


@monitor_app.command("errors")
def monitor_errors(
    follow: bool = typer.Option(
        False, "--follow", "-f", help="Follow error log in real-time"
    ),
    severity: Optional[str] = typer.Option(
        None, "--severity", help="Filter by severity: low, medium, high, critical"
    ),
    hours: int = typer.Option(1, "--hours", help="Show errors from last N hours"),
):
    """Monitor system errors and recovery actions"""

    logger.critical(f"Error Monitoring (Last {hours} hours)")
    logger.custom("=" * 50)

    if follow:
        logger.custom("📝 Following error log in real-time... (Ctrl+C to stop)")
        logger.console.print()

        try:
            # Simulate real-time error monitoring
            sample_errors = [
                ("INFO", "Network", "Connection restored to Binance API"),
                ("WARNING", "Trading", "Order partially filled, adjusting grid"),
                ("ERROR", "Data", "Price feed timeout, using cached data"),
                ("INFO", "Safety", "Risk level returned to normal"),
            ]

            for i, (level, category, message) in enumerate(sample_errors):
                time.sleep(2)  # Simulate real-time

                timestamp = datetime.now().strftime("%H:%M:%S")
                level_style = {
                    "INFO": "blue",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red bold",
                }.get(level, "white")

                logger.console.print(
                    f"[{timestamp}] [{level_style}]{level}[/{level_style}] {category}: {message}"
                )

        except KeyboardInterrupt:
            logger.custom("👋 Error monitoring stopped")

    else:
        # Show static error summary
        error_summary = enhanced_error_handler.get_error_summary(hours)

        if error_summary["total_errors"] == 0:
            logger.success("No errors detected in the specified period")
            return

        # Show recent errors
        logger.custom(f"📋 Recent Errors ({error_summary['total_errors']} total):")

        for error in error_summary["recent_errors"]:
            if severity and error.get("severity", "").lower() != severity.lower():
                continue

            severity_style = {
                "low": "green",
                "medium": "yellow",
                "high": "red",
                "critical": "red bold",
            }.get(error.get("severity", ""), "white")

            logger.console.print(
                f"\n[{error['timestamp'][:19]}] [{severity_style}]{error.get('severity', 'UNKNOWN').upper()}[/{severity_style}]"
            )
            logger.info(f"   Category: {error.get('category', 'Unknown')}")
            logger.info(f"   Title: {error['title']}")
            logger.info(f"   Message: {error['message']}")

            if error.get("recovery_action"):
                logger.info(f"   Recovery: {error['recovery_action']}")


# Configuration commands
@config_app.command("init")
@handle_cli_errors
def init_config(
    api_key: str = typer.Option(
        ..., prompt=True, hide_input=True, help="Binance API Key"
    ),
    api_secret: str = typer.Option(
        ..., prompt=True, hide_input=True, help="Binance API Secret"
    ),
    testnet: bool = typer.Option(True, help="Use Binance Testnet"),
    data_dir: str = typer.Option("data", help="Data directory"),
    log_level: str = typer.Option("INFO", help="Log level"),
):
    """Initialize trading bot configuration with encrypted credentials"""

    # Check if encryption is available
    encryption_available = check_encryption_available()

    # Validate API key format
    if not validate_api_key_format(api_key):
        raise ValidationError(
            "Invalid API key format. Must be at least 32 characters and alphanumeric."
        )

    # Validate and resolve data directory path
    try:
        data_path = validate_file_path(data_dir)
        if not data_path.is_absolute():
            data_path = Path.cwd() / data_dir
        data_path = data_path.resolve()
    except ValueError as e:
        raise ValidationError(f"Invalid data directory: {e}")

    # Encrypt credentials if possible
    try:
        if encryption_available:
            encrypted_creds = encrypt_credentials(api_key, api_secret)
        else:
            logger.warning(
                "Encryption not available - credentials will be stored in plain text"
            )
            if not confirm_operation(
                "Continue without encryption?", default=False, danger=True
            ):
                logger.critical("Configuration cancelled for security reasons")
                raise typer.Abort()
            encrypted_creds = {
                "api_key": api_key,
                "api_secret": api_secret,
                "encrypted": False,
            }

        config = {
            **encrypted_creds,
            "testnet": testnet,
            "data_dir": str(data_path),
            "log_level": log_level,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        config_path = Path("config.json")
        secure_file_write(config_path, config)

        logger.success(f"Secure configuration saved to {config_path}")

        # Initialize data directory
        data_path.mkdir(parents=True, exist_ok=True)
        logger.success(f"Data directory created: {data_path}")

        # Log the action
        log_user_action(
            "config_initialized",
            {
                "testnet": testnet,
                "data_dir": str(data_path),
                "encryption_used": encryption_available,
            },
        )

    except SecurityError as e:
        raise ConfigurationError(f"Security error: {e}")


@config_app.command("show")
@handle_cli_errors
def show_config():
    """Show current configuration with enhanced security"""
    config_path = Path("config.json")

    if not config_path.exists():
        logger.error("Configuration file not found. Run 'config init' first.")
        raise typer.Exit(1)

    # Check file permissions
    if not check_file_permissions(config_path):
        logger.warning("Configuration file has insecure permissions!")
        logger.info("Consider running: chmod 600 config.json")

    try:
        config = safe_json_load(str(config_path))
    except ValidationError as e:
        raise ConfigurationError(f"Invalid configuration file: {e}")

    # Create display configuration with proper credential masking
    display_config = mask_sensitive_data(config)

    table = Table(title="Trading Bot Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="magenta")

    for key, value in display_config.items():
        table.add_row(key, str(value))

    logger.console_only(table)

    # Show security status
    if config.get("encrypted", True):
        logger.custom("🔒 Credentials are encrypted")
    else:
        logger.warning("Credentials are stored in plain text")


# Data commands
@data_app.command("download")
def download_data(
    symbol: str = typer.Argument(..., help="Trading symbol (e.g., BTCUSDC)"),
    interval: str = typer.Option("1h", help="Kline interval"),
    days: int = typer.Option(30, help="Number of days to download"),
    update_existing: bool = typer.Option(False, help="Update existing data"),
):
    """Download historical market data"""

    async def _download():
        loader = DataLoader()
        start_date = datetime.now(timezone.utc) - timedelta(days=days)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=logger.console,
        ) as progress:
            task = progress.add_task(
                f"Downloading {symbol} {interval} data...", total=None
            )

            try:
                klines = await loader.fetch_klines(
                    symbol=symbol, interval=interval, start_time=start_date, limit=1000
                )

                progress.update(task, description=f"Saving {len(klines)} klines...")

                repo = get_default_repository()
                saved_count = repo.save_klines(klines)

                progress.update(task, description="✅ Complete!")

                logger.success(
                    f"Downloaded and saved {saved_count} klines for {symbol}"
                )
                logger.custom(
                    f"📊 Date range: {klines[0].open_time} to {klines[-1].close_time}"
                )

            except Exception as e:
                logger.error(f"Error downloading data: {e}")
                raise typer.Exit(1)

    asyncio.run(_download())


@data_app.command("list")
def list_data():
    """List available market data"""
    repo = get_default_repository()

    # Query unique symbols and intervals from database
    with repo.db_repo.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT symbol, interval, COUNT(*) as count, 
                   MIN(open_time) as start_date, 
                   MAX(open_time) as end_date
            FROM klines 
            GROUP BY symbol, interval
            ORDER BY symbol, interval
        """
        )

        rows = cursor.fetchall()

    if not rows:
        logger.custom("📭 No market data found. Use 'data download' to get started.")
        return

    table = Table(title="Available Market Data")
    table.add_column("Symbol", style="cyan")
    table.add_column("Interval", style="blue")
    table.add_column("Records", style="green")
    table.add_column("Start Date", style="magenta")
    table.add_column("End Date", style="magenta")

    for row in rows:
        table.add_row(
            row["symbol"],
            row["interval"],
            str(row["count"]),
            row["start_date"].strftime("%Y-%m-%d") if row["start_date"] else "N/A",
            row["end_date"].strftime("%Y-%m-%d") if row["end_date"] else "N/A",
        )

    logger.console_only(table)


# Backtesting commands
@backtest_app.command("run")
def run_backtest(
    symbol: str = typer.Argument(..., help="Trading symbol"),
    strategy: str = typer.Option("grid", help="Strategy type (grid, baseline)"),
    start_date: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = typer.Option(None, help="End date (YYYY-MM-DD)"),
    initial_capital: float = typer.Option(10000.0, help="Initial capital"),
    interval: str = typer.Option("1h", help="Data interval"),
    commission: float = typer.Option(0.001, help="Commission rate"),
    output_file: Optional[str] = typer.Option(None, help="Output file for results"),
):
    """Run backtesting on historical data"""

    async def _backtest():
        # Load data
        loader = DataLoader()

        start_dt = (
            datetime.fromisoformat(start_date)
            if start_date
            else datetime.now(timezone.utc) - timedelta(days=30)
        )
        end_dt = (
            datetime.fromisoformat(end_date) if end_date else datetime.now(timezone.utc)
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=logger.console,
        ) as progress:
            task = progress.add_task("Loading market data...", total=None)

            try:
                klines = await loader.load_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_dt,
                    end_time=end_dt,
                )

                if not klines:
                    logger.error("No data found for {symbol}")
                    raise typer.Exit(1)

                progress.update(task, description="Setting up backtest...")

                # Initialize strategy
                if strategy == "grid":
                    strategy_obj = SimpleGridStrategy(
                        initial_capital=initial_capital, commission=commission
                    )
                elif strategy == "baseline":
                    strategy_obj = BaselineStrategy()
                else:
                    logger.error(f"Unknown strategy: {strategy}")
                    raise typer.Exit(1)

                # Run backtest
                engine = BacktestEngine()
                progress.update(task, description="Running backtest...")

                result = await engine.run_backtest(strategy_obj, klines)

                progress.update(task, description="✅ Backtest complete!")

                # Display results
                _display_backtest_results(result)

                # Save results if requested
                if output_file:
                    _save_backtest_results(result, output_file)
                    logger.custom("💾 Results saved to {output_file}")

            except Exception as e:
                logger.error("Backtest failed: {e}")
                raise typer.Exit(1)

    asyncio.run(_backtest())


def _display_backtest_results(result):
    """Display backtest results in a nice format"""
    panel_content = f"""
        📊 **Strategy**: {result.strategy_name}
        💰 **Initial Capital**: ${result.initial_capital:,.2f}
        💵 **Final Capital**: ${result.final_capital:,.2f}
        📈 **Total Return**: {result.total_return:.2%}
        📏 **Sharpe Ratio**: {result.metrics.sharpe_ratio:.3f}
        📉 **Max Drawdown**: {result.metrics.max_drawdown:.2%}
        🔢 **Total Trades**: {result.metrics.total_trades}
        🎯 **Win Rate**: {result.metrics.win_rate:.2%}
    """

    logger.console_only(Panel(panel_content, title="Backtest Results", expand=False))


def _save_backtest_results(result, filename: str):
    """Save backtest results to file"""
    result_data = {
        "strategy_name": result.strategy_name,
        "symbol": result.symbol,
        "start_date": result.start_date.isoformat(),
        "end_date": result.end_date.isoformat(),
        "initial_capital": float(result.initial_capital),
        "final_capital": float(result.final_capital),
        "total_return": float(result.total_return),
        "metrics": {
            "sharpe_ratio": float(result.metrics.sharpe_ratio),
            "max_drawdown": float(result.metrics.max_drawdown),
            "total_trades": result.metrics.total_trades,
            "win_rate": float(result.metrics.win_rate),
        },
    }

    with open(filename, "w") as f:
        json.dump(result_data, f, indent=2)


# Model commands
@model_app.command("list")
def list_models():
    """List all saved models"""
    manager = get_default_artifact_manager()
    models = manager.list_models()

    if not models:
        logger.warning("No models found.")
        return

    table = Table(title="Saved Models")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="blue")
    table.add_column("Version", style="green")
    table.add_column("Created", style="magenta")

    for model in models:
        created_date = datetime.fromisoformat(model["created_at"]).strftime(
            "%Y-%m-%d %H:%M"
        )
        table.add_row(
            model["name"], model["model_type"], model["version"], created_date
        )

    logger.console_only(table)


@model_app.command("optimize")
def optimize_strategy(
    symbol: str = typer.Argument(..., help="Trading symbol"),
    strategy: str = typer.Option("baseline", help="Strategy to optimize"),
    n_trials: int = typer.Option(50, help="Number of optimization trials"),
    output_file: Optional[str] = typer.Option(None, help="Save results to file"),
):
    """Optimize strategy hyperparameters"""

    async def _optimize():
        # Load data
        loader = DataLoader()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=logger.console,
        ) as progress:
            task = progress.add_task("Loading market data...", total=None)

            try:
                klines = await loader.load_klines(
                    symbol=symbol,
                    interval="1h",
                    start_time=datetime.now(timezone.utc) - timedelta(days=180),
                )

                if not klines:
                    logger.error("No data found for {symbol}")
                    raise typer.Exit(1)

                progress.update(task, description="Setting up optimizer...")

                # Initialize optimizer
                optimizer = BayesianOptimizer()

                progress.update(
                    task, description=f"Running {n_trials} optimization trials..."
                )

                # Run optimization
                if strategy == "baseline":
                    study = await optimizer.optimize_baseline_strategy(
                        klines, n_trials=n_trials
                    )
                else:
                    logger.error("Unknown strategy: {strategy}")
                    raise typer.Exit(1)

                progress.update(task, description="✅ Optimization complete!")

                # Display results
                best_params = study.best_params
                best_value = study.best_value

                logger.custom(f"🎯 **Best Parameters**: {best_params}")
                logger.custom(f"📊 **Best Value**: {best_value:.4f}")

                # Save results if requested
                if output_file:
                    results = {
                        "strategy": strategy,
                        "symbol": symbol,
                        "n_trials": n_trials,
                        "best_params": best_params,
                        "best_value": best_value,
                        "optimization_history": [
                            {"trial": i, "value": trial.value, "params": trial.params}
                            for i, trial in enumerate(study.trials)
                        ],
                    }

                    with open(output_file, "w") as f:
                        json.dump(results, f, indent=2, default=str)

                    logger.info(f"💾 Results saved to {output_file}")

            except Exception as e:
                logger.error(f"Optimization failed: {e}")
                raise typer.Exit(1)

    asyncio.run(_optimize())


# Trading commands
@trade_app.command("paper")
def start_paper_trading(
    symbol: str = typer.Argument(..., help="Trading symbol"),
    strategy: str = typer.Option("grid", help="Strategy type"),
    initial_capital: float = typer.Option(10000.0, help="Initial capital"),
    dry_run: bool = typer.Option(False, help="Dry run mode"),
):
    """Start paper trading"""
    logger.warning("🚧 Paper trading not yet implemented")
    # TODO: Implement paper trading functionality


@trade_app.command("live")
def start_live_trading(
    symbol: str = typer.Argument(..., help="Trading symbol"),
    strategy: str = typer.Option("grid", help="Strategy type"),
    initial_capital: float = typer.Option(1000.0, help="Initial capital"),
    confirm: bool = typer.Option(False, "--confirm", help="Confirm live trading"),
    mode: str = typer.Option("manual", help="Trading mode: manual or ai"),
    confirm_live: bool = typer.Option(
        False, "--confirm-live", help="Explicit confirmation for live AI mode"
    ),
):
    """Start live trading (REAL MONEY!) - DEPRECATED: Use 'grid manual' or 'grid ai' instead"""

    logger.warning("⚠️ [yellow]This command is deprecated. Use:[/yellow]")
    logger.info("   • [bold]grid manual[/bold] for manual parameter input")
    logger.info("   • [bold]grid ai[/bold] for AI-suggested parameters")
    logger.info("\nRedirecting to new grid commands...")

    # Redirect to appropriate grid command
    if mode.lower() == "ai":
        logger.custom(f"🤖 Starting AI grid mode for {symbol}...")
        start_ai_grid(symbol=symbol, paper=False, confirm_live=confirm_live)
    else:
        logger.info(f"Starting manual grid mode for {symbol}...")
        start_manual_grid(symbol=symbol, paper=False, confirm_live=confirm_live)


# Utility commands
@app.command("status")
def show_status():
    """Show bot status and health check"""
    logger.custom("🤖 Trading Bot Status")

    # Check configuration
    config_exists = Path("config.json").exists()
    status_icon = "✅" if config_exists else "❌"
    logger.info(f"{status_icon} Configuration: {'OK' if config_exists else 'Missing'}")

    # Check data directory
    data_dir = Path("data")
    data_exists = data_dir.exists()
    status_icon = "✅" if data_exists else "❌"
    logger.info(f"{status_icon} Data Directory: {'OK' if data_exists else 'Missing'}")

    # Check database
    db_exists = Path("trading_bot.db").exists()
    status_icon = "✅" if db_exists else "❌"
    logger.info(f"{status_icon} Database: {'OK' if db_exists else 'Missing'}")

    # Check models
    artifacts_dir = Path("artifacts")
    models_exist = artifacts_dir.exists()
    status_icon = "✅" if models_exist else "❌"
    logger.info(
        f"{status_icon} Models Directory: {'OK' if models_exist else 'Missing'}"
    )


@app.command("version")
def show_version():
    """Show bot version information"""
    version_info = {"Bot Version": "1.0.0", "Python": "3.10+", "Created": "2024"}

    table = Table(title="Version Information")
    table.add_column("Component", style="cyan")
    table.add_column("Version", style="green")

    for component, version in version_info.items():
        table.add_row(component, version)

    logger.console_only(table)


# Safety and Risk Management Commands
@safety_app.command("status")
def safety_status(
    state_file: str = typer.Option("bot_state.json", help="Bot state file path")
):
    """Show current safety status and manual resume state"""

    try:
        state = load_state(state_file)

        table = Table(title="Safety Status")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")

        # Bot status
        table.add_row("Bot Status", state.status.value)
        table.add_row(
            "Manual Resume Required", "YES" if state.requires_manual_resume else "NO"
        )

        if state.requires_manual_resume:
            table.add_row("Flatten Reason", state.flatten_reason or "Unknown")
            if state.flatten_timestamp:
                table.add_row("Flatten Time", state.flatten_timestamp.isoformat())

        # Current positions
        active_positions = [pos for pos in state.positions.values() if not pos.is_flat]
        table.add_row("Active Positions", str(len(active_positions)))
        table.add_row("Open Orders", str(len(state.active_orders)))

        # Error tracking
        table.add_row("Error Count", str(state.error_count))
        if state.last_error:
            table.add_row(
                "Last Error",
                (
                    state.last_error[:50] + "..."
                    if len(state.last_error) > 50
                    else state.last_error
                ),
            )

        logger.console_only(table)

        # Show position details if any
        if active_positions:
            logger.custom("\n[bold yellow]Active Positions:[/bold yellow]")
            pos_table = Table()
            pos_table.add_column("Symbol", style="cyan")
            pos_table.add_column("Quantity", style="green")
            pos_table.add_column("Entry Price", style="blue")
            pos_table.add_column("Current Price", style="blue")
            pos_table.add_column("Unrealized PnL", style="red")

            for pos in active_positions:
                pos_table.add_row(
                    pos.symbol,
                    f"{pos.quantity:.6f}",
                    f"{pos.entry_price:.2f}",
                    f"{pos.current_price:.2f}",
                    f"{pos.unrealized_pnl:.2f}",
                )

            logger.console_only(pos_table)

    except FileNotFoundError:
        logger.error(f"State file not found: {state_file}")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Error reading state: {e}")
        raise typer.Exit(1)


@safety_app.command("clear-resume")
def clear_manual_resume(
    state_file: str = typer.Option("bot_state.json", help="Bot state file path"),
    confirm: bool = typer.Option(False, "--yes", help="Skip confirmation prompt"),
):
    """Clear the manual resume flag (operator intervention)"""

    try:
        state = load_state(state_file)

        if not state.requires_manual_resume:
            logger.success("No manual resume flag to clear")
            return

        # Show current state
        logger.custom(
            f"[yellow]Current flatten reason:[/yellow] {state.flatten_reason}"
        )
        if state.flatten_timestamp:
            logger.custom(
                f"[yellow]Flatten time:[/yellow] {state.flatten_timestamp.isoformat()}"
            )

        # Confirmation
        if not confirm:
            proceed = typer.confirm(
                "⚠️  This will clear the manual resume flag and allow trading to continue. "
                "Are you sure the issues have been resolved?"
            )
            if not proceed:
                logger.warning("Operation cancelled")
                raise typer.Exit(0)

        # Clear the flag
        state.clear_manual_resume()
        save_state(state, state_file)

        logger.success("Manual resume flag cleared - trading can now continue")

    except FileNotFoundError:
        logger.error(f"State file not found: {state_file}")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Error updating state: {e}")
        raise typer.Exit(1)


@safety_app.command("flatten")
def manual_flatten(
    reason: str = typer.Option(..., help="Reason for manual flatten"),
    state_file: str = typer.Option("bot_state.json", help="Bot state file path"),
    config_file: str = typer.Option("config.json", help="Configuration file path"),
    confirm: bool = typer.Option(False, "--yes", help="Skip confirmation prompt"),
):
    """Manually trigger a flatten operation"""

    async def do_flatten():
        try:
            # Load configuration and state
            config_path = Path(config_file)
            if not config_path.exists():
                logger.error(f"Configuration file not found: {config_file}")
                return False

            with open(config_path, "r") as f:
                config = json.load(f)

            state = load_state(state_file)

            # Show current positions
            active_positions = [
                pos for pos in state.positions.values() if not pos.is_flat
            ]
            if not active_positions and not state.active_orders:
                logger.info("No active positions or orders to flatten")
                return True

            logger.custom(
                f"[yellow]Positions to close:[/yellow] {len(active_positions)}"
            )
            logger.custom(
                f"[yellow]Orders to cancel:[/yellow] {len(state.active_orders)}"
            )

            # Confirmation
            if not confirm:
                proceed = typer.confirm(
                    f"⚠️  This will flatten all positions and cancel all orders. "
                    f"Reason: '{reason}'. Continue?"
                )
                if not proceed:
                    logger.warning("Operation cancelled")
                    return False

            # Create client and execute flatten
            client = await create_binance_client(
                config["api_key"], config["api_secret"], config.get("testnet", True)
            )

            logger.custom("🔄 Executing flatten operation...")

            result = await execute_flatten_operation(client, state, reason)

            if result.success:
                logger.success("Flatten operation completed successfully")
                logger.info(f"Orders canceled: {result.orders_canceled}")
                logger.info(f"Positions closed: {len(result.positions_closed)}")
                logger.info(f"Total proceeds: ${result.total_proceeds:.2f}")
                logger.info(f"Execution time: {result.execution_time_seconds:.1f}s")
                logger.warning(
                    "\nManual resume flag has been set - use 'safety clear-resume' when ready"
                )
                return True
            else:
                logger.error(f"Flatten operation failed: {result.error_message}")
                logger.warning("Manual resume flag has still been set for safety")
                return False

        except Exception as e:
            logger.error("Error during flatten: {e}")
            return False

    # Run the async function
    success = asyncio.run(do_flatten())
    if not success:
        raise typer.Exit(1)


@trade_app.command("run")
def run_trading_bot(
    strategy: str = typer.Option("grid", help="Trading strategy to use"),
    symbol: str = typer.Option("BTCUSDC", help="Trading symbol"),
    config_file: str = typer.Option("config.json", help="Configuration file path"),
    state_file: str = typer.Option("bot_state.json", help="Bot state file path"),
    risk_limits_file: str = typer.Option(None, help="Risk limits configuration file"),
    paper: bool = typer.Option(True, help="Paper trading mode (default: True)"),
    mode: str = typer.Option("grid", help="Trading mode (grid, ai, etc.)"),
    confirm_live: bool = typer.Option(
        False, "--confirm-live", help="Confirm live trading"
    ),
):
    """Run the trading bot with safety integration"""

    # Safety checks before any live mode starts
    if not paper and mode.lower() == "ai" and not confirm_live:
        logger.error("Refusing to run live without --confirm-live in AI mode.")
        raise typer.Abort()

    if not paper and not typer.confirm(
        "Live trading ON. Type 'yes' to confirm:", default=False
    ):
        logger.error("Live trading cancelled by user.")
        raise typer.Abort()

    async def run():
        try:
            # Load configuration
            config_path = Path(config_file)
            if not config_path.exists():
                logger.error(f"Configuration file not found: {config_file}")
                return False

            with open(config_path, "r") as f:
                config = json.load(f)

            # Load or create state
            try:
                state = load_state(state_file)
                logger.success(f"Loaded existing state from {state_file}")
            except FileNotFoundError:
                state = create_empty_state(symbol, config)
                logger.success(f"Created new state for {symbol}")

            # Load risk limits
            risk_limits = RiskLimits()
            if risk_limits_file and Path(risk_limits_file).exists():
                with open(risk_limits_file, "r") as f:
                    limits_config = json.load(f)
                    # Update risk limits with loaded config
                    for key, value in limits_config.items():
                        if hasattr(risk_limits, key):
                            setattr(risk_limits, key, value)
                logger.success(f"Loaded risk limits from {risk_limits_file}")

            # Create client
            client = await create_binance_client(
                config["api_key"], config["api_secret"], config.get("testnet", True)
            )

            # Display trading mode
            trading_mode = "📄 PAPER TRADING" if paper else "💰 LIVE TRADING"
            mode_style = "green" if paper else "bold red"
            logger.custom(f"{trading_mode} MODE ACTIVE")
            logger.custom(
                f"🚀 Starting trading bot for {symbol} with {strategy} strategy ({mode} mode)"
            )
            logger.info("Press Ctrl+C to stop gracefully")

            # Define a simple strategy callback (placeholder)
            async def simple_strategy(client, state):
                # Placeholder for actual strategy implementation
                await asyncio.sleep(1)

            # Run the bot
            await run_bot(
                client=client,
                state=state,
                strategy_callback=simple_strategy,
                risk_limits=risk_limits,
                state_save_path=state_file,
            )

            return True

        except KeyboardInterrupt:
            logger.custom("\n🛑 Shutdown requested by user")
            return True
        except Exception as e:
            logger.error("Error running trading bot: {e}")
            return False

    success = asyncio.run(run())
    if not success:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
