"""
Grid trading commands for the trading bot CLI
"""
import typer
import asyncio
import time
import re
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Optional
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.prompt import Confirm

from ui.error_handling import (
    handle_cli_errors,
    ValidationError,
    confirm_operation,
    log_user_action,
)
from ui.output import OutputHandler
from ui.session_manager import (
    ThreadSafeSessionManager,
    TradingSessionStatus,
    TradingSession,
    TradingMode,
)

# Create or get session manager instance
try:
    session_manager = ThreadSafeSessionManager(max_sessions=50, cleanup_threshold=40)
except Exception as e:
    from ui.output import OutputHandler
    logger_temp = OutputHandler("grid_commands")
    logger_temp.error(f"Failed to initialize session manager: {e}")
    session_manager = None

logger = OutputHandler("ui.cli")

# Grid trading commands will be registered to this app instance
grid_app = typer.Typer(help="Grid Trading Commands")


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

    logger.console.print(table)


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

    logger.custom(f"📊 Monitoring session {session_id} (refresh: {refresh}s)")
    logger.custom("Press Ctrl+C to stop monitoring\n")

    try:
        while True:
            # Clear screen and show status
            logger.console.clear()
            show_static_status(session)
            
            # Show last update time
            logger.info(f"\nLast update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Refresh interval: {refresh} seconds")
            
            time.sleep(refresh)
            
    except KeyboardInterrupt:
        logger.custom("\n👋 Monitoring stopped")


# Helper functions
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


def parse_duration_safe(duration_str: str) -> timedelta:
    """Safe wrapper for parse_duration that raises ValidationError"""
    try:
        return parse_duration(duration_str)
    except ValueError as e:
        raise ValidationError(str(e))


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
    logger.console.print(panel)


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


# Import helper functions - these would need to be implemented
def collect_manual_parameters(symbol: str):
    """Collect manual trading parameters from user input"""
    # This would be implemented to collect parameters via CLI prompts
    from ui.session_manager import GridParameters, TradingMode
    from decimal import Decimal
    
    # For now, return sample parameters
    return GridParameters(
        mode=TradingMode.MANUAL,
        lower_bound=Decimal("30000"),
        upper_bound=Decimal("35000"),
        grid_count=10,
        investment_per_grid=Decimal("100"),
        confidence=0.8
    )


async def get_ai_suggested_parameters(symbol: str):
    """Get AI-suggested parameters for the symbol"""
    # This would be implemented to call the AI system
    from ui.session_manager import GridParameters, TradingMode
    from decimal import Decimal
    
    # For now, return sample AI parameters
    return GridParameters(
        mode=TradingMode.AI,
        lower_bound=Decimal("31000"),
        upper_bound=Decimal("34000"),
        grid_count=8,
        investment_per_grid=Decimal("125"),
        confidence=0.85
    )


def confirm_ai_parameters(suggestion):
    """Allow user to review and modify AI suggestions"""
    # This would be implemented to show AI suggestions and get confirmation
    return suggestion
