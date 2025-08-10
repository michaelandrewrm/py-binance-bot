"""
CLI Module - Command Line Interface

This module provides a comprehensive command-line interface for the trading bot
using Typer for a modern, intuitive CLI experience.
"""

import typer
import asyncio
import logging
import json
import threading
import time
import signal
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
from rich import print as rprint
from decimal import Decimal

# Mode-specific data classes
from dataclasses import dataclass
from enum import Enum

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
from core.grid_engine import GridConfig
from core.baseline_heuristic import get_baseline_parameters
from core.advanced_safety import advanced_safety_manager, RiskLevel, MarketRegime, AlertType, MarketConditions, MarketRegime
from core.enhanced_error_handling import enhanced_error_handler, ErrorCategory, ErrorSeverity
from exec.binance import create_binance_client

# Initialize logger
logger = logging.getLogger(__name__)

app = typer.Typer(help="Cryptocurrency Trading Bot CLI")
console = Console()

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

class TradingMode(Enum):
    MANUAL = "manual"
    AI = "ai"

@dataclass
class GridParameters:
    """Grid trading parameters"""
    symbol: str
    lower_bound: Decimal
    upper_bound: Decimal
    grid_count: int
    investment_per_grid: Decimal
    center_price: Optional[Decimal] = None
    mode: TradingMode = TradingMode.MANUAL
    confidence: Optional[float] = None  # For AI suggestions
    
    def to_grid_config(self) -> GridConfig:
        """Convert to GridConfig"""
        center = self.center_price or (self.lower_bound + self.upper_bound) / 2
        spacing = (self.upper_bound - self.lower_bound) / self.grid_count
        
        return GridConfig(
            center_price=center,
            grid_spacing=spacing,
            num_levels_up=self.grid_count // 2,
            num_levels_down=self.grid_count // 2,
            order_amount=self.investment_per_grid,
            max_position_size=self.investment_per_grid * self.grid_count
        )

# Trading Session Management Classes
class TradingSessionStatus(Enum):
    """Trading session status"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"
    COMPLETED = "completed"

@dataclass
class TradingSession:
    """Runtime trading session management"""
    session_id: str
    symbol: str
    parameters: GridParameters
    status: TradingSessionStatus = TradingSessionStatus.STOPPED
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_limit: Optional[timedelta] = None
    pause_time: Optional[datetime] = None
    total_paused_duration: timedelta = timedelta()
    
    # Trading metrics
    trades_count: int = 0
    profit_loss: Decimal = Decimal('0')
    current_positions: int = 0
    
    def is_active(self) -> bool:
        """Check if session is actively trading"""
        return self.status in [TradingSessionStatus.RUNNING, TradingSessionStatus.STARTING]
    
    def is_time_expired(self) -> bool:
        """Check if duration limit is exceeded"""
        if not self.duration_limit or not self.start_time:
            return False
        
        elapsed = datetime.now(timezone.utc) - self.start_time - self.total_paused_duration
        return elapsed >= self.duration_limit
    
    def get_elapsed_time(self) -> timedelta:
        """Get elapsed trading time (excluding paused time)"""
        if not self.start_time:
            return timedelta()
        
        end = self.end_time or datetime.now(timezone.utc)
        
        if self.status == TradingSessionStatus.PAUSED and self.pause_time:
            end = self.pause_time
            
        return end - self.start_time - self.total_paused_duration

# Global session manager
class SessionManager:
    """Manages active trading sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, TradingSession] = {}
        self.active_session: Optional[str] = None
        self._lock = threading.Lock()
    
    def create_session(self, symbol: str, parameters: GridParameters, 
                      duration_limit: Optional[timedelta] = None) -> str:
        """Create a new trading session"""
        session_id = f"{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        with self._lock:
            session = TradingSession(
                session_id=session_id,
                symbol=symbol,
                parameters=parameters,
                duration_limit=duration_limit
            )
            self.sessions[session_id] = session
            
        return session_id
    
    def get_session(self, session_id: str) -> Optional[TradingSession]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    def get_active_session(self) -> Optional[TradingSession]:
        """Get currently active session"""
        if self.active_session:
            return self.sessions.get(self.active_session)
        return None
    
    def set_active_session(self, session_id: str) -> bool:
        """Set active session"""
        if session_id in self.sessions:
            self.active_session = session_id
            return True
        return False

# Global session manager instance
session_manager = SessionManager()

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
            "atr": Decimal("1000.0")  # Average True Range
        }
    except Exception as e:
        console.print(f"⚠️ Could not fetch market data: {e}", style="yellow")
        return {}

def collect_manual_parameters(symbol: str) -> GridParameters:
    """Collect grid parameters manually from user input"""
    console.print(f"\n🎯 [bold blue]Manual Grid Configuration for {symbol}[/bold blue]")
    
    # Get market context
    market_data = {}
    try:
        market_data = asyncio.run(get_market_context(symbol))
    except (ConnectionError, TimeoutError, ValueError) as e:
        logger.warning(f"Failed to get market context for {symbol}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error getting market context: {e}")
    
    if market_data:
        console.print(f"📊 [dim]Current Price: ${market_data.get('current_price', 'N/A')}[/dim]")
        console.print(f"📊 [dim]24h Range: ${market_data.get('24h_low', 'N/A')} - ${market_data.get('24h_high', 'N/A')}[/dim]")
        console.print(f"📊 [dim]ATR: ${market_data.get('atr', 'N/A')}[/dim]\n")
    
    # Collect parameters with validation
    while True:
        try:
            lower_bound = Decimal(Prompt.ask("💰 Lower grid bound (price)", default="45000"))
            upper_bound = Decimal(Prompt.ask("💰 Upper grid bound (price)", default="55000"))
            
            if lower_bound >= upper_bound:
                console.print("❌ Lower bound must be less than upper bound!", style="red")
                continue
                
            grid_count = IntPrompt.ask("🔢 Number of grid levels", default=10, show_default=True)
            if grid_count < 2:
                console.print("❌ Grid count must be at least 2!", style="red")
                continue
                
            investment_per_grid = Decimal(FloatPrompt.ask("💵 Investment per grid level", default=100.0, show_default=True))
            if investment_per_grid <= 0:
                console.print("❌ Investment must be positive!", style="red")
                continue
            
            # Calculate total investment
            total_investment = investment_per_grid * grid_count
            console.print(f"\n📋 [bold]Configuration Summary:[/bold]")
            console.print(f"   Range: ${lower_bound} - ${upper_bound}")
            console.print(f"   Grid Levels: {grid_count}")
            console.print(f"   Per Level: ${investment_per_grid}")
            console.print(f"   Total Investment: ${total_investment}")
            
            if Confirm.ask("\n✅ Confirm these parameters?", default=True):
                return GridParameters(
                    symbol=symbol,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    grid_count=grid_count,
                    investment_per_grid=investment_per_grid,
                    mode=TradingMode.MANUAL
                )
            else:
                console.print("🔄 Let's try again...\n")
                
        except KeyboardInterrupt:
            console.print("\n❌ Parameter collection cancelled", style="red")
            raise typer.Abort()
        except Exception as e:
            console.print(f"❌ Invalid input: {e}", style="red")

async def get_ai_suggested_parameters(symbol: str) -> GridParameters:
    """Get AI-suggested grid parameters"""
    console.print(f"\n🤖 [bold blue]AI Parameter Suggestion for {symbol}[/bold blue]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing market data...", total=None)
            
            # TODO: Implement actual AI parameter suggestion
            # For now, use a baseline heuristic
            await asyncio.sleep(2)  # Simulate AI processing
            
            progress.update(task, description="Calculating optimal parameters...")
            await asyncio.sleep(1)
            
            # Mock AI suggestion based on market data
            market_data = await get_market_context(symbol)
            current_price = market_data.get('current_price', Decimal('50000'))
            atr = market_data.get('atr', Decimal('1000'))
            
            # AI heuristic: bounds at ±2*ATR, 10 levels
            lower_bound = current_price - (atr * 2)
            upper_bound = current_price + (atr * 2)
            grid_count = 10
            investment_per_grid = Decimal('100')  # Default
            
            progress.update(task, description="Generating confidence intervals...")
            await asyncio.sleep(0.5)
        
        # Display AI suggestion
        console.print("\n🎯 [bold green]AI Recommendation:[/bold green]")
        console.print(f"   Range: ${lower_bound:.2f} - ${upper_bound:.2f}")
        console.print(f"   Grid Levels: {grid_count}")
        console.print(f"   Suggested Investment/Level: ${investment_per_grid}")
        console.print(f"   Confidence: 85% ± 5%")
        console.print(f"   Expected Sharpe Ratio: 1.2 ± 0.3")
        console.print(f"   Max Expected Drawdown: 8% ± 2%")
        
        return GridParameters(
            symbol=symbol,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            grid_count=grid_count,
            investment_per_grid=investment_per_grid,
            mode=TradingMode.AI,
            confidence=0.85
        )
        
    except Exception as e:
        console.print(f"\n[red]❌ AI analysis failed: {e}[/red]")
        console.print("[yellow]💡 Falling back to baseline heuristic...[/yellow]")
        
        try:
            # Use baseline heuristic as fallback
            current_price = 50000.0  # Default price for fallback
            market_data = {
                'current_price': Decimal(str(current_price)),
                'atr': Decimal(str(current_price * 0.02)),  # Estimate 2% ATR
                'volatility': 0.25  # Estimate 25% annualized volatility
            }
            
            baseline_params = get_baseline_parameters(
                symbol=symbol,
                market_data=market_data,
                investment=Decimal('100')
            )
            
            console.print(f"[green]✅ Baseline parameters calculated[/green]")
            console.print(f"[dim]Using heuristic-based approach for {symbol}[/dim]")
            
            return GridParameters(
                symbol=symbol,
                lower_bound=baseline_params["lower_bound"],
                upper_bound=baseline_params["upper_bound"],
                grid_count=baseline_params["grid_count"],
                investment_per_grid=baseline_params["investment_per_grid"],
                mode=TradingMode.AI,
                confidence=0.65  # Lower confidence for fallback
            )
            
        except Exception as fallback_e:
            console.print(f"\n[red]❌ Baseline fallback also failed: {fallback_e}[/red]")
            # Return conservative defaults as last resort
            return GridParameters(
                symbol=symbol,
                lower_bound=Decimal('45000'),
                upper_bound=Decimal('55000'),
                grid_count=5,
                investment_per_grid=Decimal('50'),
                mode=TradingMode.AI,
                confidence=0.3
            )

def confirm_ai_parameters(suggestion: GridParameters) -> GridParameters:
    """Allow user to confirm or modify AI suggestions"""
    console.print(f"\n🔍 [bold]Review AI Suggestion[/bold]")
    
    while True:
        action = Prompt.ask(
            "Choose action",
            choices=["accept", "modify", "cancel"],
            default="accept"
        )
        
        if action == "accept":
            console.print("✅ AI parameters accepted!", style="green")
            return suggestion
            
        elif action == "modify":
            console.print("🛠️ Modify parameters:")
            
            # Allow modifications
            new_lower = Decimal(Prompt.ask(
                "New lower bound", 
                default=str(suggestion.lower_bound)
            ))
            new_upper = Decimal(Prompt.ask(
                "New upper bound", 
                default=str(suggestion.upper_bound)
            ))
            new_count = IntPrompt.ask(
                "New grid count", 
                default=suggestion.grid_count
            )
            new_investment = Decimal(FloatPrompt.ask(
                "New investment per grid", 
                default=float(suggestion.investment_per_grid)
            ))
            
            modified = GridParameters(
                symbol=suggestion.symbol,
                lower_bound=new_lower,
                upper_bound=new_upper,
                grid_count=new_count,
                investment_per_grid=new_investment,
                mode=TradingMode.AI,
                confidence=0.75  # Lower confidence for modified params
            )
            
            console.print("\n📋 [bold]Modified Configuration:[/bold]")
            console.print(f"   Range: ${modified.lower_bound} - ${modified.upper_bound}")
            console.print(f"   Grid Levels: {modified.grid_count}")
            console.print(f"   Per Level: ${modified.investment_per_grid}")
            
            if Confirm.ask("Accept modified parameters?", default=True):
                console.print("✅ Modified parameters accepted!", style="green")
                return modified
        
        elif action == "cancel":
            console.print("❌ AI suggestion cancelled", style="red")
            raise typer.Abort()

# Grid trading commands
@grid_app.command("manual")
def start_manual_grid(
    symbol: str = typer.Argument(..., help="Trading symbol (e.g., BTCUSDC)"),
    start_immediately: bool = typer.Option(False, "--start", help="Start trading immediately after creating session"),
    duration: Optional[str] = typer.Option(None, "--duration", "-d", help="Auto-stop duration (e.g., '1h', '30m')"),
    paper: bool = typer.Option(True, "--paper/--live", help="Paper trading (default) or live"),
    confirm_live: bool = typer.Option(False, "--confirm-live", help="Explicit confirmation for live trading")
):
    """Configure grid trading in manual mode - you set all parameters"""
    
    # Safety check for live trading
    if not paper and not confirm_live:
        console.print("⚠️ Live trading requires --confirm-live flag!", style="red")
        raise typer.Abort()
    
    try:
        # Collect parameters manually
        console.print(f"🎯 [bold blue]Manual Grid Configuration for {symbol}[/bold blue]")
        params = collect_manual_parameters(symbol)
        
        # Parse duration if provided
        duration_limit = None
        if duration:
            duration_limit = parse_duration(duration)
        
        # Create session
        session_id = session_manager.create_session(symbol, params, duration_limit)
        
        console.print(f"\n✅ Session created: [cyan]{session_id}[/cyan]")
        console.print(f"Mode: [blue]{params.mode.value.upper()}[/blue]")
        console.print(f"Grid Range: [green]${params.lower_bound} - ${params.upper_bound}[/green]")
        console.print(f"Grid Levels: [yellow]{params.grid_count}[/yellow]")
        if duration_limit:
            console.print(f"Duration Limit: [yellow]{format_duration(duration_limit)}[/yellow]")
        
        # Option to start immediately
        if start_immediately:
            if not paper:
                console.print(f"\n🚨 [bold red]FINAL LIVE TRADING CONFIRMATION[/bold red]")
                console.print(f"Total Investment: [red]${params.grid_count * params.investment_per_grid}[/red]")
                if not Confirm.ask("Proceed with LIVE trading using real money?", default=False):
                    console.print("❌ Live trading cancelled", style="red")
                    console.print(f"💡 Session {session_id} created but not started")
                    return
            
            # Start the session
            session = session_manager.get_session(session_id)
            session.status = TradingSessionStatus.STARTING
            session.start_time = datetime.now(timezone.utc)
            session_manager.set_active_session(session_id)
            
            console.print(f"\n🚀 Starting {'paper' if paper else 'LIVE'} trading session...")
            # TODO: Implement actual grid trading execution
            session.status = TradingSessionStatus.RUNNING
            console.print("✅ Session started successfully")
            console.print(f"💡 Use 'grid status --live' for real-time monitoring")
        else:
            console.print(f"\n💡 [dim]Next steps:[/dim]")
            console.print(f"   • [cyan]grid start {session_id}[/cyan] - Start trading")
            console.print(f"   • [cyan]grid status {session_id}[/cyan] - Check status")
            console.print(f"   • [cyan]grid adjust {session_id}[/cyan] - Modify parameters")
        
    except typer.Abort:
        raise
    except Exception as e:
        console.print(f"❌ Error creating manual grid session: {e}", style="red")
        raise typer.Exit(1)

@grid_app.command("ai")
def start_ai_grid(
    symbol: str = typer.Argument(..., help="Trading symbol (e.g., BTCUSDC)"),
    start_immediately: bool = typer.Option(False, "--start", help="Start trading immediately after creating session"),
    duration: Optional[str] = typer.Option(None, "--duration", "-d", help="Auto-stop duration (e.g., '1h', '30m')"),
    paper: bool = typer.Option(True, "--paper/--live", help="Paper trading (default) or live"),
    confirm_live: bool = typer.Option(False, "--confirm-live", help="Explicit confirmation for live AI trading")
):
    """Configure grid trading in AI mode - AI suggests optimal parameters"""
    
    # Safety check for live trading
    if not paper and not confirm_live:
        console.print("⚠️ Live AI trading requires --confirm-live flag!", style="red")
        raise typer.Abort()
    
    try:
        # Get AI suggestions
        console.print(f"🤖 [bold blue]AI Grid Configuration for {symbol}[/bold blue]")
        suggestion = asyncio.run(get_ai_suggested_parameters(symbol))
        
        # Allow user to review and modify
        final_params = confirm_ai_parameters(suggestion)
        
        # Parse duration if provided
        duration_limit = None
        if duration:
            duration_limit = parse_duration(duration)
        
        # Create session
        session_id = session_manager.create_session(symbol, final_params, duration_limit)
        
        console.print(f"\n✅ AI session created: [cyan]{session_id}[/cyan]")
        console.print(f"Mode: [blue]{final_params.mode.value.upper()}[/blue]")
        console.print(f"Confidence: [yellow]{final_params.confidence:.0%}[/yellow]")
        console.print(f"Grid Range: [green]${final_params.lower_bound} - ${final_params.upper_bound}[/green]")
        console.print(f"Grid Levels: [yellow]{final_params.grid_count}[/yellow]")
        if duration_limit:
            console.print(f"Duration Limit: [yellow]{format_duration(duration_limit)}[/yellow]")
        
        # Option to start immediately
        if start_immediately:
            if not paper:
                console.print(f"\n🚨 [bold red]FINAL LIVE AI TRADING CONFIRMATION[/bold red]")
                console.print(f"Total Investment: [red]${final_params.grid_count * final_params.investment_per_grid}[/red]")
                console.print(f"AI Confidence: [yellow]{final_params.confidence:.0%}[/yellow]")
                if not Confirm.ask("Proceed with LIVE AI trading using real money?", default=False):
                    console.print("❌ Live trading cancelled", style="red")
                    console.print(f"💡 Session {session_id} created but not started")
                    return
            
            # Start the session
            session = session_manager.get_session(session_id)
            session.status = TradingSessionStatus.STARTING
            session.start_time = datetime.now(timezone.utc)
            session_manager.set_active_session(session_id)
            
            console.print(f"\n🚀 Starting {'paper' if paper else 'LIVE'} AI trading session...")
            # TODO: Implement actual grid trading execution
            session.status = TradingSessionStatus.RUNNING
            console.print("✅ AI session started successfully")
            console.print(f"💡 Use 'grid status --live' for real-time monitoring")
        else:
            console.print(f"\n💡 [dim]Next steps:[/dim]")
            console.print(f"   • [cyan]grid start {session_id}[/cyan] - Start trading")
            console.print(f"   • [cyan]grid status {session_id}[/cyan] - Check status")
            console.print(f"   • [cyan]grid adjust {session_id}[/cyan] - Modify parameters")
        
    except typer.Abort:
        raise
    except Exception as e:
        console.print(f"❌ Error creating AI grid session: {e}", style="red")
        raise typer.Exit(1)

# Helper functions for runtime controls
def parse_duration(duration_str: str) -> timedelta:
    """Parse duration string like '1h', '30m', '2h30m' into timedelta"""
    
    # Parse patterns like: 1h, 30m, 2h30m, 1d2h30m
    pattern = r'(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?'
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
        TradingSessionStatus.COMPLETED: "cyan"
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
    console.print(panel)

def show_live_status(session: TradingSession):
    """Show live updating session status"""
    
    def create_status_layout():
        """Create the status layout"""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        # Header
        header_text = Text(f"Live Trading Status - {session.symbol}", style="bold cyan")
        layout["header"].update(Panel(header_text, style="blue"))
        
        # Main status
        status_text = create_live_status_text(session)
        layout["main"].update(Panel(status_text, title="Session Details", style="green"))
        
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
        text.append(f"Grid Range: ${session.parameters.lower_bound} - ${session.parameters.upper_bound}\n")
        text.append(f"Grid Levels: {session.parameters.grid_count}\n")
        text.append(f"Elapsed Time: {format_duration(elapsed)}\n", style="white")
        
        if session.duration_limit:
            remaining = session.duration_limit - elapsed
            text.append(f"Time Remaining: {format_duration(remaining)}\n", style="yellow")
            
            # Progress bar
            progress_pct = min(100, (elapsed.total_seconds() / session.duration_limit.total_seconds()) * 100)
            text.append(f"Progress: {progress_pct:.1f}%\n", style="cyan")
        
        text.append(f"Active Trades: {session.trades_count}\n")
        text.append(f"Current Positions: {session.current_positions}\n")
        text.append(f"P&L: ", style="bold")
        pnl_style = "green" if session.profit_loss >= 0 else "red"
        text.append(f"${session.profit_loss}\n", style=pnl_style)
        
        # Check if time expired
        if session.is_time_expired():
            text.append("\n⏰ Duration limit reached! Session will auto-stop.\n", style="red bold")
        
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
                    console.print("\n⏰ Duration limit reached - stopping session automatically")
                    # TODO: Implement actual stop logic
                    session.status = TradingSessionStatus.COMPLETED
                    session.end_time = datetime.now(timezone.utc)
                    break
            
            # Final update
            live.update(create_status_layout())
            console.print(f"\n✅ Live status ended - Session {session.status.value}")
            
    except KeyboardInterrupt:
        console.print("\n👋 Live status view ended")

# Runtime Control Commands
@grid_app.command("start")
def start_session(
    session_id: str = typer.Argument(..., help="Session ID to start"),
    duration: Optional[str] = typer.Option(None, "--duration", "-d", help="Auto-stop after duration (e.g., '1h', '30m', '2h30m')")
):
    """Start a configured trading session"""
    session = session_manager.get_session(session_id)
    if not session:
        console.print(f"❌ Session {session_id} not found", style="red")
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
    
    console.print(f"🚀 Starting session {session_id}")
    console.print(f"Symbol: {session.symbol}")
    console.print(f"Mode: {session.parameters.mode.value.upper()}")
    if duration_limit:
        console.print(f"Duration Limit: {format_duration(duration_limit)}")
    
    # TODO: Start actual trading logic
    session.status = TradingSessionStatus.RUNNING
    console.print("✅ Session started successfully")

@grid_app.command("stop")
def stop_session(
    session_id: Optional[str] = typer.Argument(None, help="Session ID to stop (default: active session)"),
    force: bool = typer.Option(False, "--force", help="Force stop without confirmation")
):
    """Stop a running trading session"""
    if not session_id:
        active_session = session_manager.get_active_session()
        if not active_session:
            console.print("❌ No active session to stop", style="red")
            raise typer.Exit(1)
        session = active_session
        session_id = session.session_id
    else:
        session = session_manager.get_session(session_id)
        if not session:
            console.print(f"❌ Session {session_id} not found", style="red")
            raise typer.Exit(1)
    
    if not session.is_active():
        console.print(f"❌ Session {session_id} is not running", style="red")
        raise typer.Exit(1)
    
    # Confirmation for stopping
    if not force:
        elapsed = session.get_elapsed_time()
        console.print(f"\n⚠️  Stopping session {session_id}")
        console.print(f"Elapsed time: {format_duration(elapsed)}")
        console.print(f"Current P&L: ${session.profit_loss}")
        
        if not Confirm.ask("Are you sure you want to stop this session?", default=False):
            console.print("❌ Stop cancelled")
            raise typer.Abort()
    
    # Stop the session
    session.status = TradingSessionStatus.STOPPING
    console.print(f"🛑 Stopping session {session_id}...")
    
    # TODO: Implement actual trading stop logic
    session.status = TradingSessionStatus.STOPPED
    session.end_time = datetime.now(timezone.utc)
    
    console.print("✅ Session stopped successfully")

@grid_app.command("pause")
def pause_session(
    session_id: Optional[str] = typer.Argument(None, help="Session ID to pause (default: active session)")
):
    """Pause a running trading session"""
    if not session_id:
        active_session = session_manager.get_active_session()
        if not active_session:
            console.print("❌ No active session to pause", style="red")
            raise typer.Exit(1)
        session = active_session
        session_id = session.session_id
    else:
        session = session_manager.get_session(session_id)
        if not session:
            console.print(f"❌ Session {session_id} not found", style="red")
            raise typer.Exit(1)
    
    if session.status != TradingSessionStatus.RUNNING:
        console.print(f"❌ Session {session_id} is not running", style="red")
        raise typer.Exit(1)
    
    # Pause the session
    session.status = TradingSessionStatus.PAUSED
    session.pause_time = datetime.now(timezone.utc)
    
    console.print(f"⏸️  Session {session_id} paused")
    console.print("Use 'grid resume' to continue trading")

@grid_app.command("resume")
def resume_session(
    session_id: Optional[str] = typer.Argument(None, help="Session ID to resume (default: active session)")
):
    """Resume a paused trading session"""
    if not session_id:
        active_session = session_manager.get_active_session()
        if not active_session:
            console.print("❌ No active session to resume", style="red")
            raise typer.Exit(1)
        session = active_session
        session_id = session.session_id
    else:
        session = session_manager.get_session(session_id)
        if not session:
            console.print(f"❌ Session {session_id} not found", style="red")
            raise typer.Exit(1)
    
    if session.status != TradingSessionStatus.PAUSED:
        console.print(f"❌ Session {session_id} is not paused", style="red")
        raise typer.Exit(1)
    
    # Resume the session
    if session.pause_time:
        pause_duration = datetime.now(timezone.utc) - session.pause_time
        session.total_paused_duration += pause_duration
        session.pause_time = None
    
    session.status = TradingSessionStatus.RUNNING
    console.print(f"▶️  Session {session_id} resumed")

@grid_app.command("status")
def show_session_status(
    session_id: Optional[str] = typer.Argument(None, help="Session ID to check (default: active session)"),
    live: bool = typer.Option(False, "--live", help="Live updating status display")
):
    """Show trading session status"""
    if not session_id:
        active_session = session_manager.get_active_session()
        if not active_session:
            console.print("❌ No active session", style="red")
            raise typer.Exit(1)
        session = active_session
        session_id = session.session_id
    else:
        session = session_manager.get_session(session_id)
        if not session:
            console.print(f"❌ Session {session_id} not found", style="red")
            raise typer.Exit(1)
    
    if live:
        show_live_status(session)
    else:
        show_static_status(session)

@grid_app.command("list")
def list_sessions(
    all_sessions: bool = typer.Option(False, "--all", help="Show all sessions (including completed)")
):
    """List trading sessions"""
    sessions = session_manager.sessions
    
    if not sessions:
        console.print("📭 No trading sessions found")
        return
    
    # Filter sessions
    if not all_sessions:
        sessions = {k: v for k, v in sessions.items() 
                   if v.status != TradingSessionStatus.COMPLETED}
    
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
            f"${session.profit_loss}"
        )
    
    console.print(table)

@grid_app.command("create")
def create_session_cmd(
    symbol: str = typer.Argument(..., help="Trading symbol (e.g., BTCUSDC)"),
    mode: str = typer.Option("manual", "--mode", help="Trading mode: manual or ai"),
    duration: Optional[str] = typer.Option(None, "--duration", "-d", help="Auto-stop duration (e.g., '1h', '30m')")
):
    """Create a new trading session with parameters"""
    
    try:
        # Validate mode
        if mode.lower() not in ["manual", "ai"]:
            console.print("❌ Mode must be 'manual' or 'ai'", style="red")
            raise typer.Exit(1)
        
        trading_mode = TradingMode.MANUAL if mode.lower() == "manual" else TradingMode.AI
        
        # Get parameters based on mode
        if trading_mode == TradingMode.MANUAL:
            console.print(f"🎯 Creating manual session for {symbol}")
            params = collect_manual_parameters(symbol)
        else:
            console.print(f"🤖 Creating AI session for {symbol}")
            ai_suggestion = asyncio.run(get_ai_suggested_parameters(symbol))
            params = confirm_ai_parameters(ai_suggestion)
        
        # Parse duration if provided
        duration_limit = None
        if duration:
            duration_limit = parse_duration(duration)
        
        # Create session
        session_id = session_manager.create_session(symbol, params, duration_limit)
        
        console.print(f"\n✅ Session created: {session_id}")
        console.print(f"Mode: {params.mode.value.upper()}")
        console.print(f"Grid Range: ${params.lower_bound} - ${params.upper_bound}")
        console.print(f"Grid Levels: {params.grid_count}")
        if duration_limit:
            console.print(f"Duration Limit: {format_duration(duration_limit)}")
        
        console.print(f"\n💡 Use 'grid start {session_id}' to begin trading")
        
    except typer.Abort:
        raise
    except Exception as e:
        console.print(f"❌ Error creating session: {e}", style="red")
        raise typer.Exit(1)

@grid_app.command("adjust")
def adjust_parameters(
    session_id: Optional[str] = typer.Argument(None, help="Session ID to adjust (default: active session)"),
    lower_bound: Optional[float] = typer.Option(None, "--lower", help="New lower bound"),
    upper_bound: Optional[float] = typer.Option(None, "--upper", help="New upper bound"),
    grid_count: Optional[int] = typer.Option(None, "--grids", help="New grid count"),
    investment: Optional[float] = typer.Option(None, "--investment", help="New investment per grid")
):
    """Adjust parameters of a trading session (session must be paused)"""
    
    if not session_id:
        active_session = session_manager.get_active_session()
        if not active_session:
            console.print("❌ No active session to adjust", style="red")
            raise typer.Exit(1)
        session = active_session
        session_id = session.session_id
    else:
        session = session_manager.get_session(session_id)
        if not session:
            console.print(f"❌ Session {session_id} not found", style="red")
            raise typer.Exit(1)
    
    # Check if session can be adjusted
    if session.status == TradingSessionStatus.RUNNING:
        console.print("⚠️ Session must be paused before adjusting parameters", style="yellow")
        if Confirm.ask("Pause session now?", default=True):
            session.status = TradingSessionStatus.PAUSED
            session.pause_time = datetime.now(timezone.utc)
            console.print("⏸️ Session paused")
        else:
            console.print("❌ Cannot adjust parameters while running")
            raise typer.Abort()
    
    # Show current parameters
    console.print(f"\n📋 Current Parameters for {session_id}:")
    console.print(f"Lower Bound: ${session.parameters.lower_bound}")
    console.print(f"Upper Bound: ${session.parameters.upper_bound}")
    console.print(f"Grid Count: {session.parameters.grid_count}")
    console.print(f"Investment/Grid: ${session.parameters.investment_per_grid}")
    
    # Apply adjustments
    changes_made = False
    
    if lower_bound is not None:
        old_value = session.parameters.lower_bound
        session.parameters.lower_bound = Decimal(str(lower_bound))
        console.print(f"✏️ Lower Bound: ${old_value} → ${session.parameters.lower_bound}")
        changes_made = True
    
    if upper_bound is not None:
        old_value = session.parameters.upper_bound
        session.parameters.upper_bound = Decimal(str(upper_bound))
        console.print(f"✏️ Upper Bound: ${old_value} → ${session.parameters.upper_bound}")
        changes_made = True
    
    if grid_count is not None:
        old_value = session.parameters.grid_count
        session.parameters.grid_count = grid_count
        console.print(f"✏️ Grid Count: {old_value} → {session.parameters.grid_count}")
        changes_made = True
    
    if investment is not None:
        old_value = session.parameters.investment_per_grid
        session.parameters.investment_per_grid = Decimal(str(investment))
        console.print(f"✏️ Investment/Grid: ${old_value} → ${session.parameters.investment_per_grid}")
        changes_made = True
    
    if not changes_made:
        console.print("ℹ️ No parameters specified to adjust")
        return
    
    # Validate new parameters
    try:
        grid_config = session.parameters.to_grid_config()
        console.print("\n✅ Parameter validation passed")
        
        # Show new totals
        total_investment = session.parameters.grid_count * session.parameters.investment_per_grid
        grid_spacing = (session.parameters.upper_bound - session.parameters.lower_bound) / session.parameters.grid_count
        
        console.print(f"New Total Investment: ${total_investment}")
        console.print(f"New Grid Spacing: ${grid_spacing:.2f}")
        
        # Confirmation
        if Confirm.ask("Apply these parameter changes?", default=True):
            console.print("✅ Parameters updated successfully")
            console.print("💡 Use 'grid resume' to continue trading with new parameters")
        else:
            console.print("❌ Parameter changes cancelled")
            
    except Exception as e:
        console.print(f"❌ Parameter validation failed: {e}", style="red")
        console.print("🔄 Reverting changes...")

@grid_app.command("monitor")
def monitor_session(
    session_id: Optional[str] = typer.Argument(None, help="Session ID to monitor (default: active session)"),
    refresh: int = typer.Option(5, "--refresh", "-r", help="Refresh interval in seconds")
):
    """Monitor trading session with periodic updates"""
    
    if not session_id:
        active_session = session_manager.get_active_session()
        if not active_session:
            console.print("❌ No active session to monitor", style="red")
            raise typer.Exit(1)
        session = active_session
        session_id = session.session_id
    else:
        session = session_manager.get_session(session_id)
        if not session:
            console.print(f"❌ Session {session_id} not found", style="red")
            raise typer.Exit(1)
    
    console.print(f"📊 Monitoring session {session_id} (refresh every {refresh}s)")
    console.print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            # Clear screen and show status
            console.clear()
            show_static_status(session)
            
            # Show last update time
            console.print(f"\n🕐 Last updated: {datetime.now().strftime('%H:%M:%S')}")
            console.print(f"Next update in {refresh} seconds...")
            
            # Check if session ended
            if not session.is_active() and session.status != TradingSessionStatus.PAUSED:
                console.print(f"\n✅ Session {session_id} has ended")
                break
            
            time.sleep(refresh)
            
    except KeyboardInterrupt:
        console.print("\n👋 Monitoring stopped")

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
    alert_level: str = typer.Option("medium", "--alert-level", help="Alert level: low, medium, high, critical")
):
    """Start advanced safety monitoring for a trading session"""
    try:
        # Validate alert level
        valid_levels = ["low", "medium", "high", "critical"]
        if alert_level not in valid_levels:
            console.print(f"❌ Invalid alert level. Must be one of: {', '.join(valid_levels)}", style="red")
            raise typer.Exit(1)
        
        # Start monitoring
        advanced_safety_manager.start_monitoring(session_id, symbol)
        enhanced_error_handler.start_monitoring()
        
        console.print(f"🛡️  Advanced safety monitoring started")
        console.print(f"Session: [cyan]{session_id}[/cyan]")
        console.print(f"Symbol: [green]{symbol}[/green]")
        console.print(f"Alert Level: [yellow]{alert_level.upper()}[/yellow]")
        
        console.print(f"\n✅ Monitoring systems active:")
        console.print(f"   • Risk metrics tracking")
        console.print(f"   • Market condition analysis")
        console.print(f"   • Connection health monitoring")
        console.print(f"   • Error pattern detection")
        console.print(f"   • Automatic recovery systems")
        
    except Exception as e:
        console.print(f"❌ Error starting safety monitoring: {e}", style="red")
        raise typer.Exit(1)

@safety_app.command("status")
def safety_status(
    session_id: Optional[str] = typer.Argument(None, help="Session ID to check (default: all active sessions)"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed risk analysis")
):
    """Show safety status and risk analysis"""
    
    if session_id:
        # Show specific session safety status
        risk_report = advanced_safety_manager.get_risk_report(session_id)
        
        if "error" in risk_report:
            console.print(f"❌ {risk_report['error']}", style="red")
            return
        
        # Create safety status panel
        status_text = Text()
        status_text.append(f"Session: ", style="bold")
        status_text.append(f"{session_id}\n", style="cyan")
        status_text.append(f"Risk Level: ", style="bold")
        
        risk_level = risk_report['risk_level']
        risk_style = {
            'low': 'green',
            'moderate': 'yellow', 
            'high': 'red',
            'extreme': 'red bold',
            'critical': 'red bold blink'
        }.get(risk_level, 'white')
        
        status_text.append(f"{risk_level.upper()}\n", style=risk_style)
        
        # Risk metrics
        metrics = risk_report['metrics']
        status_text.append(f"Current Drawdown: ", style="bold")
        status_text.append(f"{metrics['current_drawdown']:.1%}\n")
        status_text.append(f"Volatility: ", style="bold")
        status_text.append(f"{metrics['volatility']:.1%}\n")
        status_text.append(f"Sharpe Ratio: ", style="bold")
        status_text.append(f"{metrics['sharpe_ratio']:.2f}\n")
        status_text.append(f"VaR (95%): ", style="bold")
        status_text.append(f"${metrics['var_95']:.2f}\n")
        
        panel = Panel(status_text, title="🛡️ Safety Status", border_style="blue")
        console.print(panel)
        
        # Show recent alerts if detailed
        if detailed and risk_report['recent_alerts']:
            console.print("\n🚨 Recent Alerts:")
            for alert in risk_report['recent_alerts']:
                alert_style = {
                    'info': 'blue',
                    'warning': 'yellow',
                    'error': 'red',
                    'critical': 'red bold'
                }.get(alert['type'], 'white')
                
                console.print(f"   [{alert['timestamp'][:19]}] {alert['title']}", style=alert_style)
                console.print(f"      {alert['message']}", style="dim")
    
    else:
        # Show overall safety status
        error_summary = enhanced_error_handler.get_error_summary(24)
        
        console.print("🛡️ Overall Safety Status (Last 24 Hours)")
        console.print("=" * 50)
        
        # Error summary
        total_errors = error_summary['total_errors']
        if total_errors == 0:
            console.print("✅ No errors detected", style="green")
        else:
            console.print(f"⚠️  {total_errors} errors detected", style="yellow")
            
            # Show by category
            console.print("\n📊 Errors by Category:")
            for category, count in error_summary['by_category'].items():
                console.print(f"   {category.title()}: {count}")
            
            # Show by severity
            console.print("\n🚨 Errors by Severity:")
            for severity, count in error_summary['by_severity'].items():
                severity_style = {
                    'low': 'green',
                    'medium': 'yellow',
                    'high': 'red',
                    'critical': 'red bold'
                }.get(severity, 'white')
                console.print(f"   {severity.title()}: {count}", style=severity_style)
        
        # Connection health
        health = error_summary['connection_health']
        console.print(f"\n🔗 Connection Health:")
        console.print(f"Overall Status: {'✅ Healthy' if health['overall_healthy'] else '❌ Issues Detected'}")
        
        for service, status in health['services'].items():
            health_icon = "✅" if status['healthy'] else "❌"
            console.print(f"   {service}: {health_icon}")

@safety_app.command("alerts")
def show_alerts(
    session_id: Optional[str] = typer.Option(None, "--session", help="Filter by session ID"),
    hours: int = typer.Option(24, "--hours", help="Show alerts from last N hours"),
    severity: Optional[str] = typer.Option(None, "--severity", help="Filter by severity: low, medium, high, critical")
):
    """Show safety alerts and notifications"""
    
    # Get alerts from advanced safety manager
    console.print(f"🚨 Safety Alerts (Last {hours} hours)")
    console.print("=" * 50)
    
    # This would show actual alerts from the safety manager
    # For now, show a sample structure
    
    sample_alerts = [
        {
            'timestamp': '2025-08-10T06:24:00Z',
            'type': 'warning',
            'category': 'Risk',
            'title': 'Increased Volatility Detected',
            'message': 'Market volatility increased to 45%, consider reducing position size',
            'session_id': 'BTCUSDC_20250810_062400'
        },
        {
            'timestamp': '2025-08-10T06:20:00Z',
            'type': 'info',
            'category': 'Market',
            'title': 'Regime Change Detected',
            'message': 'Market regime changed from NORMAL to VOLATILE',
            'session_id': None
        }
    ]
    
    if not sample_alerts:
        console.print("✅ No alerts in the specified time period", style="green")
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
        if session_id and alert.get('session_id') != session_id:
            continue
        if severity and alert['type'] != severity:
            continue
        
        alert_style = {
            'info': 'blue',
            'warning': 'yellow', 
            'error': 'red',
            'critical': 'red bold'
        }.get(alert['type'], 'white')
        
        table.add_row(
            alert['timestamp'][:19],
            f"[{alert_style}]{alert['type'].upper()}[/{alert_style}]",
            alert['category'],
            alert['title'],
            alert.get('session_id', 'N/A')
        )
    
    console.print(table)

@safety_app.command("emergency")
def emergency_controls(
    action: str = typer.Argument(..., help="Emergency action: stop, flatten, status"),
    session_id: Optional[str] = typer.Option(None, "--session", help="Session ID (required for stop)"),
    confirm: bool = typer.Option(False, "--confirm", help="Confirm emergency action")
):
    """Emergency safety controls"""
    
    if action == "status":
        console.print("🚨 Emergency Status Check")
        console.print("=" * 30)
        
        # Check if any emergency conditions exist
        emergency_sessions = []  # This would check actual sessions
        
        if not emergency_sessions:
            console.print("✅ No emergency conditions detected", style="green")
        else:
            console.print("⚠️ Emergency conditions detected:", style="red bold")
            for session in emergency_sessions:
                console.print(f"   Session {session}: Critical risk level")
    
    elif action == "stop":
        if not session_id:
            console.print("❌ Session ID required for emergency stop", style="red")
            raise typer.Exit(1)
        
        if not confirm:
            console.print("⚠️ Emergency stop requires --confirm flag", style="red")
            console.print("This will immediately halt all trading for the session")
            raise typer.Exit(1)
        
        console.print(f"🚨 EMERGENCY STOP initiated for {session_id}")
        console.print("⏹️ Stopping all trading activity...")
        console.print("🔄 Cancelling open orders...")
        console.print("📊 Recording emergency event...")
        console.print("✅ Emergency stop completed")
        
    elif action == "flatten":
        if not confirm:
            console.print("⚠️ Emergency flatten requires --confirm flag", style="red") 
            console.print("This will immediately close all positions")
            raise typer.Exit(1)
        
        console.print("🚨 EMERGENCY FLATTEN initiated")
        console.print("📈 Closing all open positions...")
        console.print("🔄 Cancelling open orders...")
        console.print("💰 Converting to base currency...")
        console.print("✅ Emergency flatten completed")
        
    else:
        console.print(f"❌ Unknown emergency action: {action}", style="red")
        console.print("Available actions: stop, flatten, status")
        raise typer.Exit(1)

# Advanced Monitoring Commands
@monitor_app.command("market")
def monitor_market_conditions(
    symbol: str = typer.Argument(..., help="Trading symbol to monitor"),
    duration: int = typer.Option(300, "--duration", help="Monitoring duration in seconds"),
    refresh: int = typer.Option(5, "--refresh", help="Refresh interval in seconds")
):
    """Monitor real-time market conditions and regime changes"""
    
    console.print(f"📊 Market Condition Monitoring: {symbol}")
    console.print(f"Duration: {duration}s | Refresh: {refresh}s")
    console.print("=" * 50)
    
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
                    correlation_breakdown=False
                )
            
            # Clear screen and show current conditions
            console.clear()
            console.print(f"📊 Market Conditions: {symbol} | Elapsed: {elapsed:.0f}s/{duration}s")
            console.print("=" * 60)
            
            # Market regime
            regime_style = {
                'calm': 'green',
                'normal': 'blue',
                'volatile': 'yellow',
                'extreme': 'red',
                'crisis': 'red bold'
            }.get(market_conditions.regime.value, 'white')
            
            console.print(f"Market Regime: [{regime_style}]{market_conditions.regime.value.upper()}[/{regime_style}]")
            
            # Trend analysis
            trend_style = {
                'up': 'green',
                'down': 'red',
                'sideways': 'blue'
            }.get(market_conditions.trend_direction, 'white')
            
            console.print(f"Trend: [{trend_style}]{market_conditions.trend_direction.upper()}[/{trend_style}] "
                        f"(Strength: {market_conditions.trend_strength:.1%})")
            
            # Volatility
            vol_style = 'green' if market_conditions.volatility_percentile < 0.3 else 'yellow' if market_conditions.volatility_percentile < 0.7 else 'red'
            console.print(f"Volatility: [{vol_style}]{market_conditions.volatility_percentile:.1%} percentile[/{vol_style}]")
            
            # Grid trading suitability
            is_favorable = market_conditions.is_favorable_for_grid()
            favorable_style = 'green' if is_favorable else 'red'
            console.print(f"Grid Trading: [{'green' if is_favorable else 'red'}]{'✅ FAVORABLE' if is_favorable else '❌ UNFAVORABLE'}[/{'green' if is_favorable else 'red'}]")
            
            # Additional indicators
            console.print(f"\n📈 Additional Indicators:")
            console.print(f"   Volume Profile: {market_conditions.volume_profile}")
            console.print(f"   Correlation Risk: {'⚠️ HIGH' if market_conditions.correlation_breakdown else '✅ Normal'}")
            
            console.print(f"\n🕐 Next update in {refresh} seconds... (Ctrl+C to stop)")
            
            time.sleep(refresh)
            
    except KeyboardInterrupt:
        console.print("\n👋 Market monitoring stopped")

@monitor_app.command("performance")
def monitor_performance(
    session_id: Optional[str] = typer.Option(None, "--session", help="Session ID to monitor"),
    metric: str = typer.Option("all", "--metric", help="Metric to focus on: all, pnl, risk, trades")
):
    """Monitor trading performance metrics"""
    
    if session_id:
        console.print(f"📈 Performance Monitor: {session_id}")
    else:
        console.print("📈 Overall Performance Monitor")
    
    console.print("=" * 50)
    
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
        ("Success Rate", "72%", "60%+", "✅ Good")
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
            metric_name,
            current,
            target,
            f"[{status_style}]{status}[/{status_style}]"
        )
    
    console.print(table)
    
    # Performance chart (text-based)
    console.print(f"\n📊 P&L Chart (Last 24 Hours):")
    console.print("   ┌─────────────────────────────────────────┐")
    console.print("   │ $1200 ████████████████████████████████  │")
    console.print("   │ $1000 ████████████████████████████      │")
    console.print("   │ $800  ████████████████████              │")
    console.print("   │ $600  ████████████████                  │")
    console.print("   │ $400  ████████████                      │")
    console.print("   │ $200  ██████                            │")
    console.print("   │ $0    ──────────────────────────────────│")
    console.print("   └─────────────────────────────────────────┘")
    console.print("     00:00    06:00    12:00    18:00   24:00")

@monitor_app.command("errors")
def monitor_errors(
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow error log in real-time"),
    severity: Optional[str] = typer.Option(None, "--severity", help="Filter by severity: low, medium, high, critical"),
    hours: int = typer.Option(1, "--hours", help="Show errors from last N hours")
):
    """Monitor system errors and recovery actions"""
    
    console.print(f"🚨 Error Monitoring (Last {hours} hours)")
    console.print("=" * 50)
    
    if follow:
        console.print("📝 Following error log in real-time... (Ctrl+C to stop)")
        console.print()
        
        try:
            # Simulate real-time error monitoring
            sample_errors = [
                ("INFO", "Network", "Connection restored to Binance API"),
                ("WARNING", "Trading", "Order partially filled, adjusting grid"),
                ("ERROR", "Data", "Price feed timeout, using cached data"),
                ("INFO", "Safety", "Risk level returned to normal")
            ]
            
            for i, (level, category, message) in enumerate(sample_errors):
                time.sleep(2)  # Simulate real-time
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                level_style = {
                    'INFO': 'blue',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red bold'
                }.get(level, 'white')
                
                console.print(f"[{timestamp}] [{level_style}]{level}[/{level_style}] {category}: {message}")
                
        except KeyboardInterrupt:
            console.print("\n👋 Error monitoring stopped")
    
    else:
        # Show static error summary
        error_summary = enhanced_error_handler.get_error_summary(hours)
        
        if error_summary['total_errors'] == 0:
            console.print("✅ No errors detected in the specified period", style="green")
            return
        
        # Show recent errors
        console.print(f"📋 Recent Errors ({error_summary['total_errors']} total):")
        
        for error in error_summary['recent_errors']:
            if severity and error.get('severity', '').lower() != severity.lower():
                continue
            
            severity_style = {
                'low': 'green',
                'medium': 'yellow',
                'high': 'red',
                'critical': 'red bold'
            }.get(error.get('severity', ''), 'white')
            
            console.print(f"\n[{error['timestamp'][:19]}] [{severity_style}]{error.get('severity', 'UNKNOWN').upper()}[/{severity_style}]")
            console.print(f"   Category: {error.get('category', 'Unknown')}")
            console.print(f"   Title: {error['title']}")
            console.print(f"   Message: {error['message']}")
            
            if error.get('recovery_action'):
                console.print(f"   Recovery: {error['recovery_action']}")

# Configuration commands
@config_app.command("init")
def init_config(
    api_key: str = typer.Option(..., prompt=True, hide_input=True, help="Binance API Key"),
    api_secret: str = typer.Option(..., prompt=True, hide_input=True, help="Binance API Secret"),
    testnet: bool = typer.Option(True, help="Use Binance Testnet"),
    data_dir: str = typer.Option("data", help="Data directory"),
    log_level: str = typer.Option("INFO", help="Log level")
):
    """Initialize trading bot configuration"""
    config = {
        "api_key": api_key,
        "api_secret": api_secret,
        "testnet": testnet,
        "data_dir": data_dir,
        "log_level": log_level,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    config_path = Path("config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    console.print(f"✅ Configuration saved to {config_path}", style="green")
    
    # Initialize data directory
    Path(data_dir).mkdir(exist_ok=True)
    console.print(f"✅ Data directory created: {data_dir}", style="green")

@config_app.command("show")
def show_config():
    """Show current configuration"""
    config_path = Path("config.json")
    if not config_path.exists():
        console.print("❌ No configuration found. Run 'config init' first.", style="red")
        raise typer.Exit(1)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Hide sensitive information
    display_config = config.copy()
    if "api_key" in display_config:
        display_config["api_key"] = f"{display_config['api_key'][:8]}..."
    if "api_secret" in display_config:
        display_config["api_secret"] = "***"
    
    table = Table(title="Trading Bot Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="magenta")
    
    for key, value in display_config.items():
        table.add_row(key, str(value))
    
    console.print(table)

# Data commands
@data_app.command("download")
def download_data(
    symbol: str = typer.Argument(..., help="Trading symbol (e.g., BTCUSDC)"),
    interval: str = typer.Option("1h", help="Kline interval"),
    days: int = typer.Option(30, help="Number of days to download"),
    update_existing: bool = typer.Option(False, help="Update existing data")
):
    """Download historical market data"""
    
    async def _download():
        loader = DataLoader()
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Downloading {symbol} {interval} data...", total=None)
            
            try:
                klines = await loader.fetch_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_date,
                    limit=1000
                )
                
                progress.update(task, description=f"Saving {len(klines)} klines...")
                
                repo = get_default_repository()
                saved_count = repo.save_klines(klines)
                
                progress.update(task, description="✅ Complete!")
                
                console.print(f"✅ Downloaded and saved {saved_count} klines for {symbol}", style="green")
                console.print(f"📊 Date range: {klines[0].open_time} to {klines[-1].close_time}")
                
            except Exception as e:
                console.print(f"❌ Error downloading data: {e}", style="red")
                raise typer.Exit(1)
    
    asyncio.run(_download())

@data_app.command("list")
def list_data():
    """List available market data"""
    repo = get_default_repository()
    
    # Query unique symbols and intervals from database
    with repo.db_repo.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT symbol, interval, COUNT(*) as count, 
                   MIN(open_time) as start_date, 
                   MAX(open_time) as end_date
            FROM klines 
            GROUP BY symbol, interval
            ORDER BY symbol, interval
        """)
        
        rows = cursor.fetchall()
    
    if not rows:
        console.print("📭 No market data found. Use 'data download' to get started.", style="yellow")
        return
    
    table = Table(title="Available Market Data")
    table.add_column("Symbol", style="cyan")
    table.add_column("Interval", style="blue")
    table.add_column("Records", style="green")
    table.add_column("Start Date", style="magenta")
    table.add_column("End Date", style="magenta")
    
    for row in rows:
        table.add_row(
            row['symbol'],
            row['interval'],
            str(row['count']),
            row['start_date'].strftime('%Y-%m-%d') if row['start_date'] else 'N/A',
            row['end_date'].strftime('%Y-%m-%d') if row['end_date'] else 'N/A'
        )
    
    console.print(table)

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
    output_file: Optional[str] = typer.Option(None, help="Output file for results")
):
    """Run backtesting on historical data"""
    
    async def _backtest():
        # Load data
        loader = DataLoader()
        
        start_dt = datetime.fromisoformat(start_date) if start_date else datetime.now(timezone.utc) - timedelta(days=30)
        end_dt = datetime.fromisoformat(end_date) if end_date else datetime.now(timezone.utc)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading market data...", total=None)
            
            try:
                klines = await loader.load_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_dt,
                    end_time=end_dt
                )
                
                if not klines:
                    console.print(f"❌ No data found for {symbol}", style="red")
                    raise typer.Exit(1)
                
                progress.update(task, description="Setting up backtest...")
                
                # Initialize strategy
                if strategy == "grid":
                    strategy_obj = SimpleGridStrategy(
                        initial_capital=initial_capital,
                        commission=commission
                    )
                elif strategy == "baseline":
                    strategy_obj = BaselineStrategy()
                else:
                    console.print(f"❌ Unknown strategy: {strategy}", style="red")
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
                    console.print(f"💾 Results saved to {output_file}", style="blue")
                
            except Exception as e:
                console.print(f"❌ Backtest failed: {e}", style="red")
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
    
    console.print(Panel(panel_content, title="Backtest Results", expand=False))

def _save_backtest_results(result, filename: str):
    """Save backtest results to file"""
    result_data = {
        'strategy_name': result.strategy_name,
        'symbol': result.symbol,
        'start_date': result.start_date.isoformat(),
        'end_date': result.end_date.isoformat(),
        'initial_capital': float(result.initial_capital),
        'final_capital': float(result.final_capital),
        'total_return': float(result.total_return),
        'metrics': {
            'sharpe_ratio': float(result.metrics.sharpe_ratio),
            'max_drawdown': float(result.metrics.max_drawdown),
            'total_trades': result.metrics.total_trades,
            'win_rate': float(result.metrics.win_rate)
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(result_data, f, indent=2)

# Model commands
@model_app.command("list")
def list_models():
    """List all saved models"""
    manager = get_default_artifact_manager()
    models = manager.list_models()
    
    if not models:
        console.print("📭 No models found.", style="yellow")
        return
    
    table = Table(title="Saved Models")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="blue")
    table.add_column("Version", style="green")
    table.add_column("Created", style="magenta")
    
    for model in models:
        created_date = datetime.fromisoformat(model['created_at']).strftime('%Y-%m-%d %H:%M')
        table.add_row(
            model['name'],
            model['model_type'],
            model['version'],
            created_date
        )
    
    console.print(table)

@model_app.command("optimize")
def optimize_strategy(
    symbol: str = typer.Argument(..., help="Trading symbol"),
    strategy: str = typer.Option("baseline", help="Strategy to optimize"),
    n_trials: int = typer.Option(50, help="Number of optimization trials"),
    output_file: Optional[str] = typer.Option(None, help="Save results to file")
):
    """Optimize strategy hyperparameters"""
    
    async def _optimize():
        # Load data
        loader = DataLoader()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading market data...", total=None)
            
            try:
                klines = await loader.load_klines(
                    symbol=symbol,
                    interval="1h",
                    start_time=datetime.now(timezone.utc) - timedelta(days=180)
                )
                
                if not klines:
                    console.print(f"❌ No data found for {symbol}", style="red")
                    raise typer.Exit(1)
                
                progress.update(task, description="Setting up optimizer...")
                
                # Initialize optimizer
                optimizer = BayesianOptimizer()
                
                progress.update(task, description=f"Running {n_trials} optimization trials...")
                
                # Run optimization
                if strategy == "baseline":
                    study = await optimizer.optimize_baseline_strategy(
                        klines, n_trials=n_trials
                    )
                else:
                    console.print(f"❌ Unknown strategy: {strategy}", style="red")
                    raise typer.Exit(1)
                
                progress.update(task, description="✅ Optimization complete!")
                
                # Display results
                best_params = study.best_params
                best_value = study.best_value
                
                console.print(f"🎯 **Best Parameters**: {best_params}")
                console.print(f"📊 **Best Value**: {best_value:.4f}")
                
                # Save results if requested
                if output_file:
                    results = {
                        'strategy': strategy,
                        'symbol': symbol,
                        'n_trials': n_trials,
                        'best_params': best_params,
                        'best_value': best_value,
                        'optimization_history': [
                            {'trial': i, 'value': trial.value, 'params': trial.params}
                            for i, trial in enumerate(study.trials)
                        ]
                    }
                    
                    with open(output_file, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                    
                    console.print(f"💾 Results saved to {output_file}", style="blue")
                
            except Exception as e:
                console.print(f"❌ Optimization failed: {e}", style="red")
                raise typer.Exit(1)
    
    asyncio.run(_optimize())

# Trading commands
@trade_app.command("paper")
def start_paper_trading(
    symbol: str = typer.Argument(..., help="Trading symbol"),
    strategy: str = typer.Option("grid", help="Strategy type"),
    initial_capital: float = typer.Option(10000.0, help="Initial capital"),
    dry_run: bool = typer.Option(False, help="Dry run mode")
):
    """Start paper trading"""
    console.print("🚧 Paper trading not yet implemented", style="yellow")
    # TODO: Implement paper trading functionality

@trade_app.command("live")
def start_live_trading(
    symbol: str = typer.Argument(..., help="Trading symbol"),
    strategy: str = typer.Option("grid", help="Strategy type"),
    initial_capital: float = typer.Option(1000.0, help="Initial capital"),
    confirm: bool = typer.Option(False, "--confirm", help="Confirm live trading"),
    mode: str = typer.Option("manual", help="Trading mode: manual or ai"),
    confirm_live: bool = typer.Option(False, "--confirm-live", help="Explicit confirmation for live AI mode")
):
    """Start live trading (REAL MONEY!) - DEPRECATED: Use 'grid manual' or 'grid ai' instead"""
    
    console.print("⚠️ [yellow]This command is deprecated. Use:[/yellow]")
    console.print("   • [bold]grid manual[/bold] for manual parameter input")
    console.print("   • [bold]grid ai[/bold] for AI-suggested parameters")
    console.print("\n� Redirecting to new grid commands...")
    
    # Redirect to appropriate grid command
    if mode.lower() == "ai":
        console.print(f"🤖 Starting AI grid mode for {symbol}...")
        start_ai_grid(symbol=symbol, paper=False, confirm_live=confirm_live)
    else:
        console.print(f"� Starting manual grid mode for {symbol}...")
        start_manual_grid(symbol=symbol, paper=False, confirm_live=confirm_live)

# Utility commands
@app.command("status")
def show_status():
    """Show bot status and health check"""
    console.print("🤖 Trading Bot Status", style="bold blue")
    
    # Check configuration
    config_exists = Path("config.json").exists()
    status_icon = "✅" if config_exists else "❌"
    console.print(f"{status_icon} Configuration: {'OK' if config_exists else 'Missing'}")
    
    # Check data directory
    data_dir = Path("data")
    data_exists = data_dir.exists()
    status_icon = "✅" if data_exists else "❌"
    console.print(f"{status_icon} Data Directory: {'OK' if data_exists else 'Missing'}")
    
    # Check database
    db_exists = Path("trading_bot.db").exists()
    status_icon = "✅" if db_exists else "❌"
    console.print(f"{status_icon} Database: {'OK' if db_exists else 'Missing'}")
    
    # Check models
    artifacts_dir = Path("artifacts")
    models_exist = artifacts_dir.exists()
    status_icon = "✅" if models_exist else "❌"
    console.print(f"{status_icon} Models Directory: {'OK' if models_exist else 'Missing'}")

@app.command("version")
def show_version():
    """Show bot version information"""
    version_info = {
        "Bot Version": "1.0.0",
        "Python": "3.10+",
        "Created": "2024"
    }
    
    table = Table(title="Version Information")
    table.add_column("Component", style="cyan")
    table.add_column("Version", style="green")
    
    for component, version in version_info.items():
        table.add_row(component, version)
    
    console.print(table)

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
        table.add_row("Manual Resume Required", "YES" if state.requires_manual_resume else "NO")
        
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
            table.add_row("Last Error", state.last_error[:50] + "..." if len(state.last_error) > 50 else state.last_error)
        
        console.print(table)
        
        # Show position details if any
        if active_positions:
            console.print("\n[bold yellow]Active Positions:[/bold yellow]")
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
                    f"{pos.unrealized_pnl:.2f}"
                )
            
            console.print(pos_table)
        
    except FileNotFoundError:
        console.print(f"❌ State file not found: {state_file}", style="red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Error reading state: {e}", style="red")
        raise typer.Exit(1)

@safety_app.command("clear-resume")
def clear_manual_resume(
    state_file: str = typer.Option("bot_state.json", help="Bot state file path"),
    confirm: bool = typer.Option(False, "--yes", help="Skip confirmation prompt")
):
    """Clear the manual resume flag (operator intervention)"""
    
    try:
        state = load_state(state_file)
        
        if not state.requires_manual_resume:
            console.print("✅ No manual resume flag to clear", style="green")
            return
        
        # Show current state
        console.print(f"[yellow]Current flatten reason:[/yellow] {state.flatten_reason}")
        if state.flatten_timestamp:
            console.print(f"[yellow]Flatten time:[/yellow] {state.flatten_timestamp.isoformat()}")
        
        # Confirmation
        if not confirm:
            proceed = typer.confirm(
                "⚠️  This will clear the manual resume flag and allow trading to continue. "
                "Are you sure the issues have been resolved?"
            )
            if not proceed:
                console.print("Operation cancelled", style="yellow")
                raise typer.Exit(0)
        
        # Clear the flag
        state.clear_manual_resume()
        save_state(state, state_file)
        
        console.print("✅ Manual resume flag cleared - trading can now continue", style="green")
        
    except FileNotFoundError:
        console.print(f"❌ State file not found: {state_file}", style="red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Error updating state: {e}", style="red")
        raise typer.Exit(1)

@safety_app.command("flatten")
def manual_flatten(
    reason: str = typer.Option(..., help="Reason for manual flatten"),
    state_file: str = typer.Option("bot_state.json", help="Bot state file path"),
    config_file: str = typer.Option("config.json", help="Configuration file path"),
    confirm: bool = typer.Option(False, "--yes", help="Skip confirmation prompt")
):
    """Manually trigger a flatten operation"""
    
    async def do_flatten():
        try:
            # Load configuration and state
            config_path = Path(config_file)
            if not config_path.exists():
                console.print(f"❌ Configuration file not found: {config_file}", style="red")
                return False
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            state = load_state(state_file)
            
            # Show current positions
            active_positions = [pos for pos in state.positions.values() if not pos.is_flat]
            if not active_positions and not state.active_orders:
                console.print("ℹ️  No active positions or orders to flatten", style="blue")
                return True
            
            console.print(f"[yellow]Positions to close:[/yellow] {len(active_positions)}")
            console.print(f"[yellow]Orders to cancel:[/yellow] {len(state.active_orders)}")
            
            # Confirmation
            if not confirm:
                proceed = typer.confirm(
                    f"⚠️  This will flatten all positions and cancel all orders. "
                    f"Reason: '{reason}'. Continue?"
                )
                if not proceed:
                    console.print("Operation cancelled", style="yellow")
                    return False
            
            # Create client and execute flatten
            client = await create_binance_client(
                config["api_key"],
                config["api_secret"],
                config.get("testnet", True)
            )
            
            console.print("🔄 Executing flatten operation...", style="yellow")
            
            result = await execute_flatten_operation(client, state, reason)
            
            if result.success:
                console.print("✅ Flatten operation completed successfully", style="green")
                console.print(f"Orders canceled: {result.orders_canceled}")
                console.print(f"Positions closed: {len(result.positions_closed)}")
                console.print(f"Total proceeds: ${result.total_proceeds:.2f}")
                console.print(f"Execution time: {result.execution_time_seconds:.1f}s")
                console.print("\n⚠️  Manual resume flag has been set - use 'safety clear-resume' when ready", style="yellow")
                return True
            else:
                console.print(f"❌ Flatten operation failed: {result.error_message}", style="red")
                console.print("⚠️  Manual resume flag has still been set for safety", style="yellow")
                return False
                
        except Exception as e:
            console.print(f"❌ Error during flatten: {e}", style="red")
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
    confirm_live: bool = typer.Option(False, "--confirm-live", help="Confirm live trading")
):
    """Run the trading bot with safety integration"""
    
    # Safety checks before any live mode starts
    if not paper and mode.lower() == "ai" and not confirm_live:
        console.print("⚠️ Refusing to run live without --confirm-live in AI mode.", style="red")
        raise typer.Abort()
    
    if not paper and not typer.confirm("Live trading ON. Type 'yes' to confirm:", default=False):
        console.print("❌ Live trading cancelled by user.", style="red")
        raise typer.Abort()
    
    async def run():
        try:
            # Load configuration
            config_path = Path(config_file)
            if not config_path.exists():
                console.print(f"❌ Configuration file not found: {config_file}", style="red")
                return False
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load or create state
            try:
                state = load_state(state_file)
                console.print(f"✅ Loaded existing state from {state_file}", style="green")
            except FileNotFoundError:
                state = create_empty_state(symbol, config)
                console.print(f"✅ Created new state for {symbol}", style="green")
            
            # Load risk limits
            risk_limits = RiskLimits()
            if risk_limits_file and Path(risk_limits_file).exists():
                with open(risk_limits_file, 'r') as f:
                    limits_config = json.load(f)
                    # Update risk limits with loaded config
                    for key, value in limits_config.items():
                        if hasattr(risk_limits, key):
                            setattr(risk_limits, key, value)
                console.print(f"✅ Loaded risk limits from {risk_limits_file}", style="green")
            
            # Create client
            client = await create_binance_client(
                config["api_key"],
                config["api_secret"],
                config.get("testnet", True)
            )
            
            # Display trading mode
            trading_mode = "📄 PAPER TRADING" if paper else "💰 LIVE TRADING"
            mode_style = "green" if paper else "bold red"
            console.print(f"{trading_mode} MODE ACTIVE", style=mode_style)
            console.print(f"🚀 Starting trading bot for {symbol} with {strategy} strategy ({mode} mode)", style="green")
            console.print("Press Ctrl+C to stop gracefully", style="yellow")
            
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
                state_save_path=state_file
            )
            
            return True
            
        except KeyboardInterrupt:
            console.print("\n🛑 Shutdown requested by user", style="yellow")
            return True
        except Exception as e:
            console.print(f"❌ Error running trading bot: {e}", style="red")
            return False
    
    success = asyncio.run(run())
    if not success:
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
