"""
Trading commands for the trading bot CLI
"""
import typer
from ui.error_handling import handle_cli_errors
from ui.output import OutputHandler

logger = OutputHandler("ui.cli")

# Trade commands will be registered to this app instance
trade_app = typer.Typer(help="Trading commands")


@trade_app.command("paper")
@handle_cli_errors
def start_paper_trading(
    symbol: str = typer.Argument(..., help="Trading symbol"),
    strategy: str = typer.Option("grid", help="Strategy type"),
    initial_capital: float = typer.Option(10000.0, help="Initial capital"),
    dry_run: bool = typer.Option(False, help="Dry run mode"),
):
    """Start paper trading"""
    logger.warning("🚧 Paper trading not yet implemented")
    logger.info(f"Would start paper trading for {symbol} with ${initial_capital}")
    # TODO: Implement paper trading functionality


@trade_app.command("live")
@handle_cli_errors
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

    # Import and redirect to appropriate grid command
    from ui.commands.grid_app import start_ai_grid, start_manual_grid
    
    # Redirect to appropriate grid command
    if mode.lower() == "ai":
        logger.custom(f"🤖 Starting AI grid mode for {symbol}...")
        start_ai_grid(symbol=symbol, paper=False, confirm_live=confirm_live)
    else:
        logger.info(f"🎯 Starting manual grid mode for {symbol}...")
        start_manual_grid(symbol=symbol, paper=False, confirm_live=confirm_live)


@trade_app.command("run")
@handle_cli_errors  
def run_trading_bot(
    strategy: str = typer.Option("grid", help="Trading strategy to use"),
    symbol: str = typer.Option("BTCUSDC", help="Trading symbol"),
    config_file: str = typer.Option("config.json", help="Configuration file path"),
    state_file: str = typer.Option("bot_state.json", help="Bot state file path"),
    risk_limits_file: str = typer.Option(None, help="Risk limits configuration file"),
):
    """Run the main trading bot"""
    logger.warning("🚧 Main trading bot execution not yet implemented")
    logger.info(f"Would run {strategy} strategy for {symbol}")
    logger.info(f"Config: {config_file}, State: {state_file}")
    if risk_limits_file:
        logger.info(f"Risk limits: {risk_limits_file}")
    # TODO: Implement main bot execution
