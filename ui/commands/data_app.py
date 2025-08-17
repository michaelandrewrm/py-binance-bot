"""
Data management commands for the trading bot CLI
"""

import typer
import asyncio
from datetime import datetime, timezone, timedelta
from rich.progress import SpinnerColumn, TextColumn, Progress
from rich.table import Table

from ui.error_handling import handle_cli_errors
from data.loader import DataLoader
from utils.output import OutputHandler
from constants.icons import Icon
from storage.repo import HybridRepository

logger = OutputHandler()

# Data commands will be registered to this app instance
data_app = typer.Typer(help="Data management")


@data_app.command("download")
@handle_cli_errors
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

        stripped_symbol = symbol.upper().strip()
        if not stripped_symbol:
            raise ValueError("Trading symbol cannot be empty")

        if len(stripped_symbol) < 6:
            raise ValueError(
                "Trading symbol must be at least 6 characters (e.g., BTCUSDC)"
            )

        if not stripped_symbol.isalnum():
            raise ValueError("Trading symbol must contain only alphanumeric characters")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=logger.console,
        ) as progress:
            task = progress.add_task(
                f"Downloading {stripped_symbol} {interval} data...", total=None
            )

            try:
                klines = await loader.fetch_klines(
                    symbol=stripped_symbol,
                    interval=interval,
                    start_time=start_date,
                    limit=1000,
                )

                progress.update(task, description=f"Saving {len(klines)} klines...")

                repo = HybridRepository()
                saved_count = repo.save_klines(klines)

                progress.update(task, description="Complete!")

                logger.success(
                    f"Downloaded and saved {saved_count} klines for {symbol}"
                )
                if klines:
                    logger.print(
                        f"{Icon.BAR_CHART} Date range: {klines[0].open_time} to {klines[-1].close_time}"
                    )

            except Exception as e:
                logger.error(f"Error downloading data: {e}")
                raise typer.Exit(1)

    asyncio.run(_download())


@data_app.command("list")
@handle_cli_errors
def list_data():
    """List available market data"""
    try:
        repo = HybridRepository()

        # For now, show sample data structure
        logger.print(f"{Icon.BAR_CHART} Available Market Data")
        logger.print("=" * 50)

        # Query unique symbols and timeframes from database
        with repo.db_repo.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
            """
                SELECT symbol, timeframe, COUNT(*) as count,
                       MIN(open_time_utc) as start_date,
                       MAX(open_time_utc) as end_date
                FROM klines
                GROUP BY symbol, timeframe
                ORDER BY symbol, timeframe
            """
            )
            rows = cursor.fetchall()

        if not rows:
            logger.print(
                f"{Icon.INBOX} No market data found. Use 'data download' to get started."
            )
            return

        table = Table(title="Available Market Data")
        table.add_column("Symbol", style="cyan")
        table.add_column("Timeframe", style="blue")
        table.add_column("Records", style="green")
        table.add_column("Start Date", style="magenta")
        table.add_column("End Date", style="magenta")

        for row in rows:
            # Parse the UTC timestamps
            start_date = None
            end_date = None
            if row["start_date"]:
                try:
                    start_date = datetime.fromisoformat(row["start_date"])
                except:
                    start_date = None
            if row["end_date"]:
                try:
                    end_date = datetime.fromisoformat(row["end_date"])
                except:
                    end_date = None
            
            table.add_row(
                row["symbol"],
                row["timeframe"],
                str(row["count"]),
                (start_date.strftime("%Y-%m-%d") if start_date else "N/A"),
                (end_date.strftime("%Y-%m-%d") if end_date else "N/A"),
            )

        logger.print(table)

    except Exception as e:
        logger.error(f"Error listing data: {e}")
        raise typer.Exit(1)
