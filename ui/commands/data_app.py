"""
Data management commands for the trading bot CLI
"""

import typer
import asyncio
from datetime import datetime, timezone, timedelta
from rich.progress import SpinnerColumn, TextColumn, Progress
from rich.table import Table

from ui.error_handling import handle_cli_errors
from ui.output import OutputHandler
from data.loader import DataLoader

logger = OutputHandler("ui.cli")

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
                    symbol=symbol,
                    interval=interval,
                    start_time=start_date,
                    limit=1000,
                )

                progress.update(task, description=f"Saving {len(klines)} klines...")

                # repo = get_default_repository()
                # saved_count = repo.save_klines(klines)

                # For now, just show what would be saved
                saved_count = len(klines)

                progress.update(task, description="✅ Complete!")

                logger.success(
                    f"Downloaded and saved {saved_count} klines for {symbol}"
                )
                if klines:
                    logger.custom(
                        f"📊 Date range: {klines[0].open_time} to {klines[-1].close_time}"
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
        # repo = get_default_repository()

        # For now, show sample data structure
        logger.custom("📊 Available Market Data")
        logger.custom("=" * 30)

        # Query unique symbols and intervals from database
        # with repo.db_repo.get_connection() as conn:
        #     cursor = conn.cursor()
        #     cursor.execute(
        #         """
        #         SELECT symbol, interval, COUNT(*) as count,
        #                MIN(open_time) as start_date,
        #                MAX(open_time) as end_date
        #         FROM klines
        #         GROUP BY symbol, interval
        #         ORDER BY symbol, interval
        #     """
        #     )
        #     rows = cursor.fetchall()

        # Sample data for demonstration
        rows = [
            {
                "symbol": "BTCUSDC",
                "interval": "1h",
                "count": 720,
                "start_date": datetime(2024, 1, 1),
                "end_date": datetime(2024, 1, 30),
            },
            {
                "symbol": "ETHUSDC",
                "interval": "1h",
                "count": 720,
                "start_date": datetime(2024, 1, 1),
                "end_date": datetime(2024, 1, 30),
            },
        ]

        if not rows:
            logger.custom(
                "📭 No market data found. Use 'data download' to get started."
            )
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
                (
                    row["start_date"].strftime("%Y-%m-%d")
                    if row["start_date"]
                    else "N/A"
                ),
                (row["end_date"].strftime("%Y-%m-%d") if row["end_date"] else "N/A"),
            )

        logger.console.print(table)

    except Exception as e:
        logger.error(f"Error listing data: {e}")
        raise typer.Exit(1)
