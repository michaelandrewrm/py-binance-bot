import pandas as pd
import logging


def get_historical_klines(
    client, symbol, interval, start_date, logger: logging.Logger = None
):
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    try:
        klines = client.get_historical_klines(symbol, interval, start_date)
    except Exception:
        logger.error("Error fetching data from Binance", exc_info=False)
        return pd.DataFrame()

    data = pd.DataFrame(
        klines,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )

    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
    data.set_index("timestamp", inplace=True)
    data["close_time"] = pd.to_datetime(data["close_time"], unit="ms")

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    data[numeric_cols] = data[numeric_cols].astype(float)

    return data
