from binance.client import Client
import pandas as pd
from config import API_KEY, API_SECRET, USE_TESTNET, BASE_URL

# Create client once globally (reuse connection)
client = Client(API_KEY, API_SECRET)
if USE_TESTNET:
    client.API_URL = BASE_URL

def get_historical_klines(symbol, interval, start_date):
    """
    Fetch historical klines (candlestick data) from Binance API.

    Args:
        symbol (str): Trading pair symbol, e.g. "BTCUSDT".
        interval (str): Interval string, e.g. "1h", "1d".
        start_date (str): start_date period for data, e.g. "1 month ago UTC".

    Returns:
        pd.DataFrame: DataFrame with OHLCV data and timestamps.
    """
    try:
        klines = client.get_historical_klines(symbol, interval, start_date)
    except Exception as e:
        print(f"Error fetching data from Binance: {e}")
        return pd.DataFrame()

    data = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume", "ignore"
    ])

    # Convert time columns
    data["open_time"] = pd.to_datetime(data["open_time"], unit="ms")
    data["close_time"] = pd.to_datetime(data["close_time"], unit="ms")

    # Convert numeric columns
    numeric_cols = [
        "open", "high", "low", "close", "volume", "quote_asset_volume",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
    ]
    data[numeric_cols] = data[numeric_cols].astype(float)

    return data
