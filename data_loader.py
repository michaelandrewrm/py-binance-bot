from binance.client import Client
import pandas as pd
from config import API_KEY, API_SECRET, LOOKBACK

# Create client once globally (reuse connection)
client = Client(API_KEY, API_SECRET)

def get_historical_klines(symbol, interval, lookback=LOOKBACK):
    """
    Fetch historical klines (candlestick data) from Binance API.

    Args:
        symbol (str): Trading pair symbol, e.g. "BTCUSDT".
        interval (str): Interval string, e.g. "1h", "1d".
        lookback (str): Lookback period for data, e.g. "1 month ago UTC".

    Returns:
        pd.DataFrame: DataFrame with OHLCV data and timestamps.
    """
    try:
        klines = client.get_historical_klines(symbol, interval, lookback)
    except Exception as e:
        print(f"Error fetching data from Binance: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time", 
        "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", 
        "taker_buy_quote_asset_volume", "ignore"
    ])

    # Convert time columns
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    # Convert numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume",
                    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    return df
