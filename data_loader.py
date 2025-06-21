# data_loader.py

from binance.client import Client
import pandas as pd
from config import API_KEY, API_SECRET, BASE_URL

client = Client(API_KEY, API_SECRET)
client.API_URL = BASE_URL

def get_historical_klines(symbol, interval, start_str='90 days ago UTC'):
    klines = client.get_historical_klines(symbol, interval, start_str)
    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
