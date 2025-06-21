# trader.py

from config import SYMBOL, QUANTITY
from binance.client import Client
import numpy as np

from config import API_KEY, API_SECRET, BASE_URL

client = Client(API_KEY, API_SECRET)
client.API_URL = BASE_URL

def predict_and_trade(model, df, window):
    latest_data = df['close'].iloc[-window:].values.reshape(1, window, 1)
    pred = model.predict(latest_data)[0][0]
    print(f"Prediction: {pred:.2f}")

    if pred > 0.6:
        print("BUY signal")
        order = client.create_order(symbol=SYMBOL, side='BUY', type='MARKET', quantity=QUANTITY)
        print(order)

    elif pred < 0.4:
        print("SELL signal")
        order = client.create_order(symbol=SYMBOL, side='SELL', type='MARKET', quantity=QUANTITY)
        print(order)

    else:
        print("No trade executed.")
