import pandas as pd
import numpy as np
import pandas_ta as ta

def create_features_labels(df, window=24):
    # Add indicators/features to df using pandas_ta
    df['SMA_10'] = df.ta.sma(length=10)
    df['EMA_10'] = df.ta.ema(length=10)
    df['RSI_14'] = df.ta.rsi(length=14)
    macd = df.ta.macd()
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    bb = df.ta.bbands(length=20)
    df['BB_High'] = bb['BBU_20_2.0']
    df['BB_Low'] = bb['BBL_20_2.0']
    df['Volume_Avg_10'] = df['volume'].rolling(window=10).mean()

    # Drop rows with NaNs (due to indicator calculation)
    df = df.dropna().reset_index(drop=True)

    feature_cols = ['SMA_10', 'EMA_10', 'RSI_14', 'MACD', 'MACD_signal', 'BB_High', 'BB_Low', 'Volume_Avg_10']

    X = []
    y = []
    for i in range(window, len(df)):
        X.append(df[feature_cols].iloc[i-window:i].values)
        # Label: 1 if next close price > current close price else 0
        y.append(1 if df['close'].iloc[i] > df['close'].iloc[i-1] else 0)

    X = np.array(X)
    y = np.array(y)

    return df, X, y
