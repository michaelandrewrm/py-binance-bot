import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib  # for saving and loading

class FeatureEngineer:
    def __init__(self, window=30):
        self.window = window
        self.scaler = StandardScaler()
        self.feature_cols = [
            "SMA_10", "EMA_10", "RSI_14", "MACD", "MACD_signal",
            "BB_High", "BB_Low", "Volume_Avg_10",
            "vol_change_pct", "volatility", "log_return",
            "%K", "%D"
        ]
    
    def add_stochastic_oscillator(self, df, k_window=14, d_window=3):
        low_min = df["low"].rolling(window=k_window).min()
        high_max = df["high"].rolling(window=k_window).max()
        df["%K"] = 100 * (df["close"] - low_min) / (high_max - low_min)
        df["%D"] = df["%K"].rolling(window=d_window).mean()
        return df

    def add_indicators(self, df):
        df["SMA_10"] = df["close"].rolling(window=10).mean()
        df["EMA_10"] = df["close"].ewm(span=10, adjust=False).mean()

        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = -delta.clip(upper=0).rolling(window=14).mean()
        RS = gain / loss
        df["RSI_14"] = 100 - (100 / (1 + RS))

        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        df["BB_Mid"] = df["close"].rolling(window=20).mean()
        df["BB_Std"] = df["close"].rolling(window=20).std()
        df["BB_High"] = df["BB_Mid"] + 2 * df["BB_Std"]
        df["BB_Low"] = df["BB_Mid"] - 2 * df["BB_Std"]

        df["Volume_Avg_10"] = df["volume"].rolling(window=10).mean()
        df["vol_change_pct"] = df["volume"].pct_change()
        df["volatility"] = df["close"].rolling(window=10).std()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # add stochastic oscillator
        df = self.add_stochastic_oscillator(df)
        return df

    def create_features_labels(self, df):
        df = self.add_indicators(df).dropna().copy()

        # Normalize only on training data
        features = df[self.feature_cols].values
        features_scaled = self.scaler.fit_transform(features)
        df[self.feature_cols] = features_scaled

        X, y = [], []
        for i in range(self.window, len(df) - 1):
            X.append(df[self.feature_cols].iloc[i - self.window:i].values)
            y.append(int(df["close"].iloc[i + 1] > df["close"].iloc[i]))

        return df.iloc[self.window:-1], np.array(X), np.array(y)

    def transform_live(self, df):
        df = self.add_indicators(df).dropna().copy()
        df[self.feature_cols] = self.scaler.transform(df[self.feature_cols].values)
        return df

    def save_scaler(self, path="scaler.pkl"):
        joblib.dump(self.scaler, path)

    def load_scaler(self, path="scaler.pkl"):
        self.scaler = joblib.load(path)
