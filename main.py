import os
import argparse
import random
import time
import logging
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras_tuner import Hyperband
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from binance.client import Client
from config import API_KEY, API_SECRET, TESTNET, BASE_URL, SYMBOL, START_DATE, get_config_for_interval
from data_loader import get_historical_klines
from model import build_model
from visualizer import plot_equity_curve_with_trades
from backtester import simulate_grid_trades
from trader import Trader
from logger import Logger

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--interval", type=str, default="5m", help="Candlestick interval (e.g.: 5m, 15m, 1h, 4h)")
parser.add_argument("--mode", type=str, default="backtest", choices=["train", "backtest", "trade"])
parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
args = parser.parse_args()

logger = Logger(level=getattr(logging, args.log_level.upper(), logging.INFO))

INTERVAL = args.interval
CONFIG = get_config_for_interval(INTERVAL)
MODEL_CONFIG = CONFIG["model"]
GRID_CONFIG = CONFIG['grid']

def add_technical_indicators(data):
    data['SMA_20'] = ta.sma(data['close'], length=20)
    data['EMA_50'] = ta.ema(data['close'], length=50)
    data['RSI'] = ta.rsi(data['close'], length=14)
    data['MACD'] = ta.macd(data['close'])['MACD_12_26_9']
    data['STOCH_RSI'] = ta.stochrsi(data['close'])['STOCHRSIk_14_14_3_3']
    bbands = ta.bbands(data['close'], length=20)
    data['BB_UPPER'] = bbands['BBU_20_2.0']
    data['BB_LOWER'] = bbands['BBL_20_2.0']
    data['ADX'] = ta.adx(data['high'], data['low'], data['close'])['ADX_14']
    data['OBV'] = ta.obv(data['close'], data['volume'])
    return data

def main(client):
    N_STEPS = MODEL_CONFIG["N_STEPS"]
    EPOCHS = MODEL_CONFIG["EPOCHS"]
    BATCH_SIZE = MODEL_CONFIG["BATCH_SIZE"]

    data = get_historical_klines(
        client=client,
        symbol=SYMBOL,
        interval=INTERVAL,
        start_date=START_DATE,
        logger=logger
    )

    if data.empty:
        logger.error(f"No historical data returned for {SYMBOL} at interval {INTERVAL}. Exiting.")
        raise RuntimeError("No historical data returned.")
    
    data = add_technical_indicators(data)
    data.dropna(inplace=True)

    # Normalize features
    features = ['close', 'SMA_20', 'EMA_50', 'RSI', 'MACD', 'STOCH_RSI', 'BB_UPPER', 'BB_LOWER', 'ADX', 'OBV', 'volume']
    split_idx = int(len(data) * 0.8)

    scaler = MinMaxScaler()
    scaler.fit(data[features].iloc[:split_idx])
    scaled_train = scaler.transform(data[features].iloc[:split_idx])
    scaled_test = scaler.transform(data[features].iloc[split_idx:])

    data_scaled = pd.concat([
        pd.DataFrame(scaled_train, columns=features, index=data.index[:split_idx]),
        pd.DataFrame(scaled_test, columns=features, index=data.index[split_idx:])
    ]).reset_index(drop=True)

    # Create sequences
    X, y = [], []
    for i in range(N_STEPS, len(data_scaled)):
        X.append(data_scaled.iloc[i - N_STEPS:i].values)
        y.append(data_scaled.iloc[i]['close'])

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    # Scale target (close price) – again, fit only on the training part
    y_scaler = MinMaxScaler()
    train_seq_end = split_idx - N_STEPS  # Adjust because first N_STEPS rows have no label
    if train_seq_end <= 0:
        logger.error(f"Not enough data for the chosen N_STEPS and split. Exiting.")
        return
    y_scaler.fit(y[:train_seq_end])
    y_scaled = y_scaler.transform(y)

    logger.info(f"X shape: {X.shape}")
    logger.info(f"y_scaled shape: {y_scaled.shape}")

    X_train, X_test = X[:train_seq_end], X[train_seq_end:]
    y_train, y_test = y_scaled[:train_seq_end], y_scaled[train_seq_end:]

    # Hyperparameter search
    tuner_dir = os.path.join("tuner_dir", SYMBOL, INTERVAL)
    os.makedirs(tuner_dir, exist_ok=True)
    tuner = Hyperband(
        build_model,  # Model-building function
        objective='val_loss',  # Objective metric to minimize
        max_epochs=EPOCHS,  # Maximum number of epochs for each trial
        factor=3,  # Factor to reduce the number of epochs in each subsequent round
        directory=tuner_dir,  # Directory to store the results
        project_name='crypto_tuning',  # Project name for Keras Tuner results
    )

    tuner.search(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))
    best_hyperparameters = tuner.get_best_hyperparameters()[0]
    logger.info(f"Best Hyperparameters: {best_hyperparameters.values}")

    # Train the Model with Best Hyperparameters
    best_model = tuner.hypermodel.build(best_hyperparameters)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        os.path.join(tuner_dir, "best_model_full.keras"),
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )

    best_model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, checkpoint]
    )

    # Predict and invert scale
    predicted_scaled = best_model.predict(X_test)
    predicted_prices = y_scaler.inverse_transform(predicted_scaled)
    y_test_actual = y_scaler.inverse_transform(y_test)

    latest_close = data['close'].iloc[-1]
    predicted_next_price = predicted_prices[-1][0]
    logger.info(f"Latest close: {latest_close}, Predicted next close: {predicted_next_price}")

    mse = mean_squared_error(y_test_actual, predicted_prices)
    rmse = np.sqrt(mse)
    logger.info(f"Mean Squared Error (MSE): {mse}")
    logger.info(f"Root Mean Squared Error (RMSE): {rmse}")

    equity_curve, trades = simulate_grid_trades(
        price_series=data['close'],
        lower_price=GRID_CONFIG["lower_price"],
        upper_price=GRID_CONFIG["upper_price"],
        num_levels=GRID_CONFIG["levels"],
        quantity=GRID_CONFIG["quantity"],
        fee=GRID_CONFIG.get("fee", 0.001),
        logger=logger
    )

    plot_equity_curve_with_trades(equity_curve, pd.DataFrame(trades))

#  def get_with_retry(self, func, *args, **kwargs):
#     max_retries, delay = 5, 3
#     for attempt in range(max_retries):
#         try:
#             return func(*args, **kwargs)
#         except Exception as e:
#             self.logger.warning(f"{Colors.YELLOW}API error ({e}), retrying in {delay}s (attempt {attempt+1}/{max_retries}){Colors.RESET}")
#             time.sleep(delay)
#     raise Exception(f"Failed after {max_retries} retries")

def get_binance_client(max_retries=10):
    retries = 0
    while retries < max_retries:
        try:
            client = Client(API_KEY, API_SECRET)
            if TESTNET:
                client.API_URL = BASE_URL
            return client
        except Exception as e:
            retries += 1
            logger.error(f"Binance Client init failed: {e}. Retrying in 10 seconds... (attempt {retries})")
            time.sleep(10)
    raise RuntimeError("Failed to initialize Binance Client after multiple retries.")

if __name__ == "__main__":
    try:
        client = get_binance_client()
        if args.mode == "trade":
            trader = Trader(
                client=client,
                symbol=SYMBOL,
                lower_price=GRID_CONFIG["lower_price"],
                upper_price=GRID_CONFIG["upper_price"],
                num_levels=GRID_CONFIG["levels"],
                quantity=GRID_CONFIG["quantity"],
                poll_interval=10,
                get_ai_grid_parameters=None,
                logger=logger
            )
            trader.run()
        elif args.mode == "train":
            pass
        else:
            main(client=client)
    except Exception as e:
        logger.exception(f"Error: {e}")
