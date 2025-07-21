import os
import argparse
import random
import time
import logging
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras_tuner import Hyperband
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
FEATURES = ["close", "SMA_20", "EMA_50", "RSI", "MACD", "STOCH_RSI", "BB_UPPER", "BB_LOWER", "ADX", "OBV", "volume"]
TUNER_DIR = os.path.join("tuner_dir", SYMBOL, INTERVAL)

os.makedirs(TUNER_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(TUNER_DIR, "best_model_full.keras")
SCALER_PATH = os.path.join(TUNER_DIR, "scaler.save")
Y_SCALER_PATH = os.path.join(TUNER_DIR, "y_scaler.save")

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

def load_and_prepare_data(client, symbol, interval, start_date, logger):
    data = get_historical_klines(
        client=client,
        symbol=symbol,
        interval=interval,
        start_date=start_date,
        logger=logger
    )
    
    if data.empty:
        logger.error(f"No historical data for {symbol} at interval {interval}.")
        raise RuntimeError("No historical data returned.")

    data = add_technical_indicators(data)
    data.dropna(inplace=True)
    return data

def load_best_model():
    if not os.path.exists(BEST_MODEL_PATH):
        logger.error(f"Model file {BEST_MODEL_PATH} not found. Train the model first.")
        raise FileNotFoundError(BEST_MODEL_PATH)
    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    return model

def save_scaler(scaler, path):
    joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)

def main(client):
    data = load_and_prepare_data(
        client=client,
        symbol=SYMBOL,
        interval=INTERVAL,
        start_date=START_DATE,
        logger=logger
    )

    # 0.8: It is chosen to allocate 80% of the data for training and 20% for testing
    split_idx = int(len(data) * 0.8)

    scaler = MinMaxScaler()
    scaler.fit(data[FEATURES].iloc[:split_idx])
    save_scaler(scaler, SCALER_PATH)
    scaled_train = scaler.transform(data[FEATURES].iloc[:split_idx])
    scaled_test = scaler.transform(data[FEATURES].iloc[split_idx:])

    data_scaled = pd.concat([
        pd.DataFrame(scaled_train, columns=FEATURES, index=data.index[:split_idx]),
        pd.DataFrame(scaled_test, columns=FEATURES, index=data.index[split_idx:])
    ]).reset_index(drop=True)

    N_STEPS = MODEL_CONFIG["N_STEPS"]
    EPOCHS = MODEL_CONFIG["EPOCHS"]
    BATCH_SIZE = MODEL_CONFIG["BATCH_SIZE"]

    # Create sequences
    x, y = [], []
    for i in range(N_STEPS, len(data_scaled)):
        x.append(data_scaled.iloc[i - N_STEPS:i].values)
        y.append(data_scaled.iloc[i]['close'])

    x = np.array(x)
    y = np.array(y).reshape(-1, 1)

    # Scale target (close price)
    y_scaler = MinMaxScaler()
    train_seq_end = split_idx - N_STEPS
    if train_seq_end <= 0:
        logger.error(f"Not enough data for the chosen N_STEPS and split. Exiting.")
        return
    y_scaler.fit(y[:train_seq_end])
    save_scaler(y_scaler, Y_SCALER_PATH)
    y_scaled = y_scaler.transform(y)

    x_train, x_test = x[:train_seq_end], x[train_seq_end:]
    y_train, y_test = y_scaled[:train_seq_end], y_scaled[train_seq_end:]

    # Hyperparameter search
    tuner = Hyperband(
        build_model,  # Model-building function
        objective='val_loss',  # Objective metric to minimize
        max_epochs=EPOCHS,  # Maximum number of epochs for each trial
        factor=3,  # Factor to reduce the number of epochs in each subsequent round
        directory=TUNER_DIR,  # Directory to store the results
        project_name='crypto_tuning',  # Project name for Keras Tuner results
    )

    tuner.search(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test))
    best_hyperparameters = tuner.get_best_hyperparameters()[0]
    
    logger.info(f"Best Hyperparameters: {best_hyperparameters.values}")

    # Train the Model with Best Hyperparameters
    best_model = tuner.hypermodel.build(best_hyperparameters)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        BEST_MODEL_PATH,
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )
    best_model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping, checkpoint]
    )

    # Predict and invert scale
    predicted_scaled = best_model.predict(x_test)
    predicted_prices = y_scaler.inverse_transform(predicted_scaled)
    y_test_actual = y_scaler.inverse_transform(y_test)

    logger.info(f"Latest close: {data['close'].iloc[-1]:.3f}, Predicted next close: {predicted_prices[-1][0]}")

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

def get_with_retry(func, max_retries=10, ):
    retries = 0
    while retries < max_retries:
        try:
            return func()
        except Exception as e:
            retries += 1
            logger.error(f"Error: {e}. Retrying in 10 seconds... (attempt {retries})")
            time.sleep(10)
    raise RuntimeError("Failed to invoke the function after multiple retries.")

def get_binance_client():
    client = Client(API_KEY, API_SECRET)
    if TESTNET:
        client.API_URL = BASE_URL
    return client

def get_ai_grid_parameters():
    model = load_best_model()
    data = load_and_prepare_data(
        client=client,
        symbol=SYMBOL,
        interval=INTERVAL,
        start_date=START_DATE,
        logger=logger
    )
    scaler = load_scaler(SCALER_PATH)
    y_scaler = load_scaler(Y_SCALER_PATH)

    N_STEPS = MODEL_CONFIG["N_STEPS"]
    GRID_RANGE = 0.05
    LEVELS = GRID_CONFIG["levels"]
    QUANTITY = GRID_CONFIG["quantity"]

    # Prepare the latest N_STEPS window from your full data
    latest_features = data[FEATURES].iloc[-N_STEPS:]
    latest_scaled = scaler.transform(latest_features)
    latest_sequence = np.array(latest_scaled).reshape(1, N_STEPS, len(FEATURES))

    # Predict the next close price (scaled)
    pred_scaled = model.predict(latest_sequence)
    pred_price = y_scaler.inverse_transform(pred_scaled)[0, 0]

    lower_price = pred_price * (1 - GRID_RANGE)
    upper_price = pred_price * (1 + GRID_RANGE)

    grid_params = {
        "lower_price": float(lower_price),
        "upper_price": float(upper_price),
        "num_levels": int(LEVELS),
        "quantity": float(QUANTITY)
    }

    return grid_params, pred_price

if __name__ == "__main__":
    try:
        client = get_with_retry(get_binance_client)
        if args.mode == "trade":
            trader = Trader(
                client=client,
                symbol=SYMBOL,
                lower_price=GRID_CONFIG["lower_price"],
                upper_price=GRID_CONFIG["upper_price"],
                num_levels=GRID_CONFIG["levels"],
                quantity=GRID_CONFIG["quantity"],
                poll_interval=10,
                get_ai_grid_parameters=get_ai_grid_parameters,
                logger=logger
            )
            trader.run()
        elif args.mode == "train":
            pass
        else:
            main(client=client)
    except Exception as e:
        logger.exception(f"Error: {e}")
