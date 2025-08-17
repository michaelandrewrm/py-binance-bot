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
import optuna  # Replaced keras_tuner with optuna
from binance.client import Client
from archive.config import (
    API_KEY,
    API_SECRET,
    TESTNET,
    BASE_URL,
    SYMBOL,
    SYMBOLS,
    START_DATE,
    get_config_for_symbol,
    get_config_for_interval,
)
from data_loader import get_historical_klines
from model import build_model
from visualizer import plot_equity_curve_with_trades
from archived.backtester import simulate_grid_trades
from trader import Trader
from logger import Logger
from archived.binance_client import get_symbol_info

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--interval",
    type=str,
    default="5m",
    help="Candlestick interval (e.g.: 5m, 15m, 1h, 4h)",
)
parser.add_argument(
    "--mode",
    type=str,
    default="backtest",
    choices=["train", "backtest", "trade"],
)
parser.add_argument(
    "--log-level",
    type=str,
    default="INFO",
    help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)
parser.add_argument(
    "--symbol",
    type=str,
    default=SYMBOL,
    choices=list(SYMBOLS.keys()),
    help="Trading symbol (BTCUSDC/ETHUSDC/WLDUSDC)",
)

args = parser.parse_args()

# Input validation
valid_intervals = {"5m", "15m", "1h", "4h"}
valid_modes = {"train", "backtest", "trade"}
valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
valid_symbols = {"BTCUSDC", "ETHUSDC", "WLDUSDC"}

if args.interval not in valid_intervals:
    raise ValueError(
        f"Invalid interval: {args.interval}. Must be one of {valid_intervals}"
    )
if args.mode not in valid_modes:
    raise ValueError(f"Invalid mode: {args.mode}. Must be one of {valid_modes}")
if args.log_level.upper() not in valid_log_levels:
    raise ValueError(
        f"Invalid log-level: {args.log_level}. Must be one of {valid_log_levels}"
    )
if args.symbol not in valid_symbols:
    raise ValueError(f"Invalid symbol: {args.symbol}. Must be one of {valid_symbols}")


def get_tuner_dir(symbol, interval):
    tuner_dir = os.path.join("tuner_dir", symbol, interval)
    os.makedirs(tuner_dir, exist_ok=True)
    return tuner_dir


def add_technical_indicators(data):
    data["SMA_20"] = ta.sma(data["close"], length=20)
    data["EMA_50"] = ta.ema(data["close"], length=50)
    data["RSI"] = ta.rsi(data["close"], length=14)
    data["MACD"] = ta.macd(data["close"])["MACD_12_26_9"]
    data["STOCH_RSI"] = ta.stochrsi(data["close"])["STOCHRSIk_14_14_3_3"]
    bbands = ta.bbands(data["close"], length=20)
    data["BB_UPPER"] = bbands["BBU_20_2.0"]
    data["BB_LOWER"] = bbands["BBL_20_2.0"]
    data["ADX"] = ta.adx(data["high"], data["low"], data["close"])["ADX_14"]
    data["OBV"] = ta.obv(data["close"], data["volume"])
    return data


def load_and_prepare_data(client, symbol, interval, start_date, logger):
    data = get_historical_klines(
        client=client,
        symbol=symbol,
        interval=interval,
        start_date=start_date,
        logger=logger,
    )

    if data.empty:
        logger.error(f"No historical data for {symbol} at interval {interval}.")
        raise RuntimeError("No historical data returned.")

    data = add_technical_indicators(data)
    data.dropna(inplace=True)
    return data


def load_best_model(app_config):
    BEST_MODEL_PATH = app_config.best_model_path
    if not os.path.exists(BEST_MODEL_PATH):
        app_config.logger.critical(
            f"Model file {BEST_MODEL_PATH} not found. Train the model first."
        )
        raise FileNotFoundError(BEST_MODEL_PATH)
    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    return model


def save_scaler(scaler, path):
    joblib.dump(scaler, path)


def load_scaler(path):
    return joblib.load(path)


def compute_grid_labels(data, idx, window=20, min_levels=5, max_levels=20):
    start = max(0, idx - window)
    end = idx
    segment = data.iloc[start:end]
    recent_low = segment["low"].min()
    recent_high = segment["high"].max()
    volatility = segment["close"].pct_change().std()
    if np.isnan(volatility):
        volatility = 0.0
    # More volatile = more levels (you can tune the scaling)
    levels = int(min_levels + (max_levels - min_levels) * min(volatility / 0.02, 1))
    levels = max(min_levels, min(levels, max_levels))

    grid_width = recent_high - recent_low
    atr = (
        segment["close"].rolling(14).apply(lambda x: np.nanstd(x)).iloc[-1]
    )  # simple stdev/ATR fallback
    stop_loss = recent_low - max(atr, grid_width * 0.15)
    take_profit = recent_high + max(atr, grid_width * 0.15)
    trigger_price = recent_low * 1.002

    return [
        recent_low,
        recent_high,
        levels,
        stop_loss,
        take_profit,
        trigger_price,
    ]


def main(client, app_config):
    logger = app_config.logger
    data = load_and_prepare_data(
        client=client,
        symbol=app_config.symbol,
        interval=app_config.interval,
        start_date=app_config.start_date,
        logger=logger,
    )

    # 0.8: It is chosen to allocate 80% of the data for training and 20% for testing
    split_idx = int(len(data) * 0.8)

    scaler = MinMaxScaler()
    scaler.fit(data[app_config.features].iloc[:split_idx])
    save_scaler(scaler, app_config.scaler_path)
    scaled_train = scaler.transform(data[app_config.features].iloc[:split_idx])
    scaled_test = scaler.transform(data[app_config.features].iloc[split_idx:])

    data_scaled = pd.concat(
        [
            pd.DataFrame(
                scaled_train,
                columns=app_config.features,
                index=data.index[:split_idx],
            ),
            pd.DataFrame(
                scaled_test,
                columns=app_config.features,
                index=data.index[split_idx:],
            ),
        ]
    ).reset_index(drop=True)

    N_STEPS = app_config.model_config["N_STEPS"]
    EPOCHS = app_config.model_config["EPOCHS"]
    BATCH_SIZE = app_config.model_config["BATCH_SIZE"]

    # Create sequences
    x, y = [], []
    for i in range(N_STEPS, len(data_scaled)):
        x.append(data_scaled.iloc[i - N_STEPS : i].values)
        grid_params = compute_grid_labels(data, i + (len(data) - len(data_scaled)))
        y.append(grid_params)

    x = np.array(x)
    y = np.array(y)

    # Scale target (close price)
    y_scaler = MinMaxScaler()
    train_seq_end = split_idx - N_STEPS

    if train_seq_end <= 0:
        logger.error("Not enough data for the chosen N_STEPS and split. Exiting.")
        return

    y_scaler.fit(y[:train_seq_end])
    save_scaler(y_scaler, app_config.y_scaler_path)
    y_scaled = y_scaler.transform(y)

    x_train, x_test = x[:train_seq_end], x[train_seq_end:]
    y_train, y_test = y_scaled[:train_seq_end], y_scaled[train_seq_end:]

    # Define objective function for optuna
    def objective(trial):
        # Create hyperparameter suggestions for optuna
        class OptunaHP:
            def __init__(self, trial):
                self.trial = trial
                self._values = {}

            def Float(self, name, min_val, max_val, step=None):
                if step:
                    value = self.trial.suggest_float(name, min_val, max_val, step=step)
                else:
                    value = self.trial.suggest_float(name, min_val, max_val)
                self._values[name] = value
                return value

            def Int(self, name, min_value, max_value, step=1):
                value = self.trial.suggest_int(name, min_value, max_value, step=step)
                self._values[name] = value
                return value

            def Choice(self, name, values):
                value = self.trial.suggest_categorical(name, values)
                self._values[name] = value
                return value

            def get(self, name):
                return self._values.get(name)

        hp = OptunaHP(trial)
        model = build_model(hp, model_config=app_config.model_config)

        # Train model with early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        history = model.fit(
            x_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=[early_stopping],
            verbose=0,
        )

        # Return the best validation loss
        return min(history.history["val_loss"])

    # Hyperparameter search with optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)  # Adjust n_trials as needed

    best_params = study.best_params
    logger.info(f"Best Hyperparameters: {best_params}")

    # Train the Model with Best Hyperparameters
    class BestHP:
        def __init__(self, params):
            self.params = params

        def Float(self, name, min_val, max_val, step=None):
            return self.params[name]

        def Int(self, name, min_value, max_value, step=1):
            return self.params[name]

        def Choice(self, name, values):
            return self.params[name]

        def get(self, name):
            return self.params.get(name)

    best_hp = BestHP(best_params)
    best_model = build_model(best_hp, model_config=app_config.model_config)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        app_config.best_model_path,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
    )
    best_model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping, checkpoint],
    )

    # Predict and invert scale
    predicted_scaled = best_model.predict(x_test)
    predicted_prices = y_scaler.inverse_transform(predicted_scaled)
    y_test_actual = y_scaler.inverse_transform(y_test)

    logger.info(
        f"Latest close: {data['close'].iloc[-1]:.3f}, Predicted next close: {predicted_prices[-1][0]}"
    )

    mse = mean_squared_error(y_test_actual, predicted_prices)
    rmse = np.sqrt(mse)

    logger.info(f"Mean Squared Error (MSE): {mse}")
    logger.info(f"Root Mean Squared Error (RMSE): {rmse}")

    equity_curve, trades = simulate_grid_trades(
        price_series=data["close"],
        lower_price=app_config.grid_config["lower_price"],
        upper_price=app_config.grid_config["upper_price"],
        num_levels=app_config.grid_config["levels"],
        quantity=app_config.grid_config["quantity"],
        fee=app_config.grid_config.get("fee", 0.001),
        logger=logger,
    )

    plot_equity_curve_with_trades(equity_curve, pd.DataFrame(trades))


def get_with_retry(func, *args, max_retries=10, **kwargs):
    retries = 0
    while retries < max_retries:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            retries += 1
            logger.error(f"Error: {e}. Retrying in 10 seconds... (attempt {retries})")
            time.sleep(10)
    raise RuntimeError("Failed to invoke the function after multiple retries.")


def get_ai_grid_parameters(client, app_config):
    """
    Predicts grid parameters using the trained AI model.
    The output is always clamped to bracket the current price and to reasonable width.
    Falls back to ATR-based width if model prediction is nonsensical.
    """

    logger = app_config.logger
    model = load_best_model(app_config)
    data = load_and_prepare_data(
        client=client,
        symbol=app_config.symbol,
        interval=app_config.interval,
        start_date=app_config.start_date,
        logger=logger,
    )
    scaler = load_scaler(app_config.scaler_path)
    y_scaler = load_scaler(app_config.y_scaler_path)
    N_STEPS = app_config.model_config["N_STEPS"]
    QUANTITY = app_config.grid_config["quantity"]

    # Prepare the latest N_STEPS of features
    latest_features = data[app_config.features].iloc[-N_STEPS:]
    latest_scaled = scaler.transform(latest_features)
    latest_sequence = np.array(latest_scaled).reshape(
        1, N_STEPS, len(app_config.features)
    )

    # Predict and inverse-transform model output
    pred_scaled = model.predict(latest_sequence)
    pred_params = y_scaler.inverse_transform(pred_scaled)[0]
    (
        pred_lower,
        pred_upper,
        pred_levels,
        pred_stop_loss,
        pred_take_profit,
        pred_trigger,
    ) = pred_params

    current_price = data["close"].iloc[-1]
    levels = int(round(pred_levels))
    levels = max(2, min(levels, 50))

    # Compute ATR fallback grid width
    atr = ta.atr(data["high"], data["low"], data["close"], length=14).iloc[-1]
    fallback_width = max(current_price * 0.005, atr * 2)  # at least 0.5% or 2x ATR

    # Calculate model width, clamp to sensible bounds
    model_width = abs(pred_upper - pred_lower)
    min_width = max(current_price * 0.003, atr * 1.2)
    max_width = current_price * 0.15

    if np.isnan(model_width) or model_width < min_width or model_width > max_width:
        logger.warning(
            f"Model grid width {model_width:.6f} is out of range ({min_width:.6f}-{max_width:.6f}). "
            f"Falling back to ATR width: {fallback_width:.6f}."
        )
        grid_width = fallback_width
    else:
        grid_width = model_width

    # Center grid on current price
    lower_price = current_price - grid_width / 2
    upper_price = current_price + grid_width / 2

    # Clamp prices to be strictly positive and in a valid order
    lower_price = max(0.00001, lower_price)
    upper_price = max(lower_price + 0.00001, upper_price)

    # Final safety: refuse if the grid still doesn't bracket the price
    if not (lower_price < current_price < upper_price):
        logger.warning(
            f"Final grid does not bracket price. Re-centering on current price with width {grid_width:.6f}."
        )
        lower_price = current_price - grid_width / 2
        upper_price = current_price + grid_width / 2

    grid_params = {
        "lower_price": round(lower_price, 6),
        "upper_price": round(upper_price, 6),
        "num_levels": levels,
        "quantity": float(QUANTITY),
        "pred_stop_loss": pred_stop_loss,
        "pred_take_profit": pred_take_profit,
        "pred_trigger": pred_trigger,
    }

    logger.info(
        f"AI Grid Params: lower={grid_params['lower_price']} upper={grid_params['upper_price']} "
        f"levels={grid_params['num_levels']} current={current_price:.6f} "
        f"width={grid_width:.6f} (ATR={atr:.6f})"
    )

    return grid_params


def get_binance_client(logger) -> Client:
    if not API_KEY or not API_SECRET:
        raise RuntimeError(
            "API_KEY and API_SECRET must be set as environment variables or in config."
        )
    try:
        client = Client(API_KEY, API_SECRET)
        if TESTNET:
            client.API_URL = BASE_URL
    except Exception as e:
        logger.error(f"Failed to instantiate Binance client: {e}")
        raise RuntimeError(
            "Failed to instantiate Binance client. Please check your credentials and network connection."
        ) from e
    return client


class AppConfig:
    def __init__(self, interval, symbol, start_date):
        self.interval = interval
        self.symbol = symbol
        self.start_date = start_date
        self.config = get_config_for_interval(interval, symbol)
        self.model_config = self.config["model"]
        self.grid_config = self.config["grid"]
        self.features = [
            "close",
            "SMA_20",
            "EMA_50",
            "RSI",
            "MACD",
            "STOCH_RSI",
            "BB_UPPER",
            "BB_LOWER",
            "ADX",
            "OBV",
            "volume",
        ]
        self.tuner_dir = get_tuner_dir(symbol, interval)
        self.best_model_path = os.path.join(self.tuner_dir, "best_model_full.keras")
        self.scaler_path = os.path.join(self.tuner_dir, "scaler.save")
        self.y_scaler_path = os.path.join(self.tuner_dir, "y_scaler.save")
        self.logger = Logger(
            level=getattr(logging, args.log_level.upper(), logging.INFO)
        )


if __name__ == "__main__":
    app_config = AppConfig(args.interval, SYMBOL, START_DATE)
    logger = app_config.logger

    try:
        client = get_with_retry(get_binance_client, logger=logger)
        live = get_symbol_info(client=client, symbol=app_config.symbol)
        app_config.config.update(live)
        # predict_params = get_ai_grid_parameters(client=client, app_config=app_config)

        if args.mode == "trade":
            trader = Trader(
                client=client,
                symbol=app_config.symbol,
                lower_price=app_config.grid_config["lower_price"],
                upper_price=app_config.grid_config["upper_price"],
                num_levels=app_config.grid_config["levels"],
                quantity=app_config.grid_config["quantity"],
                poll_interval=10,
                get_ai_grid_parameters=lambda: get_ai_grid_parameters(
                    client, app_config
                ),
                logger=logger,
                trailing_up=False,
                grid_trigger_price=30000,
                stop_loss_price=25000,
                take_profit_price=35000,
                sell_all_on_stop=True,
                investment_amount=1000,
            )
            trader.run()
        elif args.mode == "train":
            pass
        else:
            main(client=client, app_config=app_config)
    except Exception as e:
        logger.exception(f"Error: {e}")
