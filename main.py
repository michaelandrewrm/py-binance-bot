import os
import numpy as np
import pandas as pd
import pandas_ta as ta
import argparse
from data_loader import get_historical_klines
from model import build_model
from config import SYMBOL, START_DATE, get_config_for_interval
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from visualizer import plot_predictions_vs_price_live, plot_backtest_profit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras_tuner import Hyperband
from datetime import timedelta
from backtester import simulate_grid_trades

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=str, default="5m", help="Candlestick interval (e.g.: 5m, 15m, 1h, 4h)")
    args = parser.parse_args()

    INTERVAL = args.interval
    config = get_config_for_interval(INTERVAL)
    N_STEPS = config["model"]["N_STEPS"]
    EPOCHS = config["model"]["EPOCHS"]
    BATCH_SIZE = config["model"]["BATCH_SIZE"]

    data = get_historical_klines(SYMBOL, INTERVAL, START_DATE)
    data = add_technical_indicators(data)
    data.dropna(inplace=True)

    # Normalize features
    features = ['close', 'SMA_20', 'EMA_50', 'RSI', 'MACD', 'STOCH_RSI', 'BB_UPPER', 'BB_LOWER', 'ADX', 'OBV', 'volume']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    data_scaled = pd.DataFrame(scaled_data, columns=features, index=data.index)

    # Create sequences
    X, y = [], []
    for i in range(N_STEPS, len(data_scaled)):
        X.append(data_scaled.iloc[i - N_STEPS:i].values)
        y.append(data_scaled.iloc[i]['close'])  # target is scaled 'close'

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)  # reshape to 2D for scaler

    # Scale target separately
    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y)

    print("X shape:", X.shape)
    print("y_scaled shape:", y_scaled.shape)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

    # Hyperparameter search
    tuner_dir = f"tuner_dir_{SYMBOL}/{INTERVAL}"
    tuner = Hyperband(
        build_model,  # Model-building function
        objective='val_loss',  # Objective metric to minimize
        max_epochs=EPOCHS,  # Maximum number of epochs for each trial
        factor=3,  # Factor to reduce the number of epochs in each subsequent round
        directory=tuner_dir,  # Directory to store the results
        project_name='crypto_lstm_tuning',  # Project name for Keras Tuner results
    )

    tuner.search(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))
    best_hyperparameters = tuner.get_best_hyperparameters()[0]
    print(f"Best Hyperparameters: {best_hyperparameters.values}")

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
    print(f"Latest close: {latest_close}, Predicted next close: {predicted_next_price}")

    mse = mean_squared_error(y_test_actual, predicted_prices)
    rmse = np.sqrt(mse)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Grid prediction logic
    grid_spacing = config["grid"]["grid_spacing"]
    num_levels = config["grid"]["levels"]
    buy_orders, sell_orders = [], []

    for i in range(1, num_levels + 1):
        buy_price = latest_close * (1 - i * grid_spacing)
        sell_price = latest_close * (1 + i * grid_spacing)
        if predicted_next_price <= buy_price:
            buy_orders.append((data.index[-1] + timedelta(minutes=60), buy_price))
        elif predicted_next_price >= sell_price:
            sell_orders.append((data.index[-1] + timedelta(minutes=60), sell_price))

    # Backtest diagnostics log
    prediction_log = []
    for i in range(len(X_test)):
        timestamp_index = train_size + N_STEPS + i
        if timestamp_index < len(data):
            timestamp = data.index[timestamp_index]
            actual_price = y_test_actual[i][0]
            predicted_price = predicted_prices[i][0]
            prediction_log.append({
                "timestamp": timestamp,
                "price": actual_price,
                "prediction": predicted_price
            })

    # Run backtest
    aligned_data = data.iloc[-len(predicted_prices):]
    trades, diagnostics, trade_stats = simulate_grid_trades(aligned_data, predictions=predicted_prices[:, 0], **config["grid"])

    for d in diagnostics[:5]:
        print(d)
    print(trade_stats)

    # Plot results
    plot_backtest_profit(y_test_actual[:, 0], predictions=predicted_prices[:, 0])

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
