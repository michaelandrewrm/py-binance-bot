import os
import numpy as np
import pandas as pd
import pandas_ta as ta
from data_loader import get_historical_klines
from model import build_model
from config import SYMBOL, INTERVAL, START_DATE, N_STEPS, EPOCHS, BATCH_SIZE
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from visualizer import plot_predictions_vs_price_live, plot_backtest_profit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras_tuner import Hyperband
from datetime import timedelta
from backtester import simulate_grid_trades

def add_technical_indicators(data):
    data['SMA_20'] = ta.sma(data['close'], length=20)
    data['RSI'] = ta.rsi(data['close'], length=14)
    data['EMA_50'] = ta.ema(data['close'], length=50)
    return data

def main():
    data = get_historical_klines(SYMBOL, INTERVAL, START_DATE)
    data = add_technical_indicators(data)

    data.dropna(inplace=True)

    # Normalize selected features
    features = ['close', 'SMA_20', 'EMA_50', 'RSI', 'volume']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    data_scaled = pd.DataFrame(scaled_data, columns=features, index=data.index)

    # Create sequences
    X, y = [], []
    for i in range(N_STEPS, len(data_scaled)):
        X.append(data_scaled.iloc[i-N_STEPS:i].values)
        y.append(data_scaled.iloc[i]['close'])

    X = np.array(X)
    y = np.array(y)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Hyperband search method is one of the most efficient methods for tuning hyperparameters
    tuner_dir = f"tuner_dir_{SYMBOL}"
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

    predicted_prices = best_model.predict(X_test)
    predicted_prices = scaler.inverse_transform(np.hstack([predicted_prices] + [np.zeros((len(predicted_prices), len(features)-1))]))[:, 0]
    y_test_actual = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1)] + [np.zeros((len(y_test), len(features)-1))]))[:, 0]

    latest_close = data['close'].iloc[-1]
    predicted_next_price = predicted_prices[-1]
    print(f"Latest close: {latest_close}, Predicted next close: {predicted_next_price}")

    # Calculate MSE and RMSE
    # Mean Squared Error (MSE) gives you a measure of how far off the predicted values are from the actual values on average.
    # Root Mean Squared Error (RMSE) is the square root of MSE, which is easier to interpret because it has the same units as the predicted values (e.g., USDT).
    mse = mean_squared_error(y_test_actual, predicted_prices)
    rmse = np.sqrt(mse)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Define grid parameters
    grid_spacing = 0.01  # 1% up/down
    num_levels = 5

    buy_orders = []
    sell_orders = []

    for i in range(1, num_levels + 1):
        buy_price = latest_close * (1 - i * grid_spacing)
        sell_price = latest_close * (1 + i * grid_spacing)

        if predicted_next_price <= buy_price:
            buy_orders.append((data.index[-1] + timedelta(minutes=60), buy_price))
        elif predicted_next_price >= sell_price:
            sell_orders.append((data.index[-1] + timedelta(minutes=60), sell_price))

    prediction_log = []

    for i in range(len(X_test)):
        timestamp_index = train_size + N_STEPS + i
        if timestamp_index < len(data):
            timestamp = data.index[timestamp_index]
            actual_price = y_test_actual[i]
            predicted_price = predicted_prices[i]
            prediction_log.append({
                "timestamp": timestamp,
                "price": actual_price,
                "prediction": predicted_price
            })

    # plot_candlestick_with_trades(data, buy_orders, sell_orders, predicted_next_price)
    plot_predictions_vs_price_live(prediction_log)

    aligned_data = data.iloc[-len(predicted_prices):]
    trades = simulate_grid_trades(aligned_data, grid_spacing=0.01, levels=5, quantity=100, predictions=predicted_prices)
    plot_backtest_profit(trades)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
