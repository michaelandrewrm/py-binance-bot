import pandas_ta as ta
# UserWarning: pkg_resources is deprecated as an API.
# See https://setuptools.pypa.io/en/latest/pkg_resources.html. 
# The pkg_resources package is slated for removal as early as 2025-11-30. 
# Refrain from using this package or pin to Setuptools
# <81. from pkg_resources import get_distribution, DistributionNotFound

from data_loader import get_historical_klines
from config import LOOKBACK, SYMBOL, INTERVAL, LOOKBACK, SYMBOL

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Dropout

import matplotlib.pyplot as plt

def test():
    # Plot the closing price
    # plt.figure(figsize=(10, 5))
    # plt.plot(df['open_time'], df['close'], label="Close Price")
    # plt.title(f"{symbol} Price Chart")
    # plt.xlabel("OpenTime")
    # plt.ylabel("Price (USDC)")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    df = get_historical_klines(SYMBOL, INTERVAL, LOOKBACK)

    # Calculate simple moving average (SMA) for the last 20 periods
    df['SMA_20'] = ta.sma(df['close'], length=20)

    # Use percentage price change as an additional feature
    df['pct_change'] = df['close'].pct_change()

    # Drop rows with NaN values
    df = df.dropna()

    print(df)
    print(df.head())

    # Features (SMA_20, pct_change)
    X = df[['SMA_20', 'pct_change']]

    # Target (next period's close price)
    y = df['close'].shift(-1).dropna()

    # Align X and y
    X = X.iloc[:-1]  # Remove the last row because y is shifted

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Example: If the predicted price is higher than the current price, buy. Otherwise, sell.
    current_price = df['close'].iloc[-1]
    predicted_price = y_pred[-1]

    if predicted_price > current_price:
        print("Buy Order Triggered!")
        # Place Buy Order (simulated for now, use Binance API in real case)
        # client.order_market_buy(symbol=symbol, quantity=quantity)
    else:
        print("Sell Order Triggered!")
        # Place Sell Order (simulated for now, use Binance API in real case)
        # client.order_market_sell(symbol=symbol, quantity=quantity)

    # --start--
    # Normalize the 'close' price to be between 0 and 1 (important for LSTM training)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['close_scaled'] = scaler.fit_transform(df[['close']])

    # Create sequences of 'n' previous prices to predict the next one
    n_steps = 60  # 60 time steps (e.g., last 60 minutes of price)

    # Prepare sequences (X) and labels (y)
    X = []
    y = []

    for i in range(n_steps, len(df)):
        X.append(df['close_scaled'][i-n_steps:i].values)
        y.append(df['close_scaled'][i])

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Reshape X to be in the format [samples, time steps, features] for LSTM
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split into train and test sets (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    # --end--

    # Build the LSTM model
    model = Sequential()

    # model.add(Input(shape=))

    # Add the LSTM layer with 50 units (you can experiment with more or fewer units)
    model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))

    # Dropout to avoid overfitting (can be adjusted)
    model.add(Dropout(0.2))

    # Add a Dense layer with 1 unit to output the predicted price
    model.add(Dense(units=1))

    # Compile the model (using Mean Squared Error for regression)
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)


    # Predict on the test data
    predicted_prices = model.predict(X_test)

    # Invert the scaling to get the actual predicted prices
    predicted_prices = scaler.inverse_transform(predicted_prices)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot the results

    plt.figure(figsize=(10, 5))
    plt.plot(y_test_actual, color='blue', label="Actual Price")
    plt.plot(predicted_prices, color='red', label="Predicted Price")
    plt.title(f"{SYMBOL} Price Prediction (LSTM)")
    plt.xlabel("Time")
    plt.ylabel("Price (USDC)")
    plt.legend()
    plt.show()

    # Mean Squared Error (MSE) gives you a measure of how far off the predicted values are from the actual values on average.
    # Root Mean Squared Error (RMSE) is the square root of MSE, which is easier to interpret because it has the same units as the predicted values (e.g., USDT).

    # Calculate MSE and RMSE
    mse = mean_squared_error(y_test_actual, predicted_prices)
    rmse = np.sqrt(mse)

    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")


test()
