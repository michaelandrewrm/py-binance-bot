import keras_tuner as kt
import os
import numpy as np
# import joblib
# from feature_engineering import FeatureEngineer
from data_loader import get_historical_klines
from model import build_model
# from trader import TradeSimulator
from config import SYMBOL, INTERVAL, WINDOW, START_DATE, N_STEPS, EPOCHS, BATCH_SIZE

# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.utils import resample
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from tensorflow.keras.models import load_model
# from tensorflow.keras.optimizers import Adam
# from sklearn.linear_model import LogisticRegression
from visualizer import plot_predictions
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras_tuner import Hyperband, HyperParameters

import pandas_ta as ta

# def balance_data(X, y):
#     X_0 = X[y == 0]
#     X_1 = X[y == 1]
#     y_0 = y[y == 0]
#     y_1 = y[y == 1]

#     if len(X_0) > len(X_1):
#         X_1_upsampled, y_1_upsampled = resample(X_1, y_1, replace=True, n_samples=len(X_0), random_state=42)
#         X_bal = np.concatenate([X_0, X_1_upsampled])
#         y_bal = np.concatenate([y_0, y_1_upsampled])
#     else:
#         X_0_upsampled, y_0_upsampled = resample(X_0, y_0, replace=True, n_samples=len(X_1), random_state=42)
#         X_bal = np.concatenate([X_0_upsampled, X_1])
#         y_bal = np.concatenate([y_0_upsampled, y_1])

#     return X_bal, y_bal

# def tune_model(X_train, y_train, input_shape, symbol):
#     tuner_dir = f"kt_tuner_dir_{symbol}"
#     os.makedirs(tuner_dir, exist_ok=True)

#     tuner = kt.Hyperband(
#         lambda hp: build_model(hp, input_shape),
#         objective="val_accuracy",
#         max_epochs=20,
#         factor=3,
#         directory=tuner_dir,
#         project_name="trade_model_tuning"
#     )

#     early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
#     reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
#     checkpoint = ModelCheckpoint(
#         os.path.join(tuner_dir, "best_model_full.keras"),
#         monitor="val_loss",
#         save_best_only=True,
#         mode="min",
#         verbose=1,
#         save_weights_only=False,
#     )

#     tuner.search(
#         X_train, y_train,
#         epochs=20,
#         validation_split=0.1,
#         callbacks=[early_stop, reduce_lr, checkpoint]
#     )

#     best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

#     model_path = os.path.join(tuner_dir, "best_model_full.keras")

#     if os.path.exists(model_path):
#         best_model = load_model(model_path, compile=False)
#         best_model.compile(
#             optimizer=Adam(learning_rate=best_hp.get("learning_rate")),
#             loss="binary_crossentropy",
#             metrics=["accuracy"]
#         )
#     else:
#         best_model = tuner.get_best_models(num_models=1)[0]
#         best_model.compile(
#             optimizer=Adam(learning_rate=best_hp.get("learning_rate")),
#             loss="binary_crossentropy",
#             metrics=["accuracy"]
#         )
#         best_model.save(model_path)

#     return best_model, best_hp

# def retrain_final_model(best_hp, X, y, input_shape):
#     model = build_model(best_hp, input_shape)
#     model.compile(
#         optimizer=Adam(learning_rate=best_hp.get("learning_rate")),
#         loss="binary_crossentropy",
#         metrics=["accuracy"]
#     )

#     early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
#     reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

#     model.fit(
#         X, y,
#         epochs=30,
#         validation_split=0.1,
#         callbacks=[early_stop, reduce_lr]
#     )

#     model.save("final_model_full_data.keras")
#     return model

# def main():
#     df = get_historical_klines(SYMBOL, INTERVAL)
#     fe = FeatureEngineer(window=WINDOW)
#     df, X, y = fe.create_features_labels(df)
#     fe.save_scaler("scaler.pkl")

#     input_shape = (X.shape[1], X.shape[2])
#     tscv = TimeSeriesSplit(n_splits=5)

#     fold_accuracies, fold_losses = [], []
#     best_accuracy, best_model, best_hp_overall = -np.inf, None, None

#     all_probs, all_true = [], []

#     for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]

#         X_train_bal, y_train_bal = balance_data(X_train, y_train)
#         model, best_hp = tune_model(X_train_bal, y_train_bal, input_shape, SYMBOL)

#         loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
#         y_probs = model.predict(X_test).flatten()
#         all_probs.extend(y_probs)
#         all_true.extend(y_test)

#         fold_accuracies.append(accuracy)
#         fold_losses.append(loss)

#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_model = model
#             best_hp_overall = best_hp

#     print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
#     print(f"Mean Loss: {np.mean(fold_losses):.4f} ± {np.std(fold_losses):.4f}")

#     # Fit and apply Platt scaling
#     platt_model = LogisticRegression()
#     platt_model.fit(np.array(all_probs).reshape(-1, 1), all_true)
#     joblib.dump(platt_model, "platt_scaler.pkl")
#     calibrated_probs = platt_model.predict_proba(np.array(all_probs).reshape(-1, 1))[:, 1]

#     # Reliability diagram
#     plot_reliability_diagram(all_true, calibrated_probs)

#     final_model = retrain_final_model(best_hp_overall, X, y, input_shape)
#     df_live = get_historical_klines(SYMBOL, INTERVAL)
#     fe.load_scaler("scaler.pkl")

#     simulator = TradeSimulator(final_model, platt_model, WINDOW, fe, symbol=SYMBOL, trade_quantity=1)
#     simulator.predict_and_trade(df_live, simulate=True)

def add_technical_indicators(data):
    # Calculate SMA (Simple Moving Average) for the last 20 periods
    data['SMA_20'] = ta.sma(data['close'], length=20)

    # Calculate RSI (Relative Strength Index)
    data['RSI'] = ta.rsi(data['close'], length=14)

    # Add other indicators if needed
    # data['EMA_50'] = ta.ema(data['close'], length=50)
    
    # Return the modified DataFrame
    return data

def main():
    # Step 1: Load historical data
    data = get_historical_klines(SYMBOL, INTERVAL, START_DATE)
    data = add_technical_indicators(data)

    # Step 2: Prepare data for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['close_scaled'] = scaler.fit_transform(data[['close']])

    # Create sequences
    X, y = [], []
    for i in range(N_STEPS, len(data)):
        X.append(data['close_scaled'][i-N_STEPS:i].values)
        y.append(data['close_scaled'][i])

    X = np.array(X)
    y = np.array(y)

    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Step 3: Split into train and test sets
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

    # Define the hyperparameters search space
    # tuner.oracle.hyperparameters = [
    #     HyperParameters.Int('units', min_value=50, max_value=200, step=50),  # Number of LSTM units
    #     HyperParameters.Int('batch_size', values=[16, 32, 64]),  # Batch size
    #     HyperParameters.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log'),  # Learning rate
    # ]

    # Perform Hyperparameter Search
    tuner.search(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

    # Get the Best Hyperparameters
    best_hyperparameters = tuner.get_best_hyperparameters()[0]
    print(f"Best Hyperparameters: {best_hyperparameters.values}")

    # Train the Model with Best Hyperparameters
    best_model = tuner.hypermodel.build(best_hyperparameters)

    # Use early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Model checkpoint to save the best model
    checkpoint = ModelCheckpoint(
        os.path.join(tuner_dir, "best_model_full.keras"),
        save_best_only=True, 
        monitor='val_loss', 
        mode='min'
    )

    best_model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test),
                   callbacks=[early_stopping, checkpoint])

    # Step 4: Build and train the LSTM model
    # model = build_model(X_train)
    # Train the model with early stopping
    # model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Step 5: Make predictions
    predicted_prices = best_model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate MSE and RMSE
    # Mean Squared Error (MSE) gives you a measure of how far off the predicted values are from the actual values on average.
    # Root Mean Squared Error (RMSE) is the square root of MSE, which is easier to interpret because it has the same units as the predicted values (e.g., USDT).
    mse = mean_squared_error(y_test_actual, predicted_prices)
    rmse = np.sqrt(mse)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Step 6: Visualize the results
    plot_predictions(y_test_actual, predicted_prices, symbol=SYMBOL)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
