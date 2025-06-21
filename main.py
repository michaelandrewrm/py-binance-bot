import keras_tuner as kt
import os
from feature_engineering import FeatureEngineer
from data_loader import get_historical_klines
from model import build_model
from trader import predict_and_trade
from config import SYMBOL, INTERVAL, WINDOW

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

def balance_data(X, y):
    X_0 = X[y == 0]
    X_1 = X[y == 1]
    y_0 = y[y == 0]
    y_1 = y[y == 1]

    if len(X_0) > len(X_1):
        X_1_upsampled, y_1_upsampled = resample(X_1, y_1, replace=True, n_samples=len(X_0), random_state=42)
        X_bal = np.concatenate([X_0, X_1_upsampled])
        y_bal = np.concatenate([y_0, y_1_upsampled])
    else:
        X_0_upsampled, y_0_upsampled = resample(X_0, y_0, replace=True, n_samples=len(X_1), random_state=42)
        X_bal = np.concatenate([X_0_upsampled, X_1])
        y_bal = np.concatenate([y_0_upsampled, y_1])

    return X_bal, y_bal

def tune_model(X_train, y_train, input_shape):
    os.makedirs("kt_tuner_dir", exist_ok=True)

    tuner = kt.Hyperband(
        lambda hp: build_model(hp, input_shape),
        objective='val_accuracy',
        max_epochs=20,
        factor=3,
        directory='kt_tuner_dir',
        project_name='trade_model_tuning'
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    checkpoint = ModelCheckpoint(
        "kt_tuner_dir/best_model_full.keras",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
        save_weights_only=False,
    )

    tuner.search(
        X_train, y_train,
        epochs=20,
        validation_split=0.1,
        callbacks=[early_stop, reduce_lr, checkpoint]
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("Best hyperparameters:")
    print(f"units1: {best_hp.get('units1')}")
    print(f"dropout1: {best_hp.get('dropout1')}")
    print(f"units2: {best_hp.get('units2')}")
    print(f"dropout2: {best_hp.get('dropout2')}")
    print(f"dense_units: {best_hp.get('dense_units')}")
    print(f"learning_rate: {best_hp.get('learning_rate')}")

    model_path = "kt_tuner_dir/best_model_full.keras"

    if os.path.exists(model_path):
        # Load model without optimizer state to avoid warning
        best_model = load_model(model_path, compile=False)
        best_model.compile(
            optimizer=Adam(learning_rate=best_hp.get("learning_rate")),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
    else:
        print(f"Warning: {model_path} not found, using in-memory best model.")
        best_model = tuner.get_best_models(num_models=1)[0]
        best_model.compile(
            optimizer=Adam(learning_rate=best_hp.get("learning_rate")),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        # Optionally save again if needed
        best_model.save(model_path)

    return best_model

def main():
    print("Fetching historical data...")
    df = get_historical_klines(SYMBOL, INTERVAL)

    print("Creating features and labels...")
    fe = FeatureEngineer(window=WINDOW)
    df, X, y = fe.create_features_labels(df)
    fe.save_scaler("scaler.pkl")

    print("Class distribution before balancing:", np.bincount(y))
    X, y = balance_data(X, y)
    print("Class distribution after balancing:", np.bincount(y))

    print("Splitting data into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_shape = (X.shape[1], X.shape[2])  # (window, number_of_features)

    print("Tuning model hyperparameters...")
    best_model = tune_model(X_train, y_train, input_shape)

    print("Evaluating best model on test set...")
    loss, accuracy = best_model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    print("Making final prediction and trade decision...")
    df_live = get_historical_klines(SYMBOL, INTERVAL)
    fe.load_scaler("scaler.pkl")
    predict_and_trade(best_model, df_live, WINDOW, fe)

if __name__ == "__main__":
    main()
