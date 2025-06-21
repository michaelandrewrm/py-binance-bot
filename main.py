import keras_tuner as kt
import os
from feature_engineering import FeatureEngineer
from data_loader import get_historical_klines
from model import build_model
from trader import predict_and_trade
from config import SYMBOL, INTERVAL, WINDOW

from sklearn.model_selection import TimeSeriesSplit
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

def tune_model(X_train, y_train, input_shape, symbol):
    tuner_dir = f"kt_tuner_dir_{symbol}"
    os.makedirs(tuner_dir, exist_ok=True)

    tuner = kt.Hyperband(
        lambda hp: build_model(hp, input_shape),
        objective="val_accuracy",
        max_epochs=20,
        factor=3,
        directory=tuner_dir,
        project_name="trade_model_tuning"
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    checkpoint = ModelCheckpoint(
        os.path.join(tuner_dir, "best_model_full.keras"),
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

    model_path = os.path.join(tuner_dir, "best_model_full.keras")

    if os.path.exists(model_path):
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
        best_model.save(model_path)

    return best_model

def retrain_final_model(best_hp, X, y, input_shape):
    print("\nRetraining final model on full dataset with best hyperparameters...")

    model = build_model(best_hp, input_shape)
    model.compile(
        optimizer=Adam(learning_rate=best_hp.get("learning_rate")),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

    model.fit(
        X, y,
        epochs=30,
        validation_split=0.1,
        callbacks=[early_stop, reduce_lr]
    )

    model.save("final_model_full_data.keras")
    print("Final model saved as final_model_full_data.keras")

    return model

def main():
    print("Fetching historical data...")
    df = get_historical_klines(SYMBOL, INTERVAL)

    print("Creating features and labels...")
    fe = FeatureEngineer(window=WINDOW)
    df, X, y = fe.create_features_labels(df)
    fe.save_scaler("scaler.pkl")

    print("Class distribution before balancing:", np.bincount(y))

    input_shape = (X.shape[1], X.shape[2])  # (window, number_of_features)

    tscv = TimeSeriesSplit(n_splits=5)

    fold_accuracies = []
    fold_losses = []

    best_accuracy = -np.inf
    best_model = None
    best_hp_overall = None

    for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
        print(f"\nFold {fold}")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

        X_train_bal, y_train_bal = balance_data(X_train, y_train)
        print("Class distribution after balancing:", np.bincount(y_train_bal))

        print("Tuning model hyperparameters...")
        model, best_hp = tune_model(X_train_bal, y_train_bal, input_shape, SYMBOL)

        print("Evaluating model on test fold...")
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Fold {fold} Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

        fold_accuracies.append(accuracy)
        fold_losses.append(loss)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_hp_overall = best_hp

    print("\nCross-validation results:")
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    print(f"Mean Loss: {np.mean(fold_losses):.4f} ± {np.std(fold_losses):.4f}")

    # Retrain final model on full dataset with best hyperparameters
    final_model = retrain_final_model(best_hp_overall, X, y, input_shape)

    print("\nMaking final prediction and trade decision with final model...")
    df_live = get_historical_klines(SYMBOL, INTERVAL)
    fe.load_scaler("scaler.pkl")
    predict_and_trade(final_model, df_live, WINDOW, fe, simulate=True, symbol=SYMBOL, trade_quantity=1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
