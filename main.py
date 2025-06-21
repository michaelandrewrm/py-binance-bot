from data_loader import get_historical_klines
from feature_engineering import create_features_labels
from model import build_model
from trader import predict_and_trade
from config import SYMBOL, INTERVAL, WINDOW

from sklearn.model_selection import train_test_split

def main():
    # Step 1: Load raw data from Binance
    df = get_historical_klines(SYMBOL, INTERVAL)

    # Step 2: Create features & labels, get updated df with features
    df, X, y = create_features_labels(df, WINDOW)

    # Step 3: Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Build LSTM model (input shape = (window, feature_count))
    model = build_model((X.shape[1], X.shape[2]))

    # Step 5: Train model
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Step 6: Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Step 7: Use latest data to predict & decide trade
    predict_and_trade(model, df, WINDOW)

if __name__ == "__main__":
    main()
