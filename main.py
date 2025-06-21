# main.py

from data_loader import get_historical_klines
from feature_engineering import create_features_labels
from model import build_model
from trader import predict_and_trade
from config import SYMBOL, INTERVAL, WINDOW

from sklearn.model_selection import train_test_split

# Step 1: Load and prepare data
df = get_historical_klines(SYMBOL, INTERVAL)
X, y = create_features_labels(df, WINDOW)

# Step 2: Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = build_model((X.shape[1], 1))
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Step 3: Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

# Step 4: Predict and execute trade
predict_and_trade(model, df, WINDOW)
