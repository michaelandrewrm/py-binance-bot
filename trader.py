import numpy as np

def predict_and_trade(model, df, window=24):
    feature_cols = ['SMA_10', 'EMA_10', 'RSI_14', 'MACD', 'MACD_signal', 'BB_High', 'BB_Low', 'Volume_Avg_10']

    # Get latest window of features for prediction
    latest_data = df[feature_cols].iloc[-window:].values

    # Reshape to (1, window, features) for LSTM input
    input_data = np.expand_dims(latest_data, axis=0)

    # Predict probability
    pred_prob = model.predict(input_data)[0][0]
    print(f"Prediction: {pred_prob:.2f}")

    # Simple threshold decision (adjust threshold as you want)
    if pred_prob > 0.6:
        print("Signal: BUY")
        # Here you would place buy order via Binance API
    elif pred_prob < 0.4:
        print("Signal: SELL")
        # Here you would place sell order via Binance API
    else:
        print("No trade executed.")
