import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import pandas as pd

def plot_predictions(actual, predicted, symbol):
    plt.figure(figsize=(10, 5))
    plt.plot(actual, color='blue', label="Actual Price")
    plt.plot(predicted, color='red', label="Predicted Price")
    plt.title(f"{symbol} Actual vs Predicted Prices (LSTM)")
    plt.xlabel("Time")
    plt.ylabel("Price (USDC)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_reliability_diagram(y_true, y_probs, title="Reliability Diagram"):
    """
    Plots a reliability diagram (calibration curve).

    Parameters:
        y_true (array-like): True binary labels.
        y_probs (array-like): Predicted probabilities.
        title (str): Title for the plot.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Platt-Calibrated')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
    plt.title(title)
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_predictions_vs_price(timestamps, prices, calibrated_probs):
    """
    Plots predicted probabilities against actual prices.

    Parameters:
        timestamps (array-like): Time indices.
        prices (array-like): Actual closing prices.
        calibrated_probs (array-like): Calibrated prediction probabilities.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, prices, label="Price", color="blue")
    plt.plot(timestamps, calibrated_probs, label="Calibrated Prediction", color="orange")
    plt.title("Calibrated Predictions vs Price")
    plt.xlabel("Time")
    plt.ylabel("Price / Probability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_cumulative_profit(timestamps, cumulative_profit):
    """
    Optional: Plot cumulative profit over time.

    Parameters:
        timestamps (array-like): Time indices.
        cumulative_profit (array-like): Cumulative profit values.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(timestamps, cumulative_profit, marker='o')
    plt.title("Simulated Cumulative Profit Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Cumulative Profit")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_trade_history(df_trades):
    df_trades['cumulative_profit'] = 0
    position = 0
    for i, trade in df_trades.iterrows():
        if trade['action'] == 'BUY':
            position -= trade['quantity'] * trade['price']
        else:  # SELL
            position += trade['quantity'] * trade['price']
        df_trades.at[i, 'cumulative_profit'] = position

    plt.figure(figsize=(10, 4))
    plt.plot(df_trades['timestamp'], df_trades['cumulative_profit'], marker='o')
    plt.title("Simulated Cumulative Profit Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Cumulative Profit")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_prediction_distribution(pred_history):
    plt.figure(figsize=(6, 4))
    plt.hist(pred_history, bins=20, range=(0, 1), edgecolor='black')
    plt.title("Distribution of Calibrated Predictions")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_predictions_vs_price_live(prediction_log):
    if not prediction_log:
        print("No prediction log data to plot.")
        return

    pred_df = pd.DataFrame(prediction_log)

    fig, ax1 = plt.subplots(figsize=(10, 4))

    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("Price", color='tab:blue')
    ax1.plot(pred_df["timestamp"], pred_df["price"], label="Price", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Prediction", color='tab:orange')
    ax2.plot(pred_df["timestamp"], pred_df["prediction"], label="Prediction", color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title("Prediction vs. Price Over Time")
    fig.tight_layout()
    plt.show()
