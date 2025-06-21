import matplotlib.pyplot as plt

def plot_predictions(df, predictions=None):
    """
    Plot closing price over time, optionally with model predictions.

    Args:
        df (pd.DataFrame): DataFrame containing 'close' prices.
        predictions (list or np.array, optional): Predicted values to plot alongside close prices.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df["close"], label="Close Price")

    if predictions is not None:
        # Adjust predictions length if needed or align with df index before plotting
        plt.plot(predictions, label="Predictions", linestyle='--')

    plt.title("Close Price Over Time")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
