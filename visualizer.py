import matplotlib.pyplot as plt

def plot_predictions(df):
    plt.figure(figsize=(14,7))
    plt.plot(df['close'], label='Close Price')
    plt.title('Price Chart')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
