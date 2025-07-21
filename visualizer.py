import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import pandas as pd
import mplfinance as mpf
import numpy as np

def plot_equity_curve_with_trades(equity_series, trades, title="Grid Strategy Equity Curve"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.plot(equity_series.index, equity_series.values, label="Equity")
    # Overlay buy/sell markers
    if isinstance(trades, pd.DataFrame):
        buy_trades = trades[trades['side'] == 'buy']
        sell_trades = trades[trades['side'] == 'sell']
        plt.scatter(buy_trades['time'], equity_series.loc[buy_trades['time']], color='green', marker='^', label='Buy', zorder=10)
        plt.scatter(sell_trades['time'], equity_series.loc[sell_trades['time']], color='red', marker='v', label='Sell', zorder=10)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Account Equity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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

def plot_candlestick_with_trades(df, buy_orders, sell_orders, predicted_price):
    df_plot = df[['open', 'high', 'low', 'close', 'volume']].copy()

    buys = pd.DataFrame(buy_orders, columns=['Time', 'Price']) if buy_orders else pd.DataFrame(columns=['Time', 'Price'])
    sells = pd.DataFrame(sell_orders, columns=['Time', 'Price']) if sell_orders else pd.DataFrame(columns=['Time', 'Price'])

    apds = []

    if not buys.empty:
        apds.append(mpf.make_addplot(buys.set_index('Time')['Price'], type='scatter', markersize=150, marker='^', color='green'))
    if not sells.empty:
        apds.append(mpf.make_addplot(sells.set_index('Time')['Price'], type='scatter', markersize=150, marker='v', color='red'))

    predicted_line = pd.Series([predicted_price] * len(df_plot), index=df_plot.index)
    apds.append(mpf.make_addplot(predicted_line, panel=0, color='orange'))

    mpf.plot(df_plot, type='candle', addplot=apds, volume=True, style='charles', title='Candlestick with Grid Trades and Prediction')

def plot_backtest_profit(trades):
    df = pd.DataFrame(trades)
    df['cumulative_profit'] = 0.0
    position = 0

    for i, trade in df.iterrows():
        if trade['action'] == 'BUY':
            position -= trade['quantity'] * trade['price']
        elif trade['action'] == 'SELL':
            position += trade['quantity'] * trade['price']
        df.at[i, 'cumulative_profit'] = float(position)

    plt.figure(figsize=(12, 5))
    plt.plot(df['timestamp'], df['cumulative_profit'], color='lime', marker='o')
    plt.title("Backtested Cumulative Profit", color='white')
    plt.xlabel("Timestamp")
    plt.ylabel("Profit (USDC)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\nSummary:")
    print(f"Total Trades: {len(df)}")
    print(f"Final Profit: {df['cumulative_profit'].iloc[-1]:.2f} USDC")
    print(f"Winning Trades: {(df['action'] == 'SELL').sum()} out of {len(df)//2}")

def backtest_predictions(preds, y_true, close_prices, threshold=0.5):
    signals = (preds >= threshold).astype(int)

    # Entry logic: if signal == 1, buy at today's close, sell at next close
    returns = (np.roll(close_prices, -1) - close_prices) / close_prices
    strategy_returns = returns * signals

    # Avoid the last trade (no next close)
    strategy_returns = strategy_returns[:-1]
    returns = returns[:-1]

    # Metrics
    total_return = np.sum(strategy_returns)
    hit_rate = np.mean((strategy_returns > 0)[signals[:-1] == 1])
    num_trades = np.sum(signals[:-1])
    avg_return_per_trade = total_return / max(num_trades, 1)
    cumulative = np.cumsum(strategy_returns)

    print("\nBacktest Results:")
    print(f"Total return: {total_return:.4f}")
    print(f"Win rate: {hit_rate:.2%}")
    print(f"Number of trades: {num_trades}")
    print(f"Average return/trade: {avg_return_per_trade:.4f}")

    plt.figure(figsize=(10, 4))
    plt.plot(cumulative, label="Strategy Equity Curve")
    plt.title("Backtest Equity Curve")
    plt.xlabel("Trades")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
