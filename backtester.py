import numpy as np
import matplotlib.pyplot as plt

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
