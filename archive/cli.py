import questionary

def main():
    mode = questionary.select(
        "Select mode",
        choices=["train", "backtest", "trade"]
    ).ask()

    symbol = questionary.select(   # Use select, not prompt
        "Select symbol",
        choices=["BTCUSDC", "ETHUSDC", "WLDUSDC"]
    ).ask()
    interval = questionary.select(
        "Select interval",
        choices=["5m", "15m", "1h", "4h"]
    ).ask()

    return {
        "mode": mode,
        "symbol": symbol,
        "interval": interval
    }

if __name__ == "__main__":
    main()
