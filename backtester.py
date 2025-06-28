from datetime import timedelta

def simulate_grid_trades(
    df,
    grid_spacing=0.01,
    levels=3,
    quantity=100,
    predictions=None,
    base_take_profit_pct=0.015,
    base_stop_loss_pct=0.015,  # Adjusted SL to match TP
    min_prediction_change=0.0035,  # Require higher prediction confidence
    volatility_window=10,
    cooldown_period=timedelta(hours=2)  # Cooldown between trades
):
    """
    Simulate grid trading with prediction direction, including short trades.

    Args:
        df (pd.DataFrame): Price data with 'close' and datetime index.
        grid_spacing (float): Grid % spacing.
        levels (int): Grid levels.
        quantity (float): Quantity per trade.
        predictions (list): Optional predicted prices aligned with df.
        base_take_profit_pct (float): Base profit target per trade.
        base_stop_loss_pct (float): Base max loss per trade.
        min_prediction_change (float): Minimum predicted movement to trigger trade.
        volatility_window (int): Number of candles for volatility (std) calculation.
        cooldown_period (timedelta): Minimum wait time between trades.

    Returns:
        list of dict: Simulated trades.
    """
    df = df.copy()
    trades = []
    long_position = None
    short_position = None
    last_trade_time = None

    # Calculate rolling volatility (standard deviation)
    df.loc[:, 'volatility'] = df['close'].rolling(window=volatility_window).std().bfill()

    for i in range(1, len(df)):
        current_price = df['close'].iloc[i]
        current_time = df.index[i]
        volatility = df['volatility'].iloc[i]

        # Skip trading in ultra-low volatility zones
        if volatility < 0.002:
            continue

        # Volatility-adjusted thresholds
        take_profit_pct = base_take_profit_pct * (1 + volatility)
        stop_loss_pct = base_stop_loss_pct * (1 + volatility)

        prediction_trend = 0
        if predictions is not None and i < len(predictions):
            predicted_price = predictions[i]
            prediction_trend = predicted_price - df['close'].iloc[i - 1]

        base_price = df['close'].iloc[i - 1]
        adaptive_spacing = grid_spacing * (1 + volatility * 2)
        buy_levels = [base_price * (1 - adaptive_spacing * j) for j in range(1, levels + 1)]
        sell_levels = [base_price * (1 + adaptive_spacing * j) for j in range(1, levels + 1)]

        # Manage long position
        if long_position:
            if current_price >= long_position['price'] * (1 + take_profit_pct):
                trades.append({
                    'timestamp': current_time,
                    'action': 'SELL (TP)',
                    'price': current_price,
                    'quantity': quantity
                })
                long_position = None
                last_trade_time = current_time
            elif current_price <= long_position['price'] * (1 - stop_loss_pct):
                trades.append({
                    'timestamp': current_time,
                    'action': 'SELL (SL)',
                    'price': current_price,
                    'quantity': quantity
                })
                long_position = None
                last_trade_time = current_time

        # Manage short position
        if short_position:
            if current_price <= short_position['price'] * (1 - take_profit_pct):
                trades.append({
                    'timestamp': current_time,
                    'action': 'BUY to Cover (TP)',
                    'price': current_price,
                    'quantity': quantity
                })
                short_position = None
                last_trade_time = current_time
            elif current_price >= short_position['price'] * (1 + stop_loss_pct):
                trades.append({
                    'timestamp': current_time,
                    'action': 'BUY to Cover (SL)',
                    'price': current_price,
                    'quantity': quantity
                })
                short_position = None
                last_trade_time = current_time

        can_trade = last_trade_time is None or (current_time - last_trade_time) >= cooldown_period

        # Open new long if conditions match
        if prediction_trend > min_prediction_change and long_position is None and can_trade:
            for bp in buy_levels:
                if current_price <= bp:
                    trades.append({
                        'timestamp': current_time,
                        'action': 'BUY',
                        'price': current_price,
                        'quantity': quantity
                    })
                    long_position = {
                        'price': current_price,
                        'time': current_time
                    }
                    last_trade_time = current_time
                    break

        # Open new short if conditions match
        elif prediction_trend < -min_prediction_change and short_position is None and can_trade:
            for sp in sell_levels:
                if current_price >= sp:
                    trades.append({
                        'timestamp': current_time,
                        'action': 'SELL (Short)',
                        'price': current_price,
                        'quantity': quantity
                    })
                    short_position = {
                        'price': current_price,
                        'time': current_time
                    }
                    last_trade_time = current_time
                    break

    return trades
