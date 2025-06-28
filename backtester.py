from datetime import timedelta

def simulate_grid_trades(
    df,
    grid_spacing=0.002,
    levels=3,
    quantity=100,
    predictions=None,
    base_take_profit_pct=0.004,
    base_stop_loss_pct=0.004,
    min_prediction_change=0.0015,
    volatility_window=10,
    cooldown_period=timedelta(minutes=15)
):
    df = df.copy()
    trades = []
    long_position = None
    short_position = None
    last_trade_time = None

    df.loc[:, 'volatility'] = df['close'].rolling(window=volatility_window).std().bfill()

    for i in range(1, len(df)):
        current_price = df['close'].iloc[i]
        current_time = df.index[i]
        volatility = df['volatility'].iloc[i]

        if volatility < 0.002:
            continue

        take_profit_pct = base_take_profit_pct * (1 + volatility)
        stop_loss_pct = base_stop_loss_pct * (1 + volatility)
        trailing_trigger_pct = take_profit_pct * 0.5
        trailing_buffer_pct = 0.003

        prediction_trend = 0
        if predictions is not None and i < len(predictions):
            predicted_price = predictions[i]
            prediction_trend = predicted_price - df['close'].iloc[i - 1]

        base_price = df['close'].iloc[i - 1]
        adaptive_spacing = grid_spacing * (1 + volatility * 2)
        buy_levels = [base_price * (1 - adaptive_spacing * j) for j in range(1, levels + 1)]
        sell_levels = [base_price * (1 + adaptive_spacing * j) for j in range(1, levels + 1)]

        if long_position:
            entry_price = long_position['price']
            sl_price = entry_price * (1 - stop_loss_pct)
            max_price = long_position.get('max_price', entry_price)

            if current_price > max_price:
                max_price = current_price

            if (max_price / entry_price - 1) >= trailing_trigger_pct:
                sl_price = max(sl_price, max_price * (1 - trailing_buffer_pct))

            long_position['max_price'] = max_price

            if current_price >= entry_price * (1 + take_profit_pct):
                trades.append({
                    'timestamp': current_time,
                    'action': 'SELL (TP)',
                    'price': current_price,
                    'quantity': quantity
                })
                long_position = None
                last_trade_time = current_time
            elif current_price <= sl_price:
                trades.append({
                    'timestamp': current_time,
                    'action': 'SELL (SL)',
                    'price': current_price,
                    'quantity': quantity
                })
                long_position = None
                last_trade_time = current_time

        if short_position:
            entry_price = short_position['price']
            sl_price = entry_price * (1 + stop_loss_pct)
            min_price = short_position.get('min_price', entry_price)

            if current_price < min_price:
                min_price = current_price

            if (1 - min_price / entry_price) >= trailing_trigger_pct:
                sl_price = min(sl_price, min_price * (1 + trailing_buffer_pct))

            short_position['min_price'] = min_price

            if current_price <= entry_price * (1 - take_profit_pct):
                trades.append({
                    'timestamp': current_time,
                    'action': 'BUY to Cover (TP)',
                    'price': current_price,
                    'quantity': quantity
                })
                short_position = None
                last_trade_time = current_time
            elif current_price >= sl_price:
                trades.append({
                    'timestamp': current_time,
                    'action': 'BUY to Cover (SL)',
                    'price': current_price,
                    'quantity': quantity
                })
                short_position = None
                last_trade_time = current_time

        can_trade = last_trade_time is None or (current_time - last_trade_time) >= cooldown_period

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
                        'time': current_time,
                        'max_price': current_price
                    }
                    last_trade_time = current_time
                    break

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
                        'time': current_time,
                        'min_price': current_price
                    }
                    last_trade_time = current_time
                    break

    return trades
