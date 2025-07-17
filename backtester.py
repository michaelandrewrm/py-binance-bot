import numpy as np

def simulate_grid_trades(
    df,
    grid_spacing,
    levels,
    quantity,
    predictions,
    volatility_window,
    min_volatility_threshold,
    min_prediction_change,
    min_expected_gain,
    max_trades_per_day,
    max_daily_loss,
    cooldown_period,
    position_timeout_minutes,
    trailing_trigger_factor,
    take_profit_pct,
    stop_loss_pct,
    debug=True
):
    df = df.copy()  # Work on a copy of the dataframe to avoid side effects
    trades = []  # List to store all trade positions
    diagnostics = []  # Store debug/diagnostic information
    long_position = None  # Track open long positions
    short_position = None  # Track open short positions

    # Track various statistics from the trades
    trade_stats = {
        'TP': 0, 'SL': 0, 'TIMEOUT': 0,
        'pnl_total': 0, 'pnl_avg': 0,
        'returns': [], 'usd_returns': [],
        'mfe_total': 0, 'mue_total': 0,
        'mfe_avg': 0, 'mue_avg': 0,
    }

    last_trade_time_long = None  # Last timestamp a long was opened
    last_trade_time_short = None  # Last timestamp a short was opened
    trades_today = 0  # Count of trades today
    last_day = None  # Track current day
    daily_loss = 0  # Track PnL per day

    # Calculate rolling volatility and fill missing values using backward fill
    df['volatility'] = df['close'].rolling(window=volatility_window).std().bfill()
    # Exponential moving averages for trend detection
    df['ema_short'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=30, adjust=False).mean()

    for i in range(len(df)):
        current_price = df['close'].iloc[i]  # Current market price
        current_time = df.index[i]  # Current timestamp
        volatility = df['volatility'].iloc[i]  # Current volatility value

        # Reset daily counters when new day begins
        if last_day != current_time.date():
            trades_today = 0
            daily_loss = 0
            last_day = current_time.date()

        # Skip trade evaluation if volatility is too low
        if volatility < min_volatility_threshold:
            if debug:
                print(f"[SKIP] Low volatility: {volatility:.6f} < {min_volatility_threshold}")
            continue

        predicted_price = predictions[i] if predictions is not None and i < len(predictions) else None
        prev_price = df['close'].iloc[i - 1] if i > 0 else current_price
        prediction_trend = predicted_price - prev_price if predicted_price else 0

        if debug:
            print(f"[DEBUG] Predicted: {predicted_price:.2f} | Current: {current_price:.2f} | Trend: {prediction_trend:.4f}")

        pred_change = abs(prediction_trend / current_price)
        if pred_change < min_prediction_change:
            if debug:
                print(f"[SKIP] Prediction change too small: {pred_change:.6f} < {min_prediction_change}")
            continue

        expected_gain = abs(predicted_price - current_price) / current_price if predicted_price else 0
        required_gain = min_expected_gain + min(volatility * 0.2, 2.0)  # capped volatility multiplier
        if debug:
            print(f"[INFO] Volatility: {volatility:.4f} | Required Gain: {required_gain:.6f} | Expected Gain: {expected_gain:.6f}")
        if expected_gain < required_gain:
            if debug:
                print(f"[SKIP] Expected gain too low: {expected_gain:.6f} < {required_gain:.6f}")
            continue

        # Enforce trade limits per day and max allowable daily loss
        if trades_today >= max_trades_per_day or daily_loss <= -max_daily_loss:
            if debug:
                print("[SKIP] Daily trade limit or max loss hit")
            continue

        # Function to open a trade position (long or short)
        def open_position(pos_type):
            # Adjust TP/SL based on volatility
            dynamic_tp = take_profit_pct * (1 + 0.5 * volatility)
            dynamic_sl = stop_loss_pct * (1 + 0.5 * volatility)
            pos = {
                'timestamp': current_time,
                'entry_price': current_price,
                'take_profit': current_price * (
                    1 + dynamic_tp if pos_type == 'LONG'
                    else 1 - dynamic_tp
                ),
                'stop_loss': current_price * (
                    1 - dynamic_sl if pos_type == 'LONG'
                    else 1 + dynamic_sl
                ),
                'trailing_trigger': current_price * (
                    1 + dynamic_tp * trailing_trigger_factor if pos_type == 'LONG'
                    else 1 - dynamic_tp * trailing_trigger_factor
                ),
                'trailing_buffer': dynamic_tp * trailing_trigger_factor * 0.5,
                'active': True,
                'type': pos_type
            }

            if pos_type == 'LONG':
                nonlocal long_position
                long_position = pos
                nonlocal last_trade_time_long
                last_trade_time_long = current_time
            else:
                nonlocal short_position
                short_position = pos
                nonlocal last_trade_time_short
                last_trade_time_short = current_time

            trades.append(pos)
            nonlocal trades_today
            trades_today += 1

            if debug:
                print(f"[{pos_type} ENTRY] Time: {current_time}, Price: {current_price:.2f}")

        # Check conditions and open LONG or SHORT
        ema_short = df['ema_short'].iloc[i]
        ema_long = df['ema_long'].iloc[i]

        if (
            predicted_price > current_price and
            (not long_position or not long_position['active']) and
            (not last_trade_time_long or (current_time - last_trade_time_long) >= cooldown_period)
        ):
            open_position('LONG')

        if (
            predicted_price < current_price and
            (not short_position or not short_position['active']) and
            (not last_trade_time_short or (current_time - last_trade_time_short) >= cooldown_period)
        ):
            open_position('SHORT')

        # Evaluate open positions for possible exit
        def process_exit(position, is_long):
            entry_price = position['entry_price']
            tp = position['take_profit']
            sl = position['stop_loss']
            age = (current_time - position['timestamp']).total_seconds() / 60
            exited = False
            pnl = 0

            if (is_long and current_price >= tp) or (not is_long and current_price <= tp):
                pnl = (current_price - entry_price) / entry_price if is_long else (entry_price - current_price) / entry_price
                reason = 'TP'
                exited = True
            elif (is_long and current_price <= sl) or (not is_long and current_price >= sl):
                pnl = (current_price - entry_price) / entry_price if is_long else (entry_price - current_price) / entry_price
                reason = 'SL'
                exited = True
            elif age >= position_timeout_minutes:
                pnl = (current_price - entry_price) / entry_price if is_long else (entry_price - current_price) / entry_price
                reason = 'TIMEOUT'
                exited = True
            elif (is_long and current_price >= position['trailing_trigger']) or (
                not is_long and current_price <= position['trailing_trigger']):
                key = 'highest_price' if is_long else 'lowest_price'
                comp = max if is_long else min

                if key not in position:
                    position[key] = current_price
                    if debug:
                        print(f"[TRAIL START - {position['type']}] Triggered at {current_time}, Price: {current_price:.2f}")
                else:
                    position[key] = comp(position[key], current_price)
                    if debug:
                        print(f"[TRAIL UPDATE - {position['type']}] New {'High' if is_long else 'Low'}: {position[key]:.2f}")

                if (is_long and current_price <= position[key] - position['trailing_buffer']) or (
                    not is_long and current_price >= position[key] + position['trailing_buffer']):
                    pnl = (current_price - entry_price) / entry_price if is_long else (entry_price - current_price) / entry_price
                    reason = 'TRAILING'
                    exited = True

            if 'mfe' not in position:
                position['mfe'] = 0
                position['mue'] = 0

            # Track maximum favorable and unfavorable excursion
            price_diff = current_price - entry_price if is_long else entry_price - current_price
            favorable = max(0, price_diff)
            unfavorable = max(0, -price_diff)
            position['mfe'] = max(position['mfe'], favorable)
            position['mue'] = max(position['mue'], unfavorable)

            if exited:
                # Calculate percentage excursions and PnL
                position['mfe_pct'] = position['mfe'] / entry_price
                position['mue_pct'] = position['mue'] / entry_price
                trade_stats['mfe_total'] += position['mfe_pct']
                trade_stats['mue_total'] += position['mue_pct']

                if debug:
                    print(f"[EXIT - {reason}] Time: {current_time}, Price: {current_price:.2f}, "
                          f"PnL: {pnl:.4f}, MFE: {position['mfe_pct']:.4%}, MUE: {position['mue_pct']:.4%}")

                # Finalize and log trade
                position.update({
                    'exit_price': current_price,
                    'exit_time': current_time,
                    'result': reason,
                    'active': False,
                    'pnl': pnl,
                })
                position['usd_pnl'] = pnl * quantity * entry_price
                trade_stats[reason] += 1
                trade_stats['pnl_total'] += pnl
                trade_stats['returns'].append(pnl)
                trade_stats['usd_returns'].append(pnl * quantity * entry_price)
                total_trades = trade_stats['TP'] + trade_stats['SL'] + trade_stats['TIMEOUT']
                trade_stats['pnl_avg'] = trade_stats['pnl_total'] / total_trades
                trade_stats['mfe_avg'] = trade_stats['mfe_total'] / total_trades
                trade_stats['mue_avg'] = trade_stats['mue_total'] / total_trades

                nonlocal daily_loss
                daily_loss += pnl

        # Check for exit signals on active positions
        if long_position and long_position['active']:
            process_exit(long_position, is_long=True)

        if short_position and short_position['active']:
            process_exit(short_position, is_long=False)

        # Record diagnostic information per iteration
        diagnostics.append({
            'timestamp': current_time,
            'price': current_price,
            'predicted': predicted_price,
            'prediction_trend': prediction_trend,
            'volatility': volatility,
        })

        if debug:
            print(f"[SUMMARY] TP: {trade_stats['TP']} | SL: {trade_stats['SL']} | TIMEOUT: {trade_stats['TIMEOUT']} | Avg PnL: {trade_stats['pnl_avg']:.4f}")

    # Compute overall metrics like Sharpe ratio and max drawdown
    returns = np.array(trade_stats['returns'])
    sharpe = np.mean(returns) / np.std(returns) if len(returns) > 1 else 0.0
    equity_curve = np.cumsum(trade_stats['usd_returns'])
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max)
    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0.0

    trade_stats['sharpe_ratio'] = sharpe
    trade_stats['max_drawdown'] = max_dd

    return trades, diagnostics, trade_stats