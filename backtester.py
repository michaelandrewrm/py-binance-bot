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
    cooldown_period,
    max_trades_per_day,
    position_timeout_minutes,
    max_daily_loss,
    trailing_trigger_factor,
    debug=True
):
    df = df.copy()
    trades = []
    diagnostics = []
    long_position = None
    trade_stats = {'TP': 0, 'SL': 0, 'TIMEOUT': 0, 'pnl_total': 0, 'pnl_avg': 0}
    last_trade_time = None
    short_position = None
    take_profit_pct = 0.003  # 0.3% profit target
    stop_loss_pct = 0.002    # 0.2% risk

    df.loc[:, 'volatility'] = df['close'].rolling(window=volatility_window).std().bfill()

    trades_today = 0
    last_day = None
    daily_loss = 0

    for i in range(0, len(df)):
        current_price = df['close'].iloc[i]
        current_time = df.index[i]
        volatility = df['volatility'].iloc[i]

        if last_day != current_time.date():
            trades_today = 0
            daily_loss = 0
            last_day = current_time.date()

        if volatility < min_volatility_threshold:
            continue

        predicted_price = None
        prediction_trend = 0

        if predictions is not None and i < len(predictions):
            predicted_price = predictions[i]
            prev_price = df['close'].iloc[i - 1] if i > 0 else df['close'].iloc[i]
            prediction_trend = predicted_price - prev_price

        if abs(prediction_trend / current_price) < min_prediction_change:
            continue

        expected_gain = abs(predicted_price - current_price) / current_price if predicted_price else 0

        if expected_gain < min_expected_gain:
            continue

        if trades_today >= max_trades_per_day or daily_loss <= -max_daily_loss:
            continue

        if (
            predicted_price > current_price and
            (last_trade_time is None or (current_time - last_trade_time) >= cooldown_period)
        ):
            long_position = {
                'timestamp': current_time,
                'entry_price': current_price,
                'take_profit': current_price * (1 + take_profit_pct),
                'stop_loss': current_price * (1 - stop_loss_pct),
                'trailing_trigger': current_price * (1 + take_profit_pct * trailing_trigger_factor),
                'trailing_buffer': take_profit_pct * trailing_trigger_factor * 0.5,
                'active': True,
                'type': 'LONG'
            }
            trades.append(long_position)
            last_trade_time = current_time
            trades_today += 1

            if debug:
                print(f"[LONG ENTRY] Time: {current_time}, Price: {current_price:.2f}")

        if (
            predicted_price < current_price and
            (last_trade_time is None or (current_time - last_trade_time) >= cooldown_period)
        ):
            short_position = {
                'timestamp': current_time,
                'entry_price': current_price,
                'take_profit': current_price * (1 - take_profit_pct),
                'stop_loss': current_price * (1 + stop_loss_pct),
                'trailing_trigger': current_price * (1 - take_profit_pct * trailing_trigger_factor),
                'trailing_buffer': take_profit_pct * trailing_trigger_factor * 0.5,
                'active': True,
                'type': 'SHORT'
            }
            trades.append(short_position)
            last_trade_time = current_time
            trades_today += 1

            if debug:
                print(f"[SHORT ENTRY] Time: {current_time}, Price: {current_price:.2f}")

        if long_position and long_position['active']:
            entry_price = long_position['entry_price']
            tp = long_position['take_profit']
            sl = long_position['stop_loss']
            age = (current_time - long_position['timestamp']).total_seconds() / 60

            if current_price >= tp:
                long_position.update({
                    'exit_price': tp,
                    'exit_time': current_time,
                    'result': 'TP',
                    'active': False,
                    'pnl': (tp - entry_price) / entry_price,
                })
                long_position['usd_pnl'] = long_position['pnl'] * quantity * entry_price
                trade_stats['TP'] += 1
                
                if debug:
                    print(f"[EXIT - TP] Time: {current_time}, Price: {tp:.2f}, PnL: {long_position['pnl']:.4f}")


            elif current_price <= sl:
                long_position.update({
                    'exit_price': sl,
                    'exit_time': current_time,
                    'result': 'SL',
                    'active': False,
                    'pnl': (sl - entry_price) / entry_price,
                })
                long_position['usd_pnl'] = long_position['pnl'] * quantity * entry_price
                trade_stats['SL'] += 1

                if debug:
                    print(f"[EXIT - SL] Time: {current_time}, Price: {sl:.2f}, PnL: {long_position['pnl']:.4f}")

            elif age >= position_timeout_minutes:
                long_position.update({
                    'exit_price': current_price,
                    'exit_time': current_time,
                    'result': 'TIMEOUT',
                    'active': False,
                    'pnl': (current_price - entry_price) / entry_price,
                })
                long_position['usd_pnl'] = long_position['pnl'] * quantity * entry_price
                trade_stats['TIMEOUT'] += 1

                if debug:
                    print(f"[EXIT - TIMEOUT] Time: {current_time}, Price: {current_price:.2f}, PnL: {long_position['pnl']:.4f}")
            
            elif current_price >= long_position['trailing_trigger']:
                if 'highest_price' not in long_position:
                    long_position['highest_price'] = current_price

                    if debug:
                        print(f"[TRAIL START - LONG] Triggered at {current_time}, Price: {current_price:.2f}")
                else:
                    long_position['highest_price'] = max(long_position['highest_price'], current_price)

                    if debug:
                        print(f"[TRAIL UPDATE - LONG] New High: {long_position['highest_price']:.2f}")

                if current_price <= long_position['highest_price'] - long_position['trailing_buffer']:
                    long_position.update({
                        'exit_price': current_price,
                        'exit_time': current_time,
                        'result': 'TRAILING',
                        'active': False,
                        'pnl': (current_price - entry_price) / entry_price,
                    })
                    long_position['usd_pnl'] = long_position['pnl'] * quantity * entry_price
                    trade_stats['TP'] += 1

                    if debug:
                        print(f"[EXIT - TRAILING] Time: {current_time}, Price: {current_price:.2f}, PnL: {long_position['pnl']:.4f}")


            if 'pnl' in long_position:
                trade_stats['pnl_total'] += long_position['pnl']
                daily_loss += long_position['pnl']
                trade_stats['pnl_avg'] = trade_stats['pnl_total'] / (
                    trade_stats['TP'] + trade_stats['SL'] + trade_stats['TIMEOUT']
                )

        if short_position and short_position['active']:
            entry_price = short_position['entry_price']
            tp = short_position['take_profit']
            sl = short_position['stop_loss']
            age = (current_time - short_position['timestamp']).total_seconds() / 60

            if current_price <= tp:
                short_position.update({
                    'exit_price': tp,
                    'exit_time': current_time,
                    'result': 'TP',
                    'active': False,
                    'pnl': (entry_price - tp) / entry_price,
                })
                short_position['usd_pnl'] = short_position['pnl'] * quantity * entry_price
                trade_stats['TP'] += 1

                if debug:
                    print(f"[EXIT - TP] Time: {current_time}, Price: {tp:.2f}, PnL: {short_position['pnl']:.4f}")

            elif current_price >= sl:
                short_position.update({
                    'exit_price': sl,
                    'exit_time': current_time,
                    'result': 'SL',
                    'active': False,
                    'pnl': (entry_price - sl) / entry_price,
                })
                short_position['usd_pnl'] = short_position['pnl'] * quantity * entry_price
                trade_stats['SL'] += 1

                if debug:
                    print(f"[EXIT - SL] Time: {current_time}, Price: {sl:.2f}, PnL: {short_position['pnl']:.4f}")

            elif age >= position_timeout_minutes:
                short_position.update({
                    'exit_price': current_price,
                    'exit_time': current_time,
                    'result': 'TIMEOUT',
                    'active': False,
                    'pnl': (entry_price - current_price) / entry_price,
                })
                short_position['usd_pnl'] = short_position['pnl'] * quantity * entry_price
                trade_stats['TIMEOUT'] += 1

                if debug:
                    print(f"[EXIT - TIMEOUT] Time: {current_time}, Price: {current_price:.2f}, PnL: {short_position['pnl']:.4f}")
            
            elif current_price <= short_position['trailing_trigger']:
                if 'lowest_price' not in short_position:
                    short_position['lowest_price'] = current_price

                    if debug:
                        print(f"[TRAIL START - SHORT] Triggered at {current_time}, Price: {current_price:.2f}")
                else:
                    short_position['lowest_price'] = min(short_position['lowest_price'], current_price)

                    if debug:
                        print(f"[TRAIL UPDATE - SHORT] New Low: {short_position['lowest_price']:.2f}")

                if current_price >= short_position['lowest_price'] + short_position['trailing_buffer']:
                    short_position.update({
                        'exit_price': current_price,
                        'exit_time': current_time,
                        'result': 'TRAILING',
                        'active': False,
                        'pnl': (entry_price - current_price) / entry_price,
                    })
                    short_position['usd_pnl'] = short_position['pnl'] * quantity * entry_price
                    trade_stats['TP'] += 1

                    if debug:
                        print(f"[EXIT - TRAILING] Time: {current_time}, Price: {current_price:.2f}, PnL: {short_position['pnl']:.4f}")


            if 'pnl' in short_position:
                trade_stats['pnl_total'] += short_position['pnl']
                daily_loss += short_position['pnl']
                trade_stats['pnl_avg'] = trade_stats['pnl_total'] / (
                    trade_stats['TP'] + trade_stats['SL'] + trade_stats['TIMEOUT']
                )

        diagnostics.append({
            'timestamp': current_time,
            'price': current_price,
            'predicted': predicted_price,
            'prediction_trend': prediction_trend,
            'volatility': volatility,
        })

        if debug:
            print(f"[SUMMARY] TP: {trade_stats['TP']} | SL: {trade_stats['SL']} | TIMEOUT: {trade_stats['TIMEOUT']} | Avg PnL: {trade_stats['pnl_avg']:.4f}")

    return trades, diagnostics, trade_stats
