def get_symbol_info(client, symbol) -> dict[str, float]:
    info = client.get_symbol_info(symbol)
    if not info or "filters" not in info:
        raise ValueError(f"Could not fetch symbol info for {symbol}")
    tick_size = float([f for f in info['filters'] if f['filterType'] == 'PRICE_FILTER'][0]['tickSize'])
    step_size = float([f for f in info['filters'] if f['filterType'] == 'LOT_SIZE'][0]['stepSize'])
    min_qty = float([f for f in info['filters'] if f['filterType'] == 'LOT_SIZE'][0]['minQty'])
    min_notional = float([f for f in info['filters'] if f['filterType'] == 'NOTIONAL'][0]['minNotional'])

    return dict(
        base_asset=info['baseAsset'],
        quote_asset=info['quoteAsset'],
        tick_size=tick_size,
        step_size=step_size,
        min_qty=min_qty,
        min_notional=min_notional,
    )

def get_available_balance(client, asset: str) -> float:
    balances = client.get_account().get('balances', [])
    for bal in balances:
        if bal['asset'] == asset:
            return float(bal.get('free', 0))
    return 0.0

def get_current_price(client, symbol) -> float:
    ticker = client.get_symbol_ticker(symbol=symbol)
    return float(ticker['price'])