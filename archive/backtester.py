import math
import logging
import heapq
import numpy as np
import pandas as pd


def simulate_grid_trades(
    price_series: pd.Series,
    lower_price: float,
    upper_price: float,
    num_levels: int = 10,
    quantity: float = 0.001,
    fee: float = 0.001,
    allow_short: bool = False,
    logger: logging.Logger = None,
):
    """Back-test a classic staggered-grid strategy.

    Parameters
    ----------
    price_series : pd.Series - Close prices indexed by timestamp.
    lower_price : float - Lowest grid level.
    upper_price : float - Highest grid level.
    num_levels : int, default 10 - Total number of *buy* (and mirrored sell) levels.
    quantity : float, default 0.001 - Base size per limit order.
    fee : float, default 0.001 (0.1%) - Exchange taker/maker fee fraction.
    allow_short : bool, default False - If True, also place the symmetric short grid (sell-high → buy-back lower).
    logger : logging.Logger, optional
        Logger instance to use. If None (default), a new logger is created inside the function.
        If a logger is provided, its configuration (such as log level) will NOT be changed inside the function.
        This avoids mutable default arguments and ensures logger configuration is explicit.

    Returns
    -------
    equity_curve : pd.Series - Account value over time.
    trades : list[dict] - Executed trades, each as a dict with:
        time (index type): timestamp of trade,
        side (str): "buy" or "sell",
        price (float): execution price,
        qty (float): quantity traded,
        pnl (float or None): profit and loss for the trade (None for buys).
    """

    if lower_price > upper_price or math.isclose(
        lower_price, upper_price, abs_tol=1e-9
    ):
        raise ValueError("`lower_price` must be less than `upper_price`.")
    if num_levels <= 1:
        raise ValueError("`num_levels` must be greater than 1.")
    if quantity <= 0:
        raise ValueError("`quantity` must be positive.")
    if not (0 <= fee < 1):
        raise ValueError("`fee` must be between 0 (inclusive) and 1 (exclusive).")
    if price_series.empty:
        raise ValueError("`price_series` must not be empty.")
    if not price_series.index.is_monotonic_increasing:
        raise ValueError("`price_series` index must be sorted in increasing order.")

    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    def build_grid(lower_price, upper_price, num_levels):
        levels = np.linspace(lower_price, upper_price, num_levels, endpoint=False)
        grid_spacing = levels[1] - levels[0]
        return levels, grid_spacing

    levels, grid_spacing = build_grid(lower_price, upper_price, num_levels)
    cash = 0.0
    position = 0.0  # + long, − short
    active_sells = []  # min-heap of (trigger_price, qty, buy_price)
    trades = []
    equity_curve = []
    active_sell_levels = set()

    def execute_buy(ts, lvl):
        nonlocal cash, position
        cash -= quantity * lvl * (1 + fee)
        position += quantity
        trades.append(
            {
                "time": ts,
                "side": "buy",
                "price": lvl,
                "qty": quantity,
                "pnl": None,
            }
        )
        heapq.heappush(active_sells, (lvl + grid_spacing, quantity, lvl))
        active_sell_levels.add(lvl + grid_spacing)
        logger.info(f"{ts} BUY  {quantity:.6f} @ {lvl:.2f}")

    def execute_sell(ts, trig, qty, buy_price):
        nonlocal cash, position
        revenue = qty * trig * (1 - fee)
        cash += revenue
        position -= qty
        pnl = revenue - qty * buy_price
        trades.append(
            {"time": ts, "side": "sell", "price": trig, "qty": qty, "pnl": pnl}
        )
        active_sell_levels.discard(trig)
        logger.info(f"{ts} SELL {qty:.6f} @ {trig:.2f}  PnL {pnl:.2f}")

    # Precompute grid levels and their states before the loop
    grid_levels = list(levels)
    grid_level_states = {lvl: False for lvl in grid_levels}  # False = not bought yet

    for ts, price in price_series.items():
        idx = np.searchsorted(grid_levels, price, side="right") - 1
        while idx >= 0:
            lvl = grid_levels[idx]
            if not grid_level_states[lvl] and (
                price <= lvl or math.isclose(price, lvl, abs_tol=1e-9)
            ):
                execute_buy(ts, lvl)
                grid_level_states[lvl] = True
                idx -= 1
            else:
                break

        # Always check queued sells, even if a buy occurred
        while active_sells and (
            price >= active_sells[0][0]
            or math.isclose(price, active_sells[0][0], abs_tol=1e-9)
        ):
            trig, qty, buy_price = heapq.heappop(active_sells)
            execute_sell(ts, trig, qty, buy_price)
        equity_curve.append(cash + position * price)

    equity_curve = pd.Series(equity_curve, index=price_series.index, name="equity")
    return equity_curve, trades
