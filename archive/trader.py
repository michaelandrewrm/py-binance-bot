import time
import logging
from decimal import Decimal
from logger import Colors
from archived.binance_client import (
    get_symbol_info,
    get_available_balance,
    get_current_price,
)


class Trader:
    def __init__(
        self,
        client,
        symbol: str,
        lower_price: float,
        upper_price: float,
        num_levels: int,
        quantity: float,
        poll_interval: int = 10,
        ai_update_interval: int = 3600,
        get_ai_grid_parameters=None,
        logger: logging.Logger = None,
        trailing_up: bool = False,
        grid_trigger_price: float = None,
        stop_loss_price: float = None,
        take_profit_price: float = None,
        sell_all_on_stop: bool = False,
        investment_amount: float = None,
    ):
        self.client = client
        self.symbol = symbol
        self.lower_price = lower_price
        self.upper_price = upper_price
        self.num_levels = num_levels
        self.quantity = quantity
        self.poll_interval = poll_interval
        self.ai_update_interval = ai_update_interval
        self.get_ai_grid_parameters = get_ai_grid_parameters
        self.logger = logger
        self.grid_orders = {}

        self.trailing_up = trailing_up
        self.grid_trigger_price = grid_trigger_price
        self.stop_loss_price = stop_loss_price
        self.take_profit_price = take_profit_price
        self.sell_all_on_stop = sell_all_on_stop
        self.investment_amount = investment_amount

        self.grid_active = self.grid_trigger_price is None
        self.last_trade_ids = set()
        self.realized_pnl = 0.0
        self.position = 0.0
        self.avg_entry = 0.0
        self.trade_history = []

        self.filters = get_symbol_info(client=self.client, symbol=self.symbol)
        self.price_tick = self.filters.get("tick_size")
        self.qty_step = self.filters.get("step_size")
        self.init_position_and_entry()

        try:
            self.initialize_grid()
            self.adjust_grid_for_balance()
        except ValueError as e:
            self.logger.error(f"Failed to update grid: {e}")

    def round_step(self, value, step):
        d = Decimal(str(value))
        s = Decimal(str(step))
        # Return rounded DOWN to step (not up!)
        return float((d // s) * s)

    def adjust_grid_for_balance(self):
        base_asset, quote_asset = get_symbol_info(
            client=self.client, symbol=self.symbol
        )
        base_balance = get_available_balance(client=self.client, asset=base_asset)
        quote_balance = get_available_balance(client=self.client, asset=quote_asset)
        current_price = get_current_price(client=self.client, symbol=self.symbol)
        prices = self.grid_prices
        quantity = self.quantity

        # How many sell levels can we fill?
        sell_prices = [p for p in prices if p > current_price]
        max_sells = int(base_balance // quantity) if quantity > 0 else 0

        # How many buy levels can we fill?
        buy_prices = [p for p in prices if p < current_price]
        affordable_buys = [p for p in buy_prices if (quantity * p) <= quote_balance]
        max_buys = len(affordable_buys)

        safe_levels = max(
            2,
            min(
                self.num_levels,
                max_buys + len(sell_prices) + 1,
                max_sells + len(buy_prices) + 1,
            ),
        )
        if self.investment_amount is not None:
            # Further limit max buys based on investment_amount
            quote_to_use = min(self.investment_amount, quote_balance)
            total_needed = 0
            count = 0
            for p in buy_prices:
                if total_needed + (p * quantity) > quote_to_use:
                    break
                total_needed += p * quantity
                count += 1
            max_buys = count
            safe_levels = max(2, min(safe_levels, max_buys + len(sell_prices) + 1))

        if safe_levels < self.num_levels:
            self.logger.warning(
                f"Not enough balance for full grid ({self.num_levels} levels). "
                f"Reducing to {safe_levels} levels. Base: {base_balance}, Quote: {quote_balance}"
            )
            self.num_levels = safe_levels
            self.grid_prices = self.calc_grid_prices(
                self.lower_price, self.upper_price, self.num_levels
            )
        else:
            self.logger.info(
                f"Sufficient balance for full grid ({self.num_levels} levels). "
                f"Base: {base_balance}, Quote: {quote_balance}"
            )

    def init_position_and_entry(self):
        base_asset, _ = get_symbol_info(client=self.client, symbol=self.symbol)
        balance = get_available_balance(client=self.client, asset=base_asset)
        current_price = get_current_price(client=self.client, symbol=self.symbol)
        self.position = balance
        self.avg_entry = current_price if balance > 0 else 0.0
        self.logger.info(
            f"Initial spot position: {balance} {base_asset} at {self.avg_entry}"
        )

    def initialize_grid(self, force=False):
        """
        Ensure grid brackets current price and is safe to trade.
        If force=True, will recenter grid if invalid.
        If return_suggestion=True, will return (suggested_lower, suggested_upper) if input is invalid.
        """
        current_price = get_current_price(client=self.client, symbol=self.symbol)
        min_grid_range = 0.005  # 0.5% min range for safety
        max_grid_range = 0.20  # 20% max range
        fee_rate = 0.001  # 0.1% per side
        min_profit_rate = fee_rate * 2 * 3  # Require grid step >= 0.6%
        min_grid_levels = max(2, min(self.num_levels, 50))

        min_step = current_price * min_profit_rate
        min_total_width = min_step * (min_grid_levels - 1)
        suggested_range = max(
            current_price * 0.01,
            min_grid_range * current_price * min_grid_levels,
            min_total_width,
        )

        min_allowed = current_price * (1 - max_grid_range)
        max_allowed = current_price * (1 + max_grid_range)

        self.lower_price, self.upper_price = sorted(
            [self.lower_price, self.upper_price]
        )
        self.lower_price = max(min_allowed, min(max_allowed, self.lower_price))
        self.upper_price = max(
            self.lower_price + suggested_range, min(max_allowed, self.upper_price)
        )

        # Check grid brackets price and is wide enough for profit
        actual_step = (self.upper_price - self.lower_price) / (min_grid_levels - 1)
        grid_ok = self.grid_brackets_price(current_price) and actual_step >= min_step

        if not grid_ok:
            msg = (
                f"Grid does NOT bracket price (lower={self.lower_price}, upper={self.upper_price}, "
                f"current={current_price}) OR step {actual_step:.6f} < min_step {min_step:.6f}"
            )
            # Suggest a grid centered on current price, wide enough for profit
            suggested_lower = round(current_price - suggested_range / 2, 6)
            suggested_upper = round(current_price + suggested_range / 2, 6)
            self.logger.error(
                msg
                + f" — SUGGESTED: lower={suggested_lower}, upper={suggested_upper} (centered, width={suggested_range:.6f}, min step={min_step:.6f})"
            )
            if force:
                self.logger.warning("Recentering grid automatically.")
                self.lower_price = suggested_lower
                self.upper_price = suggested_upper
            else:
                raise ValueError(
                    msg
                    + f" — SUGGESTED: lower={suggested_lower}, upper={suggested_upper}"
                )

        self.num_levels = min_grid_levels
        self.grid_prices = self.calc_grid_prices(
            self.lower_price, self.upper_price, self.num_levels
        )
        self.logger.info(
            f"Initialized grid: lower={self.lower_price}, upper={self.upper_price}, levels={self.num_levels}, "
            f"current={current_price}, step={(self.upper_price - self.lower_price)/(self.num_levels-1):.6f}"
        )

    def grid_brackets_price(self, current_price):
        return self.lower_price < current_price < self.upper_price

    def get_with_retry(self, func, *args, **kwargs):
        max_retries, delay = 5, 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.logger.warning(
                    f"API error ({e}), retrying in {delay}s (attempt {attempt+1}/{max_retries})"
                )
                time.sleep(delay)
        raise Exception(f"Failed after {max_retries} retries")

    def can_place_sell_order(self, qty):
        base_asset, _ = get_symbol_info(client=self.client, symbol=self.symbol)
        available = get_available_balance(client=self.client, asset=base_asset)
        return available >= qty

    def place_order(self, price: float, side: str):
        try:
            price = self.round_step(price, self.price_tick)
            qty = self.round_step(self.quantity, self.qty_step)
            base_asset, quote_asset = get_symbol_info(
                client=self.client, symbol=self.symbol
            )
            color = Colors.RESET

            if side.upper() == "SELL":
                balance = get_available_balance(client=self.client, asset=base_asset)
                color = Colors.RED
                if balance < qty:
                    self.logger.warning(
                        f"Not enough {base_asset} to place SELL order: have {balance}, need {qty}"
                    )
                    return None
                qty = min(balance, qty)  # Never sell more than owned

            elif side.upper() == "BUY":
                required = price * qty
                balance = get_available_balance(client=self.client, asset=quote_asset)
                color = Colors.GREEN
                if balance < required:
                    self.logger.warning(
                        f"Not enough {quote_asset} to place BUY order: have {balance}, need {required}"
                    )
                    return None

            order = self.client.create_order(
                symbol=self.symbol,
                side=side,
                type="LIMIT",
                price=f"{price:.8f}",
                quantity=qty,
                timeInForce="GTC",
            )
            self.logger.info(
                f"Placed {color}{side}{Colors.RESET} order at {price} qty {qty}"
            )
            return order["orderId"]
        except Exception as e:
            self.logger.error(f"Failed to place {side} order at {price}: {e}")
            return None

    def calc_grid_prices(self, lower: float, upper: float, levels: int):
        lower_dec = Decimal(str(lower))
        upper_dec = Decimal(str(upper))
        levels_dec = Decimal(str(levels - 1))
        if levels <= 1:
            return [float(lower_dec)]
        step = (upper_dec - lower_dec) / levels_dec
        return [
            self.round_step((lower_dec + i * step), Decimal(str(self.price_tick)))
            for i in range(levels)
        ]

    def update_grid_from_ai(self):
        if self.get_ai_grid_parameters:
            ai_params = self.get_ai_grid_parameters()
            try:
                lower = float(ai_params["lower_price"])
                upper = float(ai_params["upper_price"])
                levels = int(ai_params["num_levels"])

                self.lower_price = lower
                self.upper_price = upper
                self.num_levels = levels
                self.initialize_grid()

                for oid in self.grid_orders.values():
                    self.cancel_order(oid)

                self.grid_orders = {}
                self.sync_orders()
            except (KeyError, ValueError, TypeError) as e:
                self.logger.error(f"Invalid AI grid parameters: {ai_params} ({e})")
                return
            self.logger.info(f"AI grid: lower={lower}, upper={upper}, levels={levels}")
            self.grid_prices = self.calc_grid_prices(lower, upper, levels)

    def cancel_order(self, order_id):
        try:
            self.client.cancel_order(symbol=self.symbol, orderId=order_id)
            self.logger.info(f"Cancelled order {order_id}")
        except Exception as e:
            self.logger.warning(f"Failed to cancel order {order_id}: {e}")

    def sync_orders(self):
        open_orders = self.client.get_open_orders(symbol=self.symbol)
        open_by_price = {
            self.round_step(float(order["price"]), self.price_tick): order
            for order in open_orders
        }
        current_price = get_current_price(client=self.client, symbol=self.symbol)

        for price in self.grid_prices:
            if price not in open_by_price:
                side = "BUY" if price < current_price else "SELL"
                order_id = self.place_order(price, side)
                if order_id:
                    self.grid_orders[price] = order_id
            else:
                self.grid_orders[price] = open_by_price[price]["orderId"]

        prices_to_remove = [
            p
            for p in self.grid_orders
            if p not in open_by_price and p not in self.grid_prices
        ]
        for p in prices_to_remove:
            del self.grid_orders[p]

    def log_balances(self):
        base_asset, quote_asset = get_symbol_info(
            client=self.client, symbol=self.symbol
        )
        base_balance = get_available_balance(client=self.client, asset=base_asset)
        quote_balance = get_available_balance(client=self.client, asset=quote_asset)
        self.logger.info(
            f"Balances: {base_asset}: {base_balance}, {quote_asset}: {quote_balance}"
        )

    def log_open_orders(self):
        open_orders = self.client.get_open_orders(symbol=self.symbol)
        if not open_orders:
            self.logger.info("No open orders.")
            return
        msg = "\nOpen Orders:\n"
        msg += "{:<8} {:<12} {:<12} {:<10} {:<10}\n".format(
            "SIDE", "PRICE", "QTY", "ORDER_ID", "STATUS"
        )
        for o in open_orders:
            msg += "{:<8} {:<12} {:<12} {:<10} {:<10}\n".format(
                o["side"], o["price"], o["origQty"], o["orderId"], o["status"]
            )
        self.logger.info(msg)

    def update_position_and_pnl(
        self, side, price, qty, commission, commission_asset, base_asset, quote_asset
    ):
        fee_in_quote = 0.0
        fee_in_base = 0.0

        if commission_asset == base_asset:
            fee_in_base = commission
        elif commission_asset == quote_asset:
            fee_in_quote = commission

        if side == "BUY":
            prev_position = self.position
            prev_avg_entry = self.avg_entry
            self.position += qty - fee_in_base
            if (prev_position + qty - fee_in_base) != 0:
                self.avg_entry = (
                    (prev_avg_entry * prev_position) + price * (qty - fee_in_base)
                ) / (prev_position + qty - fee_in_base)
            else:
                self.avg_entry = price
            realized_pnl = 0
        else:
            net_qty = qty
            realized_pnl = (price - self.avg_entry) * net_qty
            realized_pnl -= fee_in_quote
            self.position -= qty
            self.realized_pnl += realized_pnl

        return realized_pnl

    def check_and_log_trades(self):
        fills = self.client.get_my_trades(symbol=self.symbol)
        new_fills = [f for f in fills if f["id"] not in self.last_trade_ids]

        base_asset, quote_asset = get_symbol_info(
            client=self.client, symbol=self.symbol
        )

        for f in new_fills:
            side = "BUY" if f["isBuyer"] else "SELL"
            color = Colors.GREEN if side == "BUY" else Colors.RED
            price = float(f["price"])
            qty = float(f["qty"])
            commission = float(f["commission"])
            commission_asset = f["commissionAsset"]
            time_fill = f["time"]
            realized_pnl = self.update_position_and_pnl(
                side, price, qty, commission, commission_asset, base_asset, quote_asset
            )

            self.trade_history.append(
                {
                    "time": time_fill,
                    "side": side,
                    "price": price,
                    "qty": qty,
                    "commission": commission,
                    "commission_asset": commission_asset,
                    "pnl": realized_pnl if side == "SELL" else 0,
                }
            )
            self.last_trade_ids.add(f["id"])
            self.logger.info(
                f"Trade: {color}{side}{Colors.RESET} {qty} @ {price} | Fee: {commission} {commission_asset} | "
                f"Realized PnL: {realized_pnl:.3f} | "
                f"Net PnL: {self.realized_pnl:.3f} | Position: {self.position:.3f}"
            )

    def run(self):
        if self.grid_active:
            self.logger.info("Placing initial grid orders...")
            self.sync_orders()
            self.logger.info(f"Initial grid: {self.grid_prices}")
        last_ai_update = time.time()

        try:
            while True:
                try:
                    current_price = get_current_price(
                        client=self.client, symbol=self.symbol
                    )
                    # 1. GRID TRIGGER
                    if self.grid_trigger_price is not None and not self.grid_active:
                        if current_price >= self.grid_trigger_price:
                            self.logger.info(
                                f"Grid trigger price hit: {current_price} >= {self.grid_trigger_price}. Activating grid."
                            )
                            self.grid_active = True
                            self.sync_orders()
                            self.logger.info(f"Grid activated at {current_price}.")
                        else:
                            self.logger.info(
                                f"Waiting for grid trigger. Current: {current_price}, Trigger: {self.grid_trigger_price}"
                            )
                            time.sleep(self.poll_interval)
                            continue  # Wait for trigger

                    # 2. STOP LOSS
                    if (
                        self.stop_loss_price is not None
                        and self.grid_active
                        and current_price <= self.stop_loss_price
                    ):
                        self.logger.warning(
                            f"Stop loss triggered at {current_price} <= {self.stop_loss_price}. Cancelling grid."
                        )
                        for oid in self.grid_orders.values():
                            self.cancel_order(oid)
                        self.grid_orders = {}
                        self.grid_active = False
                        if self.sell_all_on_stop:
                            base_asset, _ = get_symbol_info(
                                client=self.client, symbol=self.symbol
                            )
                            base_balance = get_available_balance(
                                client=self.client, asset=base_asset
                            )
                            if base_balance > 0:
                                try:
                                    order = self.client.create_order(
                                        symbol=self.symbol,
                                        side="SELL",
                                        type="MARKET",
                                        quantity=base_balance,
                                    )
                                    self.logger.info(
                                        f"Sold all {base_balance} {base_asset} at market on stop loss."
                                    )
                                except Exception as e:
                                    self.logger.error(
                                        f"Failed to sell all base at stop loss: {e}"
                                    )
                        time.sleep(self.poll_interval)
                        continue

                    # 3. TAKE PROFIT
                    if (
                        self.take_profit_price is not None
                        and self.grid_active
                        and current_price >= self.take_profit_price
                    ):
                        self.logger.info(
                            f"Take profit triggered at {current_price} >= {self.take_profit_price}. Cancelling grid."
                        )
                        for oid in self.grid_orders.values():
                            self.cancel_order(oid)
                        self.grid_orders = {}
                        self.grid_active = False
                        if self.sell_all_on_stop:
                            base_asset, _ = get_symbol_info(
                                client=self.client, symbol=self.symbol
                            )
                            base_balance = get_available_balance(
                                client=self.client, asset=base_asset
                            )
                            if base_balance > 0:
                                try:
                                    order = self.client.create_order(
                                        symbol=self.symbol,
                                        side="SELL",
                                        type="MARKET",
                                        quantity=base_balance,
                                    )
                                    self.logger.info(
                                        f"Sold all {base_balance} {base_asset} at market on take profit."
                                    )
                                except Exception as e:
                                    self.logger.error(
                                        f"Failed to sell all base at take profit: {e}"
                                    )
                        time.sleep(self.poll_interval)
                        continue

                    # 4. TRAILING UP
                    if (
                        self.trailing_up
                        and self.grid_active
                        and current_price > self.upper_price
                    ):
                        grid_width = self.upper_price - self.lower_price
                        self.logger.info(
                            f"Price broke above upper grid ({current_price} > {self.upper_price}), trailing up grid."
                        )
                        self.lower_price = current_price - grid_width
                        self.upper_price = current_price
                        self.initialize_grid(force=True)
                        self.adjust_grid_for_balance()
                        self.sync_orders()

                    # 5. AI Grid update (if enabled)
                    if (
                        self.get_ai_grid_parameters
                        and (time.time() - last_ai_update) > self.ai_update_interval
                    ):
                        self.logger.info("Updating grid with AI/model...")
                        self.update_grid_from_ai()
                        self.adjust_grid_for_balance()
                        self.sync_orders()
                        last_ai_update = time.time()

                    # 6. Main grid order logic
                    if self.grid_active:
                        self.sync_orders()
                        self.log_balances()
                        self.log_open_orders()
                        self.check_and_log_trades()
                        self.logger.info(f"Total Realized PnL: {self.realized_pnl:.3f}")
                    else:
                        self.logger.info("Grid inactive, not placing orders.")

                    time.sleep(self.poll_interval)
                except Exception as e:
                    self.logger.warning(
                        f"Network or API error: {e}. Sleeping and will retry..."
                    )
                    time.sleep(10)
        except KeyboardInterrupt:
            self.logger.info("Stopping and cancelling all open grid orders...")
            for oid in self.grid_orders.values():
                self.cancel_order(oid)
            self.logger.info("Trader stopped.")
