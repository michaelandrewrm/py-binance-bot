import time
import logging
import requests
from decimal import Decimal
from logger import Colors

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
        logger: logging.Logger = None,
        get_ai_grid_parameters=None
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

        # Fetch symbol filters for price and qty rounding
        self.filters = self.get_symbol_filters()
        self.price_tick = float(self.filters['PRICE_FILTER']['tickSize'])
        self.qty_step = float(self.filters['LOT_SIZE']['stepSize'])
        self.grid_prices = self.calc_grid_prices(self.lower_price, self.upper_price, self.num_levels)

        self.last_trade_ids = set()
        self.realized_pnl = 0.0
        self.position = 0.0
        self.avg_entry = 0.0
        self.trade_history = []

    def get_with_retry(self, func, *args, **kwargs):
        max_retries, delay = 5, 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.logger.warning(f"API error ({e}), retrying in {delay}s (attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
        raise Exception(f"Failed after {max_retries} retries")


    def get_symbol_filters(self):
        info = self.client.get_symbol_info(self.symbol)
        return {f['filterType']: f for f in info['filters']}

    def get_symbol_assets(self):
        info = self.client.get_symbol_info(self.symbol)
        return info['baseAsset'], info['quoteAsset']

    def round_step(self, value, step):
        d = Decimal(str(value))
        s = Decimal(str(step))
        return float((d // s) * s)

    def get_current_price(self):
        ticker = self.client.get_symbol_ticker(symbol=self.symbol)
        return float(ticker['price'])

    def calc_grid_prices(self, lower: float, upper: float, levels: int):
        lower_dec = Decimal(str(lower))
        upper_dec = Decimal(str(upper))
        levels_dec = Decimal(str(levels - 1))
        step = (upper_dec - lower_dec) / levels_dec
        return [self.round_step((lower_dec + i * step), Decimal(str(self.price_tick))) for i in range(levels)]
    
    def get_available_balance(self, asset: str):
        balances = self.client.get_account()['balances']
        for bal in balances:
            if bal['asset'] == asset:
                return float(bal['free'])
        return 0.0

    def place_order(self, price: float, side: str):
        try:
            price = self.round_step(price, self.price_tick)
            qty = self.round_step(self.quantity, self.qty_step)
            base_asset, quote_asset = self.get_symbol_assets()
            color = Colors.RESET

            if side.upper() == 'SELL':
                balance = self.get_available_balance(base_asset)
                color = Colors.RED
                if balance < qty:
                    self.logger.warning(f"Not enough {base_asset} to place SELL order: have {balance}, need {qty}")
                    return None
            elif side.upper() == 'BUY':
                required = price * qty
                balance = self.get_available_balance(quote_asset)
                color = Colors.GREEN
                if balance < required:
                    self.logger.warning(f"Not enough {quote_asset} to place BUY order: have {balance}, need {required}")
                    return None
                
            order = self.client.create_order(
                symbol=self.symbol,
                side=side,
                type='LIMIT',
                price=f"{price:.8f}",
                quantity=qty,
                timeInForce='GTC'
            )

            self.logger.info(f"Placed {color}{side}{Colors.RESET} order at {price} qty {qty}")
            return order['orderId']
        except Exception as e:
            self.logger.error(f"Failed to place {side} order at {price}: {e}")
            return None

    def cancel_order(self, order_id):
        try:
            self.client.cancel_order(symbol=self.symbol, orderId=order_id)
            self.logger.info(f"Cancelled order {order_id}")
        except Exception as e:
            self.logger.warning(f"Failed to cancel order {order_id}: {e}")

    def sync_orders(self):
        open_orders = self.client.get_open_orders(symbol=self.symbol)
        open_by_price = {self.round_step(float(order['price']), self.price_tick): order for order in open_orders}
        current_price = self.get_current_price()

        for price in self.grid_prices:
            if price not in open_by_price:
                side = 'BUY' if price < current_price else 'SELL'
                order_id = self.place_order(price, side)
                if order_id:
                    self.grid_orders[price] = order_id
            else:
                self.grid_orders[price] = open_by_price[price]['orderId']

        prices_to_remove = [p for p in self.grid_orders if p not in open_by_price and p not in self.grid_prices]
        for p in prices_to_remove:
            del self.grid_orders[p]

    def update_grid_from_ai(self):
        if self.get_ai_grid_parameters:
            ai_params = self.get_ai_grid_parameters(self.symbol)
            try:
                lower = float(ai_params['lower_price'])
                upper = float(ai_params['upper_price'])
                levels = int(ai_params['num_levels'])
                if lower >= upper:
                    raise ValueError("Lower price must be less than upper price.")
                if levels < 2:
                    raise ValueError("Number of levels must be at least 2.")
            except (KeyError, ValueError, TypeError) as e:
                self.logger.error(f"Invalid AI grid parameters: {ai_params} ({e})")
                return
            self.logger.info(f"AI grid: lower={lower}, upper={upper}, levels={levels}")
            for oid in self.grid_orders.values():
                self.cancel_order(oid)
            self.grid_prices = self.calc_grid_prices(lower, upper, levels)
            self.grid_orders = {}
            self.lower_price = lower
            self.upper_price = upper
            self.num_levels = levels
            self.sync_orders()

    def log_balances(self):
        base_asset, quote_asset = self.get_symbol_assets()
        base_balance = self.get_available_balance(base_asset)
        quote_balance = self.get_available_balance(quote_asset)
        self.logger.info(f"Balances: {base_asset}: {base_balance}, {quote_asset}: {quote_balance}")

    def log_open_orders(self):
        open_orders = self.client.get_open_orders(symbol=self.symbol)
        if not open_orders:
            self.logger.info("No open orders.")
            return
        msg = "\nOpen Orders:\n"
        msg += "{:<8} {:<12} {:<12} {:<10} {:<10}\n".format("SIDE", "PRICE", "QTY", "ORDER_ID", "STATUS")
        for o in open_orders:
            msg += "{:<8} {:<12} {:<12} {:<10} {:<10}\n".format(
                o['side'],
                o['price'],
                o['origQty'],
                o['orderId'],
                o['status']
            )
        self.logger.info(msg)

    def check_and_log_trades(self):
        fills = self.client.get_my_trades(symbol=self.symbol)
        new_fills = [f for f in fills if f['id'] not in self.last_trade_ids]

        base_asset, quote_asset = self.get_symbol_assets()

        for f in new_fills:
            side = 'BUY' if f['isBuyer'] else 'SELL'
            price = float(f['price'])
            qty = float(f['qty'])
            commission = float(f['commission'])
            commission_asset = f['commissionAsset']
            time_fill = f['time']

            # Handle fee-adjusted PnL
            fee_in_quote = 0.0
            fee_in_base = 0.0

            if commission_asset == base_asset:
                fee_in_base = commission
            elif commission_asset == quote_asset:
                fee_in_quote = commission
            # (You can add handling for BNB as a commission asset, if desired.)

            # PnL logic
            if side == 'BUY':
                prev_position = self.position
                prev_avg_entry = self.avg_entry
                self.position += qty - fee_in_base  # fee reduces net received
                if (prev_position + qty - fee_in_base) != 0:
                    self.avg_entry = ((prev_avg_entry * prev_position) + price * (qty - fee_in_base)) / (prev_position + qty - fee_in_base)
                else:
                    self.avg_entry = price
                realized_pnl = 0
                color = Colors.GREEN
            else:
                # On sell, fee is subtracted from proceeds (if quote asset) or reduces amount sold (if base)
                net_qty = qty
                realized_pnl = (price - self.avg_entry) * net_qty
                realized_pnl -= fee_in_quote  # If fee was in quote, subtract from profit
                self.position -= qty  # (fee in base was already removed from buys)
                self.realized_pnl += realized_pnl
                color = Colors.RED

            self.trade_history.append({
                "time": time_fill,
                "side": side,
                "price": price,
                "qty": qty,
                "commission": commission,
                "commission_asset": commission_asset,
                "pnl": realized_pnl if side == "SELL" else 0
            })
            self.last_trade_ids.add(f['id'])
            self.logger.info(
                f"Trade: {color}{side}{Colors.RESET} {qty} @ {price} | Fee: {commission} {commission_asset} | "
                f"Realized PnL: {realized_pnl:.6f} | "
                f"Net PnL: {self.realized_pnl:.6f} | Position: {self.position:.6f}"
            )

    def run(self):
        self.logger.info("Placing initial grid orders...")
        self.sync_orders()
        self.logger.info(f"Initial grid: {self.grid_prices}")
        last_ai_update = time.time()

        try:
            while True:
                try:
                    if self.get_ai_grid_parameters and (time.time() - last_ai_update > self.ai_update_interval):
                        self.logger.info("Updating grid with AI/model...")
                        self.update_grid_from_ai()
                        last_ai_update = time.time()

                    self.sync_orders()
                    self.log_balances()
                    self.log_open_orders()
                    self.check_and_log_trades()
                    self.logger.info(f"Total Realized PnL: {self.realized_pnl:.6f}")   
                    time.sleep(self.poll_interval)
                except Exception as e:
                    self.logger.warning(f"Network or API error: {e}. Sleeping and will retry...")
                    time.sleep(10)
        except KeyboardInterrupt:
            self.logger.info("Stopping and cancelling all open grid orders...")
            for oid in self.grid_orders.values():
                self.cancel_order(oid)
            self.logger.info("Trader stopped.")