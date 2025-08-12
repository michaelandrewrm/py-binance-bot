"""
Paper Broker - In-memory orderbook-lite paper trading

This module provides a paper trading implementation that simulates
order execution without real money at risk.
"""

from typing import Dict, List, Optional, Tuple, Callable
from decimal import Decimal
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
import uuid

from core.state import Order, OrderSide, OrderType, OrderStatus, Position, Balance
from core.sizing import SizingCalculator, ExchangeInfo, FeeConfig, ensure_min_notional

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Current market data for simulation"""
    symbol: str
    bid: Decimal
    ask: Decimal
    last_price: Decimal
    volume: Decimal
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price"""
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread"""
        return self.ask - self.bid
    
    @property
    def spread_pct(self) -> Decimal:
        """Calculate spread as percentage"""
        return self.spread / self.mid_price * 100

@dataclass
class OrderBookLevel:
    """Single level in the order book"""
    price: Decimal
    quantity: Decimal
    
@dataclass
class OrderBook:
    """Simple order book representation"""
    symbol: str
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_best_bid(self) -> Optional[Decimal]:
        """Get best bid price"""
        return self.bids[0].price if self.bids else None
    
    def get_best_ask(self) -> Optional[Decimal]:
        """Get best ask price"""
        return self.asks[0].price if self.asks else None
    
    def get_mid_price(self) -> Optional[Decimal]:
        """Get mid price"""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid and ask:
            return (bid + ask) / 2
        return None

class SlippageModel:
    """Base class for slippage models"""
    
    def calculate_slippage(self, order: Order, market_data: MarketData) -> Decimal:
        """Calculate slippage for an order"""
        raise NotImplementedError

class LinearSlippageModel(SlippageModel):
    """Linear slippage model based on order size"""
    
    def __init__(self, base_slippage_bps: Decimal = Decimal('2'), 
                 size_impact_factor: Decimal = Decimal('0.1')):
        self.base_slippage_bps = base_slippage_bps
        self.size_impact_factor = size_impact_factor
    
    def calculate_slippage(self, order: Order, market_data: MarketData) -> Decimal:
        """Calculate linear slippage"""
        # Base slippage in basis points
        base_slippage = self.base_slippage_bps / 10000
        
        # Additional slippage based on order size relative to typical volume
        # This is a simplified model - in reality you'd use actual volume data
        size_factor = min(order.quantity * order.price / 10000, 1.0)  # Cap at 100% impact
        size_slippage = size_factor * self.size_impact_factor / 100
        
        total_slippage = base_slippage + size_slippage
        
        # Apply slippage in the direction that hurts the trader
        if order.side == OrderSide.BUY:
            return total_slippage  # Positive slippage increases price
        else:
            return -total_slippage  # Negative slippage decreases price

class PaperBroker:
    """
    Paper trading broker that simulates order execution
    """
    
    def __init__(self, initial_balances: Dict[str, Decimal], 
                 exchange_info: Dict[str, ExchangeInfo],
                 fee_config: FeeConfig,
                 slippage_model: SlippageModel = None):
        
        self.balances = {asset: Balance(asset, balance, Decimal('0')) 
                        for asset, balance in initial_balances.items()}
        self.exchange_info = exchange_info
        self.fee_config = fee_config
        self.slippage_model = slippage_model or LinearSlippageModel()
        
        # Trading state
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.positions: Dict[str, Position] = {}
        self.trades: List[Dict] = []
        
        # Market data
        self.market_data: Dict[str, MarketData] = {}
        self.order_books: Dict[str, OrderBook] = {}
        
        # Callbacks
        self.fill_callbacks: List[Callable] = []
        self.order_callbacks: List[Callable] = []
        
        # Sizing calculators
        self.sizing_calculators = {
            symbol: SizingCalculator(info, fee_config)
            for symbol, info in exchange_info.items()
        }
    
    def register_fill_callback(self, callback: Callable):
        """Register callback for order fills"""
        self.fill_callbacks.append(callback)
    
    def register_order_callback(self, callback: Callable):
        """Register callback for order updates"""
        self.order_callbacks.append(callback)
    
    def update_market_data(self, symbol: str, bid: Decimal, ask: Decimal, 
                          last_price: Decimal, volume: Decimal = Decimal('0')):
        """Update market data for a symbol"""
        self.market_data[symbol] = MarketData(
            symbol=symbol,
            bid=bid,
            ask=ask,
            last_price=last_price,
            volume=volume
        )
        
        # Check for order fills after market data update
        asyncio.create_task(self._check_order_fills(symbol))
    
    def update_order_book(self, symbol: str, bids: List[Tuple[Decimal, Decimal]], 
                         asks: List[Tuple[Decimal, Decimal]]):
        """Update order book for a symbol"""
        bid_levels = [OrderBookLevel(price, qty) for price, qty in bids]
        ask_levels = [OrderBookLevel(price, qty) for price, qty in asks]
        
        self.order_books[symbol] = OrderBook(
            symbol=symbol,
            bids=sorted(bid_levels, key=lambda x: x.price, reverse=True),
            asks=sorted(ask_levels, key=lambda x: x.price)
        )
    
    async def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                         quantity: Decimal, price: Decimal = None,
                         client_order_id: str = None) -> Order:
        """Place a new order"""
        
        # Generate order ID
        order_id = str(uuid.uuid4())
        if not client_order_id:
            client_order_id = f"paper_{int(datetime.now(timezone.utc).timestamp())}"
        
        # Validate order
        if symbol not in self.exchange_info:
            raise ValueError(f"Unknown symbol: {symbol}")
        
        sizing_calc = self.sizing_calculators[symbol]
        
        if order_type == OrderType.LIMIT and price is None:
            raise ValueError("Price required for limit orders")
        
        # Use market price for market orders
        if order_type == OrderType.MARKET:
            if symbol not in self.market_data:
                raise ValueError(f"No market data available for {symbol}")
            
            market = self.market_data[symbol]
            price = market.ask if side == OrderSide.BUY else market.bid
        
        # Check minimum notional before validation
        symbol_info = self.exchange_info[symbol]
        if not ensure_min_notional(price, quantity, symbol_info.min_notional):
            raise ValueError(
                f"Order notional {price * quantity} below minimum {symbol_info.min_notional} for {symbol}"
            )
        
        # Validate order parameters
        is_valid, error_msg = sizing_calc.validate_order(price, quantity)
        if not is_valid:
            raise ValueError(f"Invalid order: {error_msg}")
        
        # Check balance
        if not self._check_balance(symbol, side, quantity, price):
            raise ValueError("Insufficient balance")
        
        # Create order
        order = Order(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.NEW
        )
        
        # Reserve balance
        self._reserve_balance(order)
        
        # Store order
        self.orders[order_id] = order
        
        # Call callbacks
        for callback in self.order_callbacks:
            try:
                await callback(order, "NEW")
            except Exception as e:
                logger.error(f"Error in order callback: {e}")
        
        # Try immediate fill for market orders
        if order_type == OrderType.MARKET:
            await self._try_fill_order(order)
        
        logger.info(f"Placed {side.value} order: {quantity} {symbol} @ {price}")
        return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        if not order.is_active:
            return False
        
        # Update order status
        order.status = OrderStatus.CANCELED
        order.update_time = datetime.now(timezone.utc)
        
        # Release reserved balance
        self._release_balance(order)
        
        # Move to history
        self.order_history.append(order)
        del self.orders[order_id]
        
        # Call callbacks
        for callback in self.order_callbacks:
            try:
                await callback(order, "CANCELED")
            except Exception as e:
                logger.error(f"Error in order callback: {e}")
        
        logger.info(f"Canceled order {order_id}")
        return True
    
    async def _check_order_fills(self, symbol: str):
        """Check if any orders should be filled based on market data"""
        if symbol not in self.market_data:
            return
        
        market = self.market_data[symbol]
        orders_to_check = [order for order in self.orders.values() 
                          if order.symbol == symbol and order.is_active]
        
        for order in orders_to_check:
            await self._try_fill_order(order)
    
    async def _try_fill_order(self, order: Order):
        """Try to fill an order"""
        if not order.is_active:
            return
        
        symbol = order.symbol
        if symbol not in self.market_data:
            return
        
        market = self.market_data[symbol]
        should_fill = False
        fill_price = order.price
        
        if order.order_type == OrderType.MARKET:
            should_fill = True
            fill_price = market.ask if order.side == OrderSide.BUY else market.bid
        
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and market.ask <= order.price:
                should_fill = True
                fill_price = order.price  # Limit order fills at limit price or better
            elif order.side == OrderSide.SELL and market.bid >= order.price:
                should_fill = True
                fill_price = order.price
        
        if should_fill:
            await self._fill_order(order, fill_price)
    
    async def _fill_order(self, order: Order, fill_price: Decimal):
        """Execute order fill"""
        # Apply slippage
        slippage = self.slippage_model.calculate_slippage(order, self.market_data[order.symbol])
        final_price = fill_price * (1 + slippage)
        
        # Calculate fees
        sizing_calc = self.sizing_calculators[order.symbol]
        fee_info = sizing_calc.calculate_fees(final_price, order.quantity, is_maker=True)
        
        # Update order
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = final_price
        order.fee = fee_info["fee_amount"]
        order.update_time = datetime.now(timezone.utc)
        
        # Update balances
        self._execute_trade(order, final_price, fee_info["fee_amount"])
        
        # Update position
        self._update_position(order, final_price)
        
        # Record trade
        trade = {
            "id": str(uuid.uuid4()),
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "price": final_price,
            "fee": fee_info["fee_amount"],
            "timestamp": datetime.now(timezone.utc),
            "slippage": slippage
        }
        self.trades.append(trade)
        
        # Move to history
        self.order_history.append(order)
        del self.orders[order.order_id]
        
        # Call callbacks
        for callback in self.fill_callbacks:
            try:
                await callback(order, final_price)
            except Exception as e:
                logger.error(f"Error in fill callback: {e}")
        
        for callback in self.order_callbacks:
            try:
                await callback(order, "FILLED")
            except Exception as e:
                logger.error(f"Error in order callback: {e}")
        
        logger.info(f"Filled order: {order.quantity} {order.symbol} @ {final_price} (slippage: {slippage:.4%})")
    
    def _check_balance(self, symbol: str, side: OrderSide, quantity: Decimal, price: Decimal) -> bool:
        """Check if sufficient balance is available"""
        base_asset, quote_asset = self._parse_symbol(symbol)
        
        if side == OrderSide.BUY:
            # Need quote asset to buy
            required = quantity * price
            available = self.balances.get(quote_asset, Balance(quote_asset, Decimal('0'), Decimal('0'))).free
            return available >= required
        else:
            # Need base asset to sell
            available = self.balances.get(base_asset, Balance(base_asset, Decimal('0'), Decimal('0'))).free
            return available >= quantity
    
    def _reserve_balance(self, order: Order):
        """Reserve balance for an order"""
        base_asset, quote_asset = self._parse_symbol(order.symbol)
        
        if order.side == OrderSide.BUY:
            amount = order.quantity * order.price
            if quote_asset in self.balances:
                self.balances[quote_asset].free -= amount
                self.balances[quote_asset].locked += amount
        else:
            if base_asset in self.balances:
                self.balances[base_asset].free -= order.quantity
                self.balances[base_asset].locked += order.quantity
    
    def _release_balance(self, order: Order):
        """Release reserved balance"""
        base_asset, quote_asset = self._parse_symbol(order.symbol)
        
        if order.side == OrderSide.BUY:
            amount = order.quantity * order.price
            if quote_asset in self.balances:
                self.balances[quote_asset].locked -= amount
                self.balances[quote_asset].free += amount
        else:
            if base_asset in self.balances:
                self.balances[base_asset].locked -= order.quantity
                self.balances[base_asset].free += order.quantity
    
    def _execute_trade(self, order: Order, fill_price: Decimal, fee: Decimal):
        """Execute trade and update balances"""
        base_asset, quote_asset = self._parse_symbol(order.symbol)
        
        if order.side == OrderSide.BUY:
            # Release locked quote asset
            locked_amount = order.quantity * order.price
            if quote_asset in self.balances:
                self.balances[quote_asset].locked -= locked_amount
            
            # Add base asset (minus any fees in base asset)
            if base_asset not in self.balances:
                self.balances[base_asset] = Balance(base_asset, Decimal('0'), Decimal('0'))
            self.balances[base_asset].free += order.quantity
            
            # Deduct fee from quote asset
            if quote_asset in self.balances:
                self.balances[quote_asset].free -= fee
        
        else:  # SELL
            # Release locked base asset
            if base_asset in self.balances:
                self.balances[base_asset].locked -= order.quantity
            
            # Add quote asset (minus fees)
            quote_amount = order.quantity * fill_price - fee
            if quote_asset not in self.balances:
                self.balances[quote_asset] = Balance(quote_asset, Decimal('0'), Decimal('0'))
            self.balances[quote_asset].free += quote_amount
    
    def _update_position(self, order: Order, fill_price: Decimal):
        """Update position after trade"""
        symbol = order.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=Decimal('0'),
                entry_price=Decimal('0'),
                current_price=fill_price,
                unrealized_pnl=Decimal('0'),
                realized_pnl=Decimal('0')
            )
        
        position = self.positions[symbol]
        
        if order.side == OrderSide.BUY:
            # Calculate new average price
            if position.quantity + order.quantity != 0:
                total_cost = (position.quantity * position.entry_price) + (order.quantity * fill_price)
                position.quantity += order.quantity
                position.entry_price = total_cost / position.quantity
            else:
                position.quantity = order.quantity
                position.entry_price = fill_price
        else:  # SELL
            # Realize PnL
            pnl = (fill_price - position.entry_price) * order.quantity
            position.realized_pnl += pnl
            position.quantity -= order.quantity
        
        position.current_price = fill_price
        position.timestamp = datetime.now(timezone.utc)
    
    def _parse_symbol(self, symbol: str) -> Tuple[str, str]:
        """Parse trading symbol into base and quote assets"""
        # This is a simplified parser - in reality you'd use exchange info
        if symbol.endswith("USDC"):
            return symbol[:-4], "USDC"
        elif symbol.endswith("BTC"):
            return symbol[:-3], "BTC"
        elif symbol.endswith("ETH"):
            return symbol[:-3], "ETH"
        else:
            # Default assumption
            return symbol[:3], symbol[3:]
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        return {
            "balances": [
                {
                    "asset": balance.asset,
                    "free": str(balance.free),
                    "locked": str(balance.locked)
                }
                for balance in self.balances.values()
            ],
            "total_wallet_balance": str(sum(b.total for b in self.balances.values() if b.asset == "USDC")),
            "total_unrealized_profit": str(sum(p.unrealized_pnl for p in self.positions.values())),
            "total_margin_balance": str(sum(b.total for b in self.balances.values() if b.asset == "USDC"))
        }
    
    def get_position_info(self, symbol: str = None) -> List[Dict]:
        """Get position information"""
        positions = [self.positions[symbol]] if symbol and symbol in self.positions else self.positions.values()
        
        return [
            {
                "symbol": pos.symbol,
                "size": str(pos.quantity),
                "entry_price": str(pos.entry_price),
                "mark_price": str(pos.current_price),
                "unrealized_profit": str(pos.unrealized_pnl),
                "realized_profit": str(pos.realized_pnl)
            }
            for pos in positions
        ]
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get order status"""
        order = self.orders.get(order_id)
        if not order:
            # Check history
            for hist_order in self.order_history:
                if hist_order.order_id == order_id:
                    order = hist_order
                    break
        
        if not order:
            return None
        
        return {
            "order_id": order.order_id,
            "client_order_id": order.client_order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "type": order.order_type.value,
            "status": order.status.value,
            "orig_qty": str(order.quantity),
            "executed_qty": str(order.filled_quantity),
            "price": str(order.price),
            "time": int(order.timestamp.timestamp() * 1000),
            "update_time": int(order.update_time.timestamp() * 1000) if order.update_time else None
        }
    
    def get_trade_history(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get trade history"""
        trades = self.trades
        
        if symbol:
            trades = [t for t in trades if t["symbol"] == symbol]
        
        return trades[-limit:]
