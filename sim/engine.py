"""
Engine - Event-driven backtester (wraps current simulate)

This module provides a comprehensive event-driven backtesting engine
that can simulate trading strategies with realistic market conditions.
"""

from typing import Dict, List, Optional, Callable, Any, Tuple
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
from abc import ABC, abstractmethod

from data.schema import KlineData, TradeData, TickerData, TimeFrame, TradeRecord
from core.state import Order, OrderSide, OrderType, OrderStatus, Position, Balance, BotState
from core.sizing import SizingCalculator, ExchangeInfo, FeeConfig, ensure_min_notional
from exec.paper_broker import PaperBroker, MarketData, SlippageModel, LinearSlippageModel
from .slippage import SlippageContext
from .evaluation import PerformanceEvaluator, PerformanceMetrics

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Types of simulation events"""
    MARKET_DATA = "market_data"
    ORDER_FILL = "order_fill"
    TIMER = "timer"
    SIGNAL = "signal"
    RISK_CHECK = "risk_check"

@dataclass
class SimulationEvent:
    """Base simulation event"""
    timestamp: datetime
    event_type: EventType
    data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    timeframe: TimeFrame = TimeFrame.M5
    commission_rate: Decimal = Decimal('0.001')
    slippage_model: Optional[SlippageModel] = None
    enable_shorts: bool = False
    max_position_size: Decimal = Decimal('0.95')  # 95% of capital
    rebalance_frequency: timedelta = timedelta(hours=1)

class TradingStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.state: Optional[BotState] = None
        self.broker: Optional[PaperBroker] = None
    
    @abstractmethod
    async def on_market_data(self, kline: KlineData) -> List[Order]:
        """
        Handle new market data
        
        Args:
            kline: New kline data
            
        Returns:
            List of orders to place
        """
        pass
    
    @abstractmethod
    async def on_order_fill(self, order: Order, fill_price: Decimal) -> List[Order]:
        """
        Handle order fill
        
        Args:
            order: Filled order
            fill_price: Actual fill price
            
        Returns:
            List of new orders to place
        """
        pass
    
    async def on_timer(self, timestamp: datetime) -> List[Order]:
        """
        Handle timer events (optional)
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            List of orders to place
        """
        return []
    
    async def on_start(self, state: BotState, broker: PaperBroker):
        """Called when simulation starts"""
        self.state = state
        self.broker = broker
    
    async def on_end(self):
        """Called when simulation ends"""
        pass

class SimpleGridStrategy(TradingStrategy):
    """Example grid trading strategy for demonstration"""
    
    def __init__(self, grid_spacing: Decimal = Decimal('0.01'), 
                 num_levels: int = 5, order_size: Decimal = Decimal('100')):
        super().__init__("SimpleGrid")
        self.grid_spacing = grid_spacing
        self.num_levels = num_levels
        self.order_size = order_size
        self.center_price: Optional[Decimal] = None
        self.grid_orders: Dict[int, str] = {}  # level -> order_id
    
    async def on_market_data(self, kline: KlineData) -> List[Order]:
        """Place grid orders around current price"""
        current_price = kline.close_price
        
        # Initialize center price
        if self.center_price is None:
            self.center_price = current_price
            return await self._place_initial_grid(current_price)
        
        # Check if we need to rebalance grid
        price_deviation = abs(current_price - self.center_price) / self.center_price
        if price_deviation > self.grid_spacing * 2:
            self.center_price = current_price
            # Cancel existing orders and place new grid
            for order_id in self.grid_orders.values():
                await self.broker.cancel_order(order_id)
            self.grid_orders.clear()
            return await self._place_initial_grid(current_price)
        
        return []
    
    async def on_order_fill(self, order: Order, fill_price: Decimal) -> List[Order]:
        """Handle grid order fills"""
        # Remove filled order from tracking
        level_to_remove = None
        for level, order_id in self.grid_orders.items():
            if order_id == order.order_id:
                level_to_remove = level
                break
        
        if level_to_remove is not None:
            del self.grid_orders[level_to_remove]
        
        # Place opposite order
        new_orders = []
        if order.side == OrderSide.BUY:
            # Place sell order above
            sell_price = fill_price * (Decimal('1') + self.grid_spacing)
            sell_order = Order(
                order_id="",  # Will be generated
                client_order_id=f"grid_sell_{int(datetime.now(timezone.utc).timestamp())}",
                symbol=order.symbol,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=order.quantity,
                price=sell_price,
                status=OrderStatus.NEW
            )
            new_orders.append(sell_order)
        else:
            # Place buy order below
            buy_price = fill_price * (Decimal('1') - self.grid_spacing)
            buy_order = Order(
                order_id="",  # Will be generated
                client_order_id=f"grid_buy_{int(datetime.now(timezone.utc).timestamp())}",
                symbol=order.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=order.quantity,
                price=buy_price,
                status=OrderStatus.NEW
            )
            new_orders.append(buy_order)
        
        return new_orders
    
    async def _place_initial_grid(self, current_price: Decimal) -> List[Order]:
        """Place initial grid orders"""
        orders = []
        
        # Place buy orders below current price
        for i in range(1, self.num_levels + 1):
            buy_price = current_price * (Decimal('1') - self.grid_spacing * i)
            buy_order = Order(
                order_id="",
                client_order_id=f"grid_buy_{i}_{int(datetime.now(timezone.utc).timestamp())}",
                symbol=self.state.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=self.order_size,
                price=buy_price,
                status=OrderStatus.NEW
            )
            orders.append(buy_order)
        
        # Place sell orders above current price
        for i in range(1, self.num_levels + 1):
            sell_price = current_price * (Decimal('1') + self.grid_spacing * i)
            sell_order = Order(
                order_id="",
                client_order_id=f"grid_sell_{i}_{int(datetime.now(timezone.utc).timestamp())}",
                symbol=self.state.symbol,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=self.order_size,
                price=sell_price,
                status=OrderStatus.NEW
            )
            orders.append(sell_order)
        
        return orders

class BacktestEngine:
    """
    Event-driven backtesting engine
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.events: List[SimulationEvent] = []
        self.current_time: Optional[datetime] = None
        self.strategy: Optional[TradingStrategy] = None
        self.broker: Optional[PaperBroker] = None
        self.state: Optional[BotState] = None
        self.evaluator = PerformanceEvaluator()
        
        # Callbacks
        self.progress_callbacks: List[Callable] = []
        self.event_callbacks: List[Callable] = []
    
    def register_progress_callback(self, callback: Callable):
        """Register callback for progress updates"""
        self.progress_callbacks.append(callback)
    
    def register_event_callback(self, callback: Callable):
        """Register callback for simulation events"""
        self.event_callbacks.append(callback)
    
    async def run_backtest(self, strategy: TradingStrategy, 
                          market_data: List[KlineData]) -> Dict:
        """
        Run complete backtest
        
        Args:
            strategy: Trading strategy to test
            market_data: Historical market data
            
        Returns:
            Backtest results dictionary
        """
        logger.info(f"Starting backtest: {strategy.name} on {self.config.symbol}")
        
        # Initialize components
        await self._initialize_backtest(strategy, market_data)
        
        # Generate events from market data
        self._generate_events(market_data)
        
        # Run simulation
        await self._run_simulation()
        
        # Generate results
        results = await self._generate_results()
        
        logger.info(f"Backtest completed: {len(self.events)} events processed")
        return results
    
    async def _initialize_backtest(self, strategy: TradingStrategy, 
                                  market_data: List[KlineData]):
        """Initialize backtest components"""
        self.strategy = strategy
        
        # Create exchange info
        exchange_info = {
            self.config.symbol: ExchangeInfo(
                symbol=self.config.symbol,
                min_qty=Decimal('0.001'),
                max_qty=Decimal('1000000'),
                step_size=Decimal('0.001'),
                tick_size=Decimal('0.01'),
                min_notional=Decimal('10'),
                base_asset_precision=3,
                quote_asset_precision=2
            )
        }
        
        # Create fee config
        fee_config = FeeConfig(
            maker_fee=self.config.commission_rate,
            taker_fee=self.config.commission_rate,
            use_bnb=False
        )
        
        # Create slippage model
        slippage_model = self.config.slippage_model or LinearSlippageModel()
        
        # Create paper broker
        initial_balances = {
            "USDT": self.config.initial_capital,
            self.config.symbol.replace("USDT", ""): Decimal('0')
        }
        
        self.broker = PaperBroker(
            initial_balances=initial_balances,
            exchange_info=exchange_info,
            fee_config=fee_config,
            slippage_model=slippage_model
        )
        
        # Register broker callbacks
        self.broker.register_fill_callback(self._on_order_fill)
        self.broker.register_order_callback(self._on_order_update)
        
        # Create bot state
        self.state = BotState(
            symbol=self.config.symbol,
            start_time=self.config.start_date,
            config=self.config.__dict__
        )
        
        # Initialize strategy
        await strategy.on_start(self.state, self.broker)
    
    def _generate_events(self, market_data: List[KlineData]):
        """Generate simulation events from market data"""
        self.events.clear()
        
        # Create market data events
        for kline in market_data:
            if self.config.start_date <= kline.open_time <= self.config.end_date:
                event = SimulationEvent(
                    timestamp=kline.open_time,
                    event_type=EventType.MARKET_DATA,
                    data={"kline": kline}
                )
                self.events.append(event)
        
        # Create timer events
        current_time = self.config.start_date
        while current_time <= self.config.end_date:
            event = SimulationEvent(
                timestamp=current_time,
                event_type=EventType.TIMER,
                data={}
            )
            self.events.append(event)
            current_time += self.config.rebalance_frequency
        
        # Sort events by timestamp
        self.events.sort(key=lambda x: x.timestamp)
        
        logger.info(f"Generated {len(self.events)} simulation events")
    
    async def _run_simulation(self):
        """Run the main simulation loop"""
        total_events = len(self.events)
        
        for i, event in enumerate(self.events):
            self.current_time = event.timestamp
            
            # Process event
            await self._process_event(event)
            
            # Update progress
            if i % 100 == 0:  # Update every 100 events
                progress = i / total_events
                for callback in self.progress_callbacks:
                    try:
                        await callback(progress, event.timestamp)
                    except Exception as e:
                        logger.error(f"Error in progress callback: {e}")
            
            # Call event callbacks
            for callback in self.event_callbacks:
                try:
                    await callback(event, self.state)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")
        
        # Final strategy cleanup
        await self.strategy.on_end()
    
    async def _process_event(self, event: SimulationEvent):
        """Process a single simulation event"""
        try:
            if event.event_type == EventType.MARKET_DATA:
                await self._process_market_data_event(event)
            elif event.event_type == EventType.TIMER:
                await self._process_timer_event(event)
            
            # Update state timestamp
            self.state.last_update = event.timestamp
            
        except Exception as e:
            logger.error(f"Error processing event {event.event_type} at {event.timestamp}: {e}")
            self.state.record_error(str(e))
    
    async def _process_market_data_event(self, event: SimulationEvent):
        """Process market data event"""
        kline = event.data["kline"]
        
        # Update broker market data
        self.broker.update_market_data(
            symbol=kline.symbol,
            bid=kline.close_price * Decimal('0.999'),  # Approximate bid
            ask=kline.close_price * Decimal('1.001'),  # Approximate ask
            last_price=kline.close_price,
            volume=kline.volume
        )
        
        # Call strategy
        new_orders = await self.strategy.on_market_data(kline)
        
        # Place new orders
        for order in new_orders:
            try:
                await self._place_order(order)
            except Exception as e:
                logger.error(f"Error placing order: {e}")
    
    async def _process_timer_event(self, event: SimulationEvent):
        """Process timer event"""
        new_orders = await self.strategy.on_timer(event.timestamp)
        
        # Place new orders
        for order in new_orders:
            try:
                await self._place_order(order)
            except Exception as e:
                logger.error(f"Error placing order from timer: {e}")
    
    async def _place_order(self, order: Order):
        """Place order through broker with minimum notional check"""
        try:
            # Get exchange info for the symbol
            symbol_info = self.broker.exchange_info.get(order.symbol)
            if symbol_info:
                # Check minimum notional before placing order
                if not ensure_min_notional(order.price, order.quantity, symbol_info.min_notional):
                    logger.warning(
                        f"Order rejected: notional {order.price * order.quantity} "
                        f"below minimum {symbol_info.min_notional} for {order.symbol}"
                    )
                    return
            
            placed_order = await self.broker.place_order(
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                price=order.price,
                client_order_id=order.client_order_id
            )
            
            # Update state
            self.state.add_order(placed_order)
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise
    
    async def _on_order_fill(self, order: Order, fill_price: Decimal):
        """Handle order fill callback"""
        # Update state
        self.state.update_order(order.order_id, 
                               status=OrderStatus.FILLED,
                               filled_quantity=order.quantity,
                               filled_price=fill_price)
        
        # Update position
        if order.side == OrderSide.BUY:
            quantity_change = order.quantity
        else:
            quantity_change = -order.quantity
        
        self.state.update_position(order.symbol, quantity_change, fill_price)
        
        # Call strategy
        try:
            new_orders = await self.strategy.on_order_fill(order, fill_price)
            for new_order in new_orders:
                await self._place_order(new_order)
        except Exception as e:
            logger.error(f"Error in strategy order fill handler: {e}")
    
    async def _on_order_update(self, order: Order, update_type: str):
        """Handle order update callback"""
        # Update state based on update type
        if update_type == "CANCELED":
            self.state.update_order(order.order_id, status=OrderStatus.CANCELED)
    
    async def _generate_results(self) -> Dict:
        """Generate backtest results"""
        # Get final account info
        account_info = self.broker.get_account_info()
        
        # Calculate performance metrics
        equity_curve = []  # Would need to track this during simulation
        trades = self.broker.get_trade_history()
        
        # Create trade records
        trade_records = []
        for trade in trades:
            trade_record = TradeRecord(
                trade_id=trade["id"],
                symbol=trade["symbol"],
                side=trade["side"],
                quantity=Decimal(str(trade["quantity"])),
                price=Decimal(str(trade["price"])),
                fee=Decimal(str(trade["fee"])),
                fee_asset="USDT",
                timestamp=trade["timestamp"],
                order_id=trade["order_id"],
                client_order_id="",
                realized_pnl=Decimal(str(trade.get("slippage", 0)))
            )
            trade_records.append(trade_record)
        
        # Calculate final balance
        final_balance = sum(
            Decimal(balance["free"]) + Decimal(balance["locked"])
            for balance in account_info["balances"]
            if balance["asset"] == "USDT"
        )
        
        total_return = (final_balance - self.config.initial_capital) / self.config.initial_capital
        
        return {
            "symbol": self.config.symbol,
            "start_date": self.config.start_date,
            "end_date": self.config.end_date,
            "initial_capital": self.config.initial_capital,
            "final_balance": final_balance,
            "total_return": total_return,
            "total_trades": len(trade_records),
            "account_info": account_info,
            "trades": trade_records,
            "state": self.state
        }

# Utility functions

def create_simple_backtest(symbol: str, start_date: datetime, end_date: datetime,
                          initial_capital: Decimal = Decimal('10000')) -> BacktestEngine:
    """Create a simple backtest configuration"""
    config = BacktestConfig(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )
    return BacktestEngine(config)

async def run_grid_strategy_backtest(symbol: str, market_data: List[KlineData],
                                   initial_capital: Decimal = Decimal('10000')) -> Dict:
    """Run a simple grid strategy backtest"""
    # Create backtest engine
    config = BacktestConfig(
        symbol=symbol,
        start_date=market_data[0].open_time,
        end_date=market_data[-1].close_time,
        initial_capital=initial_capital
    )
    
    engine = BacktestEngine(config)
    
    # Create grid strategy
    strategy = SimpleGridStrategy(
        grid_spacing=Decimal('0.01'),  # 1% grid spacing
        num_levels=5,
        order_size=initial_capital / Decimal('20')  # 5% per order
    )
    
    # Run backtest
    results = await engine.run_backtest(strategy, market_data)
    
    return results
