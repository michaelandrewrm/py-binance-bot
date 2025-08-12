"""
State - BotState dataclasses and serialization

This module defines the core state management for the trading bot,
including data structures for tracking positions, orders, and bot status.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from decimal import Decimal
from datetime import datetime, timezone
from enum import Enum
import json
import pickle
from pathlib import Path

def utc_now() -> datetime:
    """Get current UTC datetime with timezone info"""
    return datetime.now(timezone.utc)

def utc_isoformat(dt: datetime) -> str:
    """Convert datetime to ISO 8601 string with Z suffix for UTC"""
    if dt.tzinfo is None:
        # Assume naive datetime is UTC
        dt = dt.replace(tzinfo=timezone.utc)
    elif dt.tzinfo != timezone.utc:
        # Convert to UTC if not already
        dt = dt.astimezone(timezone.utc)
    
    # Format with Z suffix for UTC
    return dt.isoformat().replace('+00:00', 'Z')

class StateJSONEncoder(json.JSONEncoder):
    """JSON encoder for bot state that handles datetime objects with Z suffix"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return utc_isoformat(obj)
        elif isinstance(obj, Decimal):
            return str(obj)
        elif hasattr(obj, 'value'):  # Enum objects
            return obj.value
        return super().default(obj)

class BotStatus(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"

class OrderStatus(Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"

@dataclass
class Position:
    """Current position information"""
    symbol: str
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    timestamp: datetime = field(default_factory=utc_now)
    
    @property
    def position_value(self) -> Decimal:
        """Calculate current position value"""
        return self.quantity * self.current_price
    
    @property
    def is_long(self) -> bool:
        """Check if position is long"""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short"""
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        """Check if position is flat (no position)"""
        return self.quantity == 0

@dataclass
class Order:
    """Order information"""
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Decimal
    status: OrderStatus
    filled_quantity: Decimal = Decimal('0')
    filled_price: Decimal = Decimal('0')
    fee: Decimal = Decimal('0')
    timestamp: datetime = field(default_factory=utc_now)
    update_time: Optional[datetime] = None
    grid_level: Optional[int] = None
    
    @property
    def remaining_quantity(self) -> Decimal:
        """Calculate remaining quantity to fill"""
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled"""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        """Check if order is active (can be filled)"""
        return self.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]

@dataclass
class Balance:
    """Account balance information"""
    asset: str
    free: Decimal
    locked: Decimal
    timestamp: datetime = field(default_factory=utc_now)
    
    @property
    def total(self) -> Decimal:
        """Total balance (free + locked)"""
        return self.free + self.locked

@dataclass
class GridState:
    """Grid trading specific state"""
    center_price: Decimal
    grid_spacing: Decimal
    num_levels_up: int
    num_levels_down: int
    active_levels: List[int] = field(default_factory=list)
    filled_levels: List[int] = field(default_factory=list)
    total_trades: int = 0
    total_profit: Decimal = Decimal('0')
    
@dataclass
class TradingMetrics:
    """Trading performance metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    unrealized_pnl: Decimal = Decimal('0')
    fees_paid: Decimal = Decimal('0')
    max_drawdown: Decimal = Decimal('0')
    sharpe_ratio: Optional[Decimal] = None
    win_rate: Optional[Decimal] = None
    profit_factor: Optional[Decimal] = None
    last_updated: datetime = field(default_factory=utc_now)
    
    def update_trade_stats(self):
        """Update calculated statistics"""
        if self.total_trades > 0:
            self.win_rate = Decimal(self.winning_trades) / Decimal(self.total_trades)
        
        if self.losing_trades > 0 and self.total_pnl > 0:
            total_losses = abs(self.total_pnl - (self.winning_trades * (self.total_pnl / self.total_trades)))
            if total_losses > 0:
                self.profit_factor = self.total_pnl / total_losses

@dataclass
class BotState:
    """Complete bot state"""
    status: BotStatus = BotStatus.STOPPED
    symbol: str = ""
    start_time: Optional[datetime] = None
    last_update: datetime = field(default_factory=utc_now)
    
    # Trading state
    positions: Dict[str, Position] = field(default_factory=dict)
    active_orders: Dict[str, Order] = field(default_factory=dict)
    order_history: List[Order] = field(default_factory=list)
    balances: Dict[str, Balance] = field(default_factory=dict)
    
    # Strategy state
    grid_state: Optional[GridState] = None
    
    # Performance
    metrics: TradingMetrics = field(default_factory=TradingMetrics)
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Error tracking
    error_count: int = 0
    last_error: Optional[str] = None
    error_timestamp: Optional[datetime] = None
    
    # Safety and manual intervention flags
    requires_manual_resume: bool = False
    flatten_reason: Optional[str] = None
    flatten_timestamp: Optional[datetime] = None
    
    def update_position(self, symbol: str, quantity: Decimal, price: Decimal):
        """Update position information"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                current_price=price,
                unrealized_pnl=Decimal('0'),
                realized_pnl=Decimal('0')
            )
        else:
            pos = self.positions[symbol]
            # Update quantity and average price
            if pos.quantity + quantity != 0:
                total_cost = (pos.quantity * pos.entry_price) + (quantity * price)
                pos.quantity += quantity
                pos.entry_price = total_cost / pos.quantity
            else:
                # Position closed
                pos.realized_pnl += (price - pos.entry_price) * quantity
                pos.quantity = Decimal('0')
            
            pos.current_price = price
            pos.timestamp = utc_now()
    
    def add_order(self, order: Order):
        """Add a new order"""
        self.active_orders[order.order_id] = order
    
    def update_order(self, order_id: str, **updates):
        """Update an existing order"""
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            for key, value in updates.items():
                if hasattr(order, key):
                    setattr(order, key, value)
            order.update_time = utc_now()
            
            # Move to history if filled or canceled
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]:
                self.order_history.append(order)
                del self.active_orders[order_id]
    
    def update_balance(self, asset: str, free: Decimal, locked: Decimal):
        """Update balance information"""
        self.balances[asset] = Balance(
            asset=asset,
            free=free,
            locked=locked
        )
    
    def record_error(self, error_message: str):
        """Record an error"""
        self.error_count += 1
        self.last_error = error_message
        self.error_timestamp = utc_now()
    
    def set_flatten_state(self, reason: str):
        """Set the bot to require manual resume after flatten"""
        self.requires_manual_resume = True
        self.flatten_reason = reason
        self.flatten_timestamp = utc_now()
        self.status = BotStatus.PAUSED
    
    def clear_manual_resume(self):
        """Clear the manual resume requirement"""
        self.requires_manual_resume = False
        self.flatten_reason = None
        self.flatten_timestamp = None
    
    def get_total_portfolio_value(self, prices: Dict[str, Decimal]) -> Decimal:
        """Calculate total portfolio value"""
        total_value = Decimal('0')
        
        for asset, balance in self.balances.items():
            if asset == "USDC":
                total_value += balance.total
            elif asset in prices:
                total_value += balance.total * prices[asset]
        
        return total_value
    
    def get_unrealized_pnl(self) -> Decimal:
        """Calculate total unrealized PnL"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def get_realized_pnl(self) -> Decimal:
        """Calculate total realized PnL"""
        return sum(pos.realized_pnl for pos in self.positions.values())

class StateSerializer:
    """Handles serialization and deserialization of bot state"""
    
    @staticmethod
    def to_dict(state: BotState) -> Dict:
        """Convert bot state to dictionary"""
        return asdict(state)
    
    @staticmethod
    def from_dict(data: Dict) -> BotState:
        """Create bot state from dictionary"""
        # Convert string enums back to enum objects
        if 'status' in data:
            data['status'] = BotStatus(data['status'])
        
        # Convert order data
        if 'active_orders' in data:
            orders = {}
            for order_id, order_data in data['active_orders'].items():
                order_data['side'] = OrderSide(order_data['side'])
                order_data['order_type'] = OrderType(order_data['order_type'])
                order_data['status'] = OrderStatus(order_data['status'])
                orders[order_id] = Order(**order_data)
            data['active_orders'] = orders
        
        # Convert order history
        if 'order_history' in data:
            history = []
            for order_data in data['order_history']:
                order_data['side'] = OrderSide(order_data['side'])
                order_data['order_type'] = OrderType(order_data['order_type'])
                order_data['status'] = OrderStatus(order_data['status'])
                history.append(Order(**order_data))
            data['order_history'] = history
        
        return BotState(**data)
    
    @staticmethod
    def save_to_file(state: BotState, filepath: Path):
        """Save bot state to file"""
        data = StateSerializer.to_dict(state)
        
        if filepath.suffix == '.json':
            # JSON serialization (human readable but loses precision)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, cls=StateJSONEncoder)
        elif filepath.suffix == '.pkl':
            # Pickle serialization (preserves types and precision)
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    @staticmethod
    def load_from_file(filepath: Path) -> BotState:
        """Load bot state from file"""
        if not filepath.exists():
            raise FileNotFoundError(f"State file not found: {filepath}")
        
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            return StateSerializer.from_dict(data)
        elif filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

# Convenience functions
def create_empty_state(symbol: str, config: Dict = None) -> BotState:
    """Create a new empty bot state"""
    return BotState(
        symbol=symbol,
        config=config or {},
        start_time=utc_now()
    )

def save_state(state: BotState, filepath: str):
    """Save bot state to file"""
    StateSerializer.save_to_file(state, Path(filepath))

def load_state(filepath: str) -> BotState:
    """Load bot state from file"""
    return StateSerializer.load_from_file(Path(filepath))
