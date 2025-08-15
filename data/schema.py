"""
Schema - Typed records for IO

This module defines data schemas and type definitions for data exchange
between different components of the trading system.
"""

from typing import List, Optional, Dict, Any, Union
from decimal import Decimal
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
import json


def utc_isoformat(dt: datetime) -> str:
    """Convert datetime to ISO 8601 string with Z suffix for UTC"""
    if dt.tzinfo is None:
        # Assume naive datetime is UTC
        dt = dt.replace(tzinfo=timezone.utc)
    elif dt.tzinfo != timezone.utc:
        # Convert to UTC if not already
        dt = dt.astimezone(timezone.utc)

    # Format with Z suffix for UTC
    return dt.isoformat().replace("+00:00", "Z")


class TimeFrame(Enum):
    """Trading timeframes"""

    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    H6 = "6h"
    H8 = "8h"
    H12 = "12h"
    D1 = "1d"
    D3 = "3d"
    W1 = "1w"
    MN1 = "1M"


class MarketDataType(Enum):
    """Types of market data"""

    KLINE = "kline"
    TRADE = "trade"
    TICKER = "ticker"
    DEPTH = "depth"
    AGG_TRADE = "aggTrade"


@dataclass
class KlineData:
    """OHLCV candlestick data"""

    symbol: str
    open_time: datetime
    close_time: datetime
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: Decimal
    quote_volume: Decimal
    trades_count: int
    taker_buy_volume: Decimal
    taker_buy_quote_volume: Decimal
    timeframe: TimeFrame

    @property
    def typical_price(self) -> Decimal:
        """Calculate typical price (HLC/3)"""
        return (self.high_price + self.low_price + self.close_price) / 3

    @property
    def weighted_price(self) -> Decimal:
        """Calculate volume weighted average price"""
        if self.volume > 0:
            return self.quote_volume / self.volume
        return self.close_price

    @property
    def price_range(self) -> Decimal:
        """Calculate price range (high - low)"""
        return self.high_price - self.low_price

    @property
    def body_size(self) -> Decimal:
        """Calculate candle body size"""
        return abs(self.close_price - self.open_price)

    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish"""
        return self.close_price > self.open_price

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "symbol": self.symbol,
            "open_time": utc_isoformat(self.open_time),
            "close_time": utc_isoformat(self.close_time),
            "open": str(self.open_price),
            "high": str(self.high_price),
            "low": str(self.low_price),
            "close": str(self.close_price),
            "volume": str(self.volume),
            "quote_volume": str(self.quote_volume),
            "trades_count": self.trades_count,
            "taker_buy_volume": str(self.taker_buy_volume),
            "taker_buy_quote_volume": str(self.taker_buy_quote_volume),
            "timeframe": self.timeframe.value,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "KlineData":
        """Create from dictionary"""
        return cls(
            symbol=data["symbol"],
            open_time=datetime.fromisoformat(data["open_time"]),
            close_time=datetime.fromisoformat(data["close_time"]),
            open_price=Decimal(data["open"]),
            high_price=Decimal(data["high"]),
            low_price=Decimal(data["low"]),
            close_price=Decimal(data["close"]),
            volume=Decimal(data["volume"]),
            quote_volume=Decimal(data["quote_volume"]),
            trades_count=data["trades_count"],
            taker_buy_volume=Decimal(data["taker_buy_volume"]),
            taker_buy_quote_volume=Decimal(data["taker_buy_quote_volume"]),
            timeframe=TimeFrame(data["timeframe"]),
        )


@dataclass
class TradeData:
    """Individual trade data"""

    symbol: str
    trade_id: int
    price: Decimal
    quantity: Decimal
    quote_quantity: Decimal
    timestamp: datetime
    is_buyer_maker: bool

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "symbol": self.symbol,
            "trade_id": self.trade_id,
            "price": str(self.price),
            "quantity": str(self.quantity),
            "quote_quantity": str(self.quote_quantity),
            "timestamp": utc_isoformat(self.timestamp),
            "is_buyer_maker": self.is_buyer_maker,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TradeData":
        """Create from dictionary"""
        return cls(
            symbol=data["symbol"],
            trade_id=data["trade_id"],
            price=Decimal(data["price"]),
            quantity=Decimal(data["quantity"]),
            quote_quantity=Decimal(data["quote_quantity"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            is_buyer_maker=data["is_buyer_maker"],
        )


@dataclass
class TickerData:
    """24hr ticker statistics"""

    symbol: str
    price_change: Decimal
    price_change_percent: Decimal
    weighted_avg_price: Decimal
    prev_close_price: Decimal
    last_price: Decimal
    last_qty: Decimal
    bid_price: Decimal
    bid_qty: Decimal
    ask_price: Decimal
    ask_qty: Decimal
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    volume: Decimal
    quote_volume: Decimal
    open_time: datetime
    close_time: datetime
    count: int

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "symbol": self.symbol,
            "price_change": str(self.price_change),
            "price_change_percent": str(self.price_change_percent),
            "weighted_avg_price": str(self.weighted_avg_price),
            "prev_close_price": str(self.prev_close_price),
            "last_price": str(self.last_price),
            "last_qty": str(self.last_qty),
            "bid_price": str(self.bid_price),
            "bid_qty": str(self.bid_qty),
            "ask_price": str(self.ask_price),
            "ask_qty": str(self.ask_qty),
            "open_price": str(self.open_price),
            "high_price": str(self.high_price),
            "low_price": str(self.low_price),
            "volume": str(self.volume),
            "quote_volume": str(self.quote_volume),
            "open_time": utc_isoformat(self.open_time),
            "close_time": utc_isoformat(self.close_time),
            "count": self.count,
        }


@dataclass
class DepthLevel:
    """Single level in order book depth"""

    price: Decimal
    quantity: Decimal


@dataclass
class DepthData:
    """Order book depth data"""

    symbol: str
    last_update_id: int
    bids: List[DepthLevel]
    asks: List[DepthLevel]
    timestamp: datetime

    @property
    def best_bid(self) -> Optional[Decimal]:
        """Get best bid price"""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        """Get best ask price"""
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread"""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price"""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "symbol": self.symbol,
            "last_update_id": self.last_update_id,
            "bids": [[str(level.price), str(level.quantity)] for level in self.bids],
            "asks": [[str(level.price), str(level.quantity)] for level in self.asks],
            "timestamp": utc_isoformat(self.timestamp),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DepthData":
        """Create from dictionary"""
        return cls(
            symbol=data["symbol"],
            last_update_id=data["last_update_id"],
            bids=[
                DepthLevel(Decimal(price), Decimal(qty)) for price, qty in data["bids"]
            ],
            asks=[
                DepthLevel(Decimal(price), Decimal(qty)) for price, qty in data["asks"]
            ],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


@dataclass
class FeatureData:
    """Technical analysis features"""

    symbol: str
    timestamp: datetime
    features: Dict[str, float]
    target: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "symbol": self.symbol,
            "timestamp": utc_isoformat(self.timestamp),
            "features": self.features,
            "target": self.target,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "FeatureData":
        """Create from dictionary"""
        return cls(
            symbol=data["symbol"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            features=data["features"],
            target=data.get("target"),
        )


@dataclass
class ModelPrediction:
    """ML model prediction result"""

    symbol: str
    timestamp: datetime
    prediction: float
    confidence: float
    features_used: List[str]
    model_version: str

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "symbol": self.symbol,
            "timestamp": utc_isoformat(self.timestamp),
            "prediction": self.prediction,
            "confidence": self.confidence,
            "features_used": self.features_used,
            "model_version": self.model_version,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelPrediction":
        """Create from dictionary"""
        return cls(
            symbol=data["symbol"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            prediction=data["prediction"],
            confidence=data["confidence"],
            features_used=data["features_used"],
            model_version=data["model_version"],
        )


@dataclass
class TradeRecord:
    """Executed trade record"""

    trade_id: str
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    fee: Decimal
    fee_asset: str
    timestamp: datetime
    order_id: str
    client_order_id: str
    realized_pnl: Optional[Decimal] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": str(self.quantity),
            "price": str(self.price),
            "fee": str(self.fee),
            "fee_asset": self.fee_asset,
            "timestamp": utc_isoformat(self.timestamp),
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "realized_pnl": str(self.realized_pnl) if self.realized_pnl else None,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TradeRecord":
        """Create from dictionary"""
        return cls(
            trade_id=data["trade_id"],
            symbol=data["symbol"],
            side=data["side"],
            quantity=Decimal(data["quantity"]),
            price=Decimal(data["price"]),
            fee=Decimal(data["fee"]),
            fee_asset=data["fee_asset"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            order_id=data["order_id"],
            client_order_id=data["client_order_id"],
            realized_pnl=(
                Decimal(data["realized_pnl"]) if data.get("realized_pnl") else None
            ),
        )


@dataclass
class EquityCurvePoint:
    """Single point in equity curve"""

    timestamp: datetime
    total_balance: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    fees_paid: Decimal
    trade_count: int

    @property
    def net_worth(self) -> Decimal:
        """Calculate net worth"""
        return self.total_balance + self.unrealized_pnl

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "timestamp": utc_isoformat(self.timestamp),
            "total_balance": str(self.total_balance),
            "unrealized_pnl": str(self.unrealized_pnl),
            "realized_pnl": str(self.realized_pnl),
            "fees_paid": str(self.fees_paid),
            "trade_count": self.trade_count,
            "net_worth": str(self.net_worth),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EquityCurvePoint":
        """Create from dictionary"""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            total_balance=Decimal(data["total_balance"]),
            unrealized_pnl=Decimal(data["unrealized_pnl"]),
            realized_pnl=Decimal(data["realized_pnl"]),
            fees_paid=Decimal(data["fees_paid"]),
            trade_count=data["trade_count"],
        )


@dataclass
class BacktestResult:
    """Backtest results summary"""

    symbol: str
    start_date: datetime
    end_date: datetime
    initial_balance: Decimal
    final_balance: Decimal
    total_return: Decimal
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal
    profit_factor: Decimal
    max_drawdown: Decimal
    sharpe_ratio: Decimal
    calmar_ratio: Decimal
    equity_curve: List[EquityCurvePoint]
    trades: List[TradeRecord]

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "symbol": self.symbol,
            "start_date": utc_isoformat(self.start_date),
            "end_date": utc_isoformat(self.end_date),
            "initial_balance": str(self.initial_balance),
            "final_balance": str(self.final_balance),
            "total_return": str(self.total_return),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": str(self.win_rate),
            "profit_factor": str(self.profit_factor),
            "max_drawdown": str(self.max_drawdown),
            "sharpe_ratio": str(self.sharpe_ratio),
            "calmar_ratio": str(self.calmar_ratio),
            "equity_curve": [point.to_dict() for point in self.equity_curve],
            "trades": [trade.to_dict() for trade in self.trades],
        }


# Utility functions for data conversion


def klines_from_binance(
    binance_data: List, symbol: str, timeframe: TimeFrame
) -> List[KlineData]:
    """Convert Binance kline data to KlineData objects"""
    klines = []

    for row in binance_data:
        kline = KlineData(
            symbol=symbol,
            open_time=datetime.fromtimestamp(row[0] / 1000),
            close_time=datetime.fromtimestamp(row[6] / 1000),
            open_price=Decimal(row[1]),
            high_price=Decimal(row[2]),
            low_price=Decimal(row[3]),
            close_price=Decimal(row[4]),
            volume=Decimal(row[5]),
            quote_volume=Decimal(row[7]),
            trades_count=int(row[8]),
            taker_buy_volume=Decimal(row[9]),
            taker_buy_quote_volume=Decimal(row[10]),
            timeframe=timeframe,
        )
        klines.append(kline)

    return klines


def trades_from_binance(binance_data: List, symbol: str) -> List[TradeData]:
    """Convert Binance trade data to TradeData objects"""
    trades = []

    for row in binance_data:
        trade = TradeData(
            symbol=symbol,
            trade_id=int(row["id"]),
            price=Decimal(row["price"]),
            quantity=Decimal(row["qty"]),
            quote_quantity=Decimal(row["quoteQty"]),
            timestamp=datetime.fromtimestamp(row["time"] / 1000),
            is_buyer_maker=row["isBuyerMaker"],
        )
        trades.append(trade)

    return trades


def depth_from_binance(binance_data: Dict, symbol: str) -> DepthData:
    """Convert Binance depth data to DepthData object"""
    bids = [
        DepthLevel(Decimal(price), Decimal(qty)) for price, qty in binance_data["bids"]
    ]
    asks = [
        DepthLevel(Decimal(price), Decimal(qty)) for price, qty in binance_data["asks"]
    ]

    return DepthData(
        symbol=symbol,
        last_update_id=binance_data["lastUpdateId"],
        bids=bids,
        asks=asks,
        timestamp=datetime.now(timezone.utc),
    )


# JSON encoders for serialization


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder for Decimal types"""

    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        elif isinstance(obj, datetime):
            return utc_isoformat(obj)
        return super().default(obj)


def serialize_to_json(data: Any) -> str:
    """Serialize data to JSON string"""
    return json.dumps(data, cls=DecimalEncoder, indent=2)


def deserialize_from_json(json_str: str) -> Any:
    """Deserialize data from JSON string"""
    return json.loads(json_str)
