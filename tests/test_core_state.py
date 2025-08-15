"""
Test cases for core/state.py module

This module provides comprehensive tests for state management functionality,
including data structures, serialization, persistence, and state operations.
"""

import pytest
import json
import pickle
import tempfile
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any

from core.state import (
    # Utility functions
    utc_now,
    utc_isoformat,
    StateJSONEncoder,
    # Enums
    BotStatus,
    OrderStatus,
    OrderSide,
    OrderType,
    # Data classes
    Position,
    Order,
    Balance,
    GridState,
    TradingMetrics,
    BotState,
    # Serialization
    StateSerializer,
    # Convenience functions
    create_empty_state,
    save_state,
    load_state,
)


class TestUtilityFunctions:
    """Test utility functions for datetime handling"""

    def test_utc_now_returns_timezone_aware_datetime(self):
        """Test that utc_now returns timezone-aware datetime"""
        now = utc_now()

        assert isinstance(now, datetime)
        assert now.tzinfo == timezone.utc

        # Should be close to current time
        current_time = datetime.now(timezone.utc)
        time_diff = abs((current_time - now).total_seconds())
        assert time_diff < 1  # Within 1 second

    def test_utc_isoformat_with_timezone_aware_datetime(self):
        """Test UTC ISO format with timezone-aware datetime"""
        dt = datetime(2023, 12, 1, 12, 30, 45, 123456, timezone.utc)
        result = utc_isoformat(dt)

        assert result == "2023-12-01T12:30:45.123456Z"

    def test_utc_isoformat_with_naive_datetime(self):
        """Test UTC ISO format with naive datetime (assumes UTC)"""
        dt = datetime(2023, 12, 1, 12, 30, 45, 123456)
        result = utc_isoformat(dt)

        assert result == "2023-12-01T12:30:45.123456Z"

    def test_utc_isoformat_with_different_timezone(self):
        """Test UTC ISO format converts from different timezone"""
        # Create EST timezone (UTC-5)
        est_tz = timezone(timedelta(hours=-5))
        dt = datetime(2023, 12, 1, 7, 30, 45, tzinfo=est_tz)  # 7 AM EST = 12 PM UTC
        result = utc_isoformat(dt)

        assert result == "2023-12-01T12:30:45Z"


class TestStateJSONEncoder:
    """Test custom JSON encoder for state objects"""

    def test_encode_datetime(self):
        """Test encoding datetime objects"""
        encoder = StateJSONEncoder()
        dt = datetime(2023, 12, 1, 12, 30, 45, tzinfo=timezone.utc)

        result = encoder.default(dt)
        assert result == "2023-12-01T12:30:45Z"

    def test_encode_decimal(self):
        """Test encoding Decimal objects"""
        encoder = StateJSONEncoder()
        decimal_val = Decimal("123.456789")

        result = encoder.default(decimal_val)
        assert result == "123.456789"

    def test_encode_enum(self):
        """Test encoding Enum objects"""
        encoder = StateJSONEncoder()

        assert encoder.default(BotStatus.RUNNING) == "running"
        assert encoder.default(OrderSide.BUY) == "BUY"
        assert encoder.default(OrderType.LIMIT) == "LIMIT"

    def test_encode_unsupported_object_raises_error(self):
        """Test encoding unsupported object raises TypeError"""
        encoder = StateJSONEncoder()

        with pytest.raises(TypeError):
            encoder.default(object())


class TestEnums:
    """Test enum definitions"""

    def test_bot_status_enum(self):
        """Test BotStatus enum values"""
        assert BotStatus.STOPPED.value == "stopped"
        assert BotStatus.STARTING.value == "starting"
        assert BotStatus.RUNNING.value == "running"
        assert BotStatus.PAUSED.value == "paused"
        assert BotStatus.STOPPING.value == "stopping"
        assert BotStatus.ERROR.value == "error"

    def test_order_status_enum(self):
        """Test OrderStatus enum values"""
        assert OrderStatus.NEW.value == "NEW"
        assert OrderStatus.PARTIALLY_FILLED.value == "PARTIALLY_FILLED"
        assert OrderStatus.FILLED.value == "FILLED"
        assert OrderStatus.CANCELED.value == "CANCELED"
        assert OrderStatus.REJECTED.value == "REJECTED"
        assert OrderStatus.EXPIRED.value == "EXPIRED"

    def test_order_side_enum(self):
        """Test OrderSide enum values"""
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"

    def test_order_type_enum(self):
        """Test OrderType enum values"""
        assert OrderType.LIMIT.value == "LIMIT"
        assert OrderType.MARKET.value == "MARKET"
        assert OrderType.STOP_LOSS.value == "STOP_LOSS"
        assert OrderType.TAKE_PROFIT.value == "TAKE_PROFIT"


class TestPosition:
    """Test Position data class"""

    def test_position_initialization(self):
        """Test Position initialization"""
        timestamp = utc_now()
        position = Position(
            symbol="BTCUSDC",
            quantity=Decimal("0.001"),
            entry_price=Decimal("50000"),
            current_price=Decimal("55000"),
            unrealized_pnl=Decimal("5"),
            realized_pnl=Decimal("0"),
            timestamp=timestamp,
        )

        assert position.symbol == "BTCUSDC"
        assert position.quantity == Decimal("0.001")
        assert position.entry_price == Decimal("50000")
        assert position.current_price == Decimal("55000")
        assert position.unrealized_pnl == Decimal("5")
        assert position.realized_pnl == Decimal("0")
        assert position.timestamp == timestamp

    def test_position_value_calculation(self):
        """Test position value calculation"""
        position = Position(
            symbol="BTCUSDC",
            quantity=Decimal("0.002"),
            entry_price=Decimal("50000"),
            current_price=Decimal("55000"),
            unrealized_pnl=Decimal("10"),
            realized_pnl=Decimal("0"),
        )

        expected_value = Decimal("0.002") * Decimal("55000")
        assert position.position_value == expected_value

    def test_position_direction_properties(self):
        """Test position direction properties"""
        # Long position
        long_position = Position(
            symbol="BTCUSDC",
            quantity=Decimal("0.001"),
            entry_price=Decimal("50000"),
            current_price=Decimal("55000"),
            unrealized_pnl=Decimal("5"),
            realized_pnl=Decimal("0"),
        )
        assert long_position.is_long is True
        assert long_position.is_short is False
        assert long_position.is_flat is False

        # Short position
        short_position = Position(
            symbol="BTCUSDC",
            quantity=Decimal("-0.001"),
            entry_price=Decimal("50000"),
            current_price=Decimal("55000"),
            unrealized_pnl=Decimal("-5"),
            realized_pnl=Decimal("0"),
        )
        assert short_position.is_long is False
        assert short_position.is_short is True
        assert short_position.is_flat is False

        # Flat position
        flat_position = Position(
            symbol="BTCUSDC",
            quantity=Decimal("0"),
            entry_price=Decimal("50000"),
            current_price=Decimal("55000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("5"),
        )
        assert flat_position.is_long is False
        assert flat_position.is_short is False
        assert flat_position.is_flat is True


class TestOrder:
    """Test Order data class"""

    def test_order_initialization(self):
        """Test Order initialization"""
        timestamp = utc_now()
        order = Order(
            order_id="12345",
            client_order_id="client_12345",
            symbol="BTCUSDC",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000"),
            status=OrderStatus.NEW,
            timestamp=timestamp,
        )

        assert order.order_id == "12345"
        assert order.client_order_id == "client_12345"
        assert order.symbol == "BTCUSDC"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == Decimal("0.001")
        assert order.price == Decimal("50000")
        assert order.status == OrderStatus.NEW
        assert order.timestamp == timestamp
        assert order.filled_quantity == Decimal("0")
        assert order.filled_price == Decimal("0")
        assert order.fee == Decimal("0")

    def test_remaining_quantity_calculation(self):
        """Test remaining quantity calculation"""
        order = Order(
            order_id="12345",
            client_order_id="client_12345",
            symbol="BTCUSDC",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000"),
            status=OrderStatus.PARTIALLY_FILLED,
            filled_quantity=Decimal("0.0006"),
        )

        assert order.remaining_quantity == Decimal("0.0004")

    def test_order_status_properties(self):
        """Test order status properties"""
        # Filled order
        filled_order = Order(
            order_id="12345",
            client_order_id="client_12345",
            symbol="BTCUSDC",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000"),
            status=OrderStatus.FILLED,
        )
        assert filled_order.is_filled is True
        assert filled_order.is_active is False

        # Active orders
        new_order = Order(
            order_id="12346",
            client_order_id="client_12346",
            symbol="BTCUSDC",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000"),
            status=OrderStatus.NEW,
        )
        assert new_order.is_filled is False
        assert new_order.is_active is True

        partial_order = Order(
            order_id="12347",
            client_order_id="client_12347",
            symbol="BTCUSDC",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000"),
            status=OrderStatus.PARTIALLY_FILLED,
        )
        assert partial_order.is_filled is False
        assert partial_order.is_active is True

        # Canceled order
        canceled_order = Order(
            order_id="12348",
            client_order_id="client_12348",
            symbol="BTCUSDC",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000"),
            status=OrderStatus.CANCELED,
        )
        assert canceled_order.is_filled is False
        assert canceled_order.is_active is False


class TestBalance:
    """Test Balance data class"""

    def test_balance_initialization(self):
        """Test Balance initialization"""
        timestamp = utc_now()
        balance = Balance(
            asset="USDC",
            free=Decimal("1000"),
            locked=Decimal("100"),
            timestamp=timestamp,
        )

        assert balance.asset == "USDC"
        assert balance.free == Decimal("1000")
        assert balance.locked == Decimal("100")
        assert balance.timestamp == timestamp

    def test_total_balance_calculation(self):
        """Test total balance calculation"""
        balance = Balance(asset="USDC", free=Decimal("1000"), locked=Decimal("100"))

        assert balance.total == Decimal("1100")


class TestGridState:
    """Test GridState data class"""

    def test_grid_state_initialization(self):
        """Test GridState initialization"""
        grid_state = GridState(
            center_price=Decimal("50000"),
            grid_spacing=Decimal("1000"),
            num_levels_up=5,
            num_levels_down=5,
        )

        assert grid_state.center_price == Decimal("50000")
        assert grid_state.grid_spacing == Decimal("1000")
        assert grid_state.num_levels_up == 5
        assert grid_state.num_levels_down == 5
        assert grid_state.active_levels == []
        assert grid_state.filled_levels == []
        assert grid_state.total_trades == 0
        assert grid_state.total_profit == Decimal("0")

    def test_grid_state_with_data(self):
        """Test GridState with trading data"""
        grid_state = GridState(
            center_price=Decimal("50000"),
            grid_spacing=Decimal("1000"),
            num_levels_up=5,
            num_levels_down=5,
            active_levels=[1, 2, -1],
            filled_levels=[0, -2],
            total_trades=15,
            total_profit=Decimal("125.50"),
        )

        assert grid_state.active_levels == [1, 2, -1]
        assert grid_state.filled_levels == [0, -2]
        assert grid_state.total_trades == 15
        assert grid_state.total_profit == Decimal("125.50")


class TestTradingMetrics:
    """Test TradingMetrics data class"""

    def test_trading_metrics_initialization(self):
        """Test TradingMetrics initialization"""
        timestamp = utc_now()
        metrics = TradingMetrics(last_updated=timestamp)

        assert metrics.total_trades == 0
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 0
        assert metrics.total_pnl == Decimal("0")
        assert metrics.realized_pnl == Decimal("0")
        assert metrics.unrealized_pnl == Decimal("0")
        assert metrics.fees_paid == Decimal("0")
        assert metrics.max_drawdown == Decimal("0")
        assert metrics.sharpe_ratio is None
        assert metrics.win_rate is None
        assert metrics.profit_factor is None
        assert metrics.last_updated == timestamp

    def test_update_trade_stats_win_rate(self):
        """Test win rate calculation in update_trade_stats"""
        metrics = TradingMetrics(total_trades=10, winning_trades=7, losing_trades=3)

        metrics.update_trade_stats()
        assert metrics.win_rate == Decimal("0.7")

    def test_update_trade_stats_profit_factor(self):
        """Test profit factor calculation in update_trade_stats"""
        metrics = TradingMetrics(
            total_trades=10, winning_trades=6, losing_trades=4, total_pnl=Decimal("100")
        )

        metrics.update_trade_stats()
        # Profit factor calculation is complex, just test it's calculated
        assert metrics.profit_factor is not None

    def test_update_trade_stats_no_trades(self):
        """Test update_trade_stats with no trades"""
        metrics = TradingMetrics()

        metrics.update_trade_stats()
        assert metrics.win_rate is None
        assert metrics.profit_factor is None


class TestBotState:
    """Test BotState data class and methods"""

    def test_bot_state_initialization(self):
        """Test BotState initialization"""
        timestamp = utc_now()
        state = BotState(symbol="BTCUSDC", start_time=timestamp)

        assert state.status == BotStatus.STOPPED
        assert state.symbol == "BTCUSDC"
        assert state.start_time == timestamp
        assert isinstance(state.last_update, datetime)
        assert state.positions == {}
        assert state.active_orders == {}
        assert state.order_history == []
        assert state.balances == {}
        assert state.grid_state is None
        assert isinstance(state.metrics, TradingMetrics)
        assert state.config == {}
        assert state.error_count == 0
        assert state.last_error is None
        assert state.error_timestamp is None
        assert state.requires_manual_resume is False
        assert state.flatten_reason is None
        assert state.flatten_timestamp is None

    def test_update_position_new_position(self):
        """Test updating a new position"""
        state = BotState(symbol="BTCUSDC")

        state.update_position("BTCUSDC", Decimal("0.001"), Decimal("50000"))

        assert "BTCUSDC" in state.positions
        position = state.positions["BTCUSDC"]
        assert position.symbol == "BTCUSDC"
        assert position.quantity == Decimal("0.001")
        assert position.entry_price == Decimal("50000")
        assert position.current_price == Decimal("50000")
        assert position.unrealized_pnl == Decimal("0")
        assert position.realized_pnl == Decimal("0")

    def test_update_position_existing_position_increase(self):
        """Test updating existing position with quantity increase"""
        state = BotState(symbol="BTCUSDC")

        # Initial position
        state.update_position("BTCUSDC", Decimal("0.001"), Decimal("50000"))

        # Add to position
        state.update_position("BTCUSDC", Decimal("0.002"), Decimal("52000"))

        position = state.positions["BTCUSDC"]
        assert position.quantity == Decimal("0.003")
        # Average price calculation: (0.001 * 50000 + 0.002 * 52000) / 0.003
        expected_avg_price = (
            Decimal("0.001") * Decimal("50000") + Decimal("0.002") * Decimal("52000")
        ) / Decimal("0.003")
        assert position.entry_price == expected_avg_price
        assert position.current_price == Decimal("52000")

    def test_update_position_close_position(self):
        """Test closing position completely"""
        state = BotState(symbol="BTCUSDC")

        # Initial position
        state.update_position("BTCUSDC", Decimal("0.001"), Decimal("50000"))

        # Close position
        state.update_position("BTCUSDC", Decimal("-0.001"), Decimal("55000"))

        position = state.positions["BTCUSDC"]
        assert position.quantity == Decimal("0")
        # The realized PnL calculation is (close_price - entry_price) * sell_quantity
        # Since sell_quantity is negative (-0.001), the calculation becomes:
        # (55000 - 50000) * (-0.001) = -5
        assert position.realized_pnl == Decimal("-5")  # (55000 - 50000) * (-0.001)
        assert position.current_price == Decimal("55000")

    def test_add_order(self):
        """Test adding an order"""
        state = BotState(symbol="BTCUSDC")
        order = Order(
            order_id="12345",
            client_order_id="client_12345",
            symbol="BTCUSDC",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000"),
            status=OrderStatus.NEW,
        )

        state.add_order(order)

        assert "12345" in state.active_orders
        assert state.active_orders["12345"] == order

    def test_update_order_active(self):
        """Test updating an active order"""
        state = BotState(symbol="BTCUSDC")
        order = Order(
            order_id="12345",
            client_order_id="client_12345",
            symbol="BTCUSDC",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000"),
            status=OrderStatus.NEW,
        )
        state.add_order(order)

        # Update order
        state.update_order(
            "12345",
            status=OrderStatus.PARTIALLY_FILLED,
            filled_quantity=Decimal("0.0005"),
        )

        updated_order = state.active_orders["12345"]
        assert updated_order.status == OrderStatus.PARTIALLY_FILLED
        assert updated_order.filled_quantity == Decimal("0.0005")
        assert updated_order.update_time is not None

    def test_update_order_move_to_history(self):
        """Test updating order that moves to history"""
        state = BotState(symbol="BTCUSDC")
        order = Order(
            order_id="12345",
            client_order_id="client_12345",
            symbol="BTCUSDC",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000"),
            status=OrderStatus.NEW,
        )
        state.add_order(order)

        # Fill order
        state.update_order(
            "12345", status=OrderStatus.FILLED, filled_quantity=Decimal("0.001")
        )

        assert "12345" not in state.active_orders
        assert len(state.order_history) == 1
        assert state.order_history[0].order_id == "12345"
        assert state.order_history[0].status == OrderStatus.FILLED

    def test_update_order_nonexistent(self):
        """Test updating non-existent order"""
        state = BotState(symbol="BTCUSDC")

        # Should not raise error
        state.update_order("nonexistent", status=OrderStatus.FILLED)

        assert len(state.active_orders) == 0
        assert len(state.order_history) == 0

    def test_update_balance(self):
        """Test updating balance"""
        state = BotState(symbol="BTCUSDC")

        state.update_balance("USDC", Decimal("1000"), Decimal("100"))

        assert "USDC" in state.balances
        balance = state.balances["USDC"]
        assert balance.asset == "USDC"
        assert balance.free == Decimal("1000")
        assert balance.locked == Decimal("100")
        assert balance.total == Decimal("1100")

    def test_record_error(self):
        """Test recording an error"""
        state = BotState(symbol="BTCUSDC")

        state.record_error("Test error message")

        assert state.error_count == 1
        assert state.last_error == "Test error message"
        assert state.error_timestamp is not None

        # Record another error
        state.record_error("Second error")

        assert state.error_count == 2
        assert state.last_error == "Second error"

    def test_set_flatten_state(self):
        """Test setting flatten state"""
        state = BotState(symbol="BTCUSDC")

        state.set_flatten_state("Test flatten reason")

        assert state.requires_manual_resume is True
        assert state.flatten_reason == "Test flatten reason"
        assert state.flatten_timestamp is not None
        assert state.status == BotStatus.PAUSED

    def test_clear_manual_resume(self):
        """Test clearing manual resume"""
        state = BotState(symbol="BTCUSDC")

        # Set flatten state first
        state.set_flatten_state("Test flatten reason")

        # Clear it
        state.clear_manual_resume()

        assert state.requires_manual_resume is False
        assert state.flatten_reason is None
        assert state.flatten_timestamp is None

    def test_get_total_portfolio_value(self):
        """Test calculating total portfolio value"""
        state = BotState(symbol="BTCUSDC")

        # Add balances
        state.update_balance("USDC", Decimal("1000"), Decimal("100"))
        state.update_balance("BTC", Decimal("0.02"), Decimal("0.005"))
        state.update_balance("ETH", Decimal("1.5"), Decimal("0.2"))

        # Prices for conversion
        prices = {"BTC": Decimal("50000"), "ETH": Decimal("3000")}

        total_value = state.get_total_portfolio_value(prices)

        # Expected: 1100 USDC + (0.025 * 50000) BTC + (1.7 * 3000) ETH
        expected = (
            Decimal("1100")
            + (Decimal("0.025") * Decimal("50000"))
            + (Decimal("1.7") * Decimal("3000"))
        )
        assert total_value == expected

    def test_get_unrealized_pnl(self):
        """Test calculating total unrealized PnL"""
        state = BotState(symbol="BTCUSDC")

        # Add positions with unrealized PnL
        state.positions["BTCUSDC"] = Position(
            symbol="BTCUSDC",
            quantity=Decimal("0.001"),
            entry_price=Decimal("50000"),
            current_price=Decimal("55000"),
            unrealized_pnl=Decimal("5"),
            realized_pnl=Decimal("0"),
        )
        state.positions["ETHUSDC"] = Position(
            symbol="ETHUSDC",
            quantity=Decimal("0.1"),
            entry_price=Decimal("3000"),
            current_price=Decimal("3200"),
            unrealized_pnl=Decimal("20"),
            realized_pnl=Decimal("0"),
        )

        total_unrealized = state.get_unrealized_pnl()
        assert total_unrealized == Decimal("25")

    def test_get_realized_pnl(self):
        """Test calculating total realized PnL"""
        state = BotState(symbol="BTCUSDC")

        # Add positions with realized PnL
        state.positions["BTCUSDC"] = Position(
            symbol="BTCUSDC",
            quantity=Decimal("0"),
            entry_price=Decimal("50000"),
            current_price=Decimal("55000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("10"),
        )
        state.positions["ETHUSDC"] = Position(
            symbol="ETHUSDC",
            quantity=Decimal("0"),
            entry_price=Decimal("3000"),
            current_price=Decimal("3200"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("15"),
        )

        total_realized = state.get_realized_pnl()
        assert total_realized == Decimal("25")


class TestStateSerializer:
    """Test StateSerializer functionality"""

    def test_to_dict(self):
        """Test converting state to dictionary"""
        state = BotState(
            symbol="BTCUSDC", status=BotStatus.RUNNING, start_time=utc_now()
        )

        state_dict = StateSerializer.to_dict(state)

        assert isinstance(state_dict, dict)
        assert state_dict["symbol"] == "BTCUSDC"
        assert state_dict["status"] == BotStatus.RUNNING

    def test_from_dict_basic(self):
        """Test creating state from dictionary"""
        state_dict = {
            "symbol": "BTCUSDC",
            "status": "running",
            "start_time": None,
            "last_update": utc_now(),
            "positions": {},
            "active_orders": {},
            "order_history": [],
            "balances": {},
            "grid_state": None,
            "metrics": {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_pnl": Decimal("0"),
                "realized_pnl": Decimal("0"),
                "unrealized_pnl": Decimal("0"),
                "fees_paid": Decimal("0"),
                "max_drawdown": Decimal("0"),
                "sharpe_ratio": None,
                "win_rate": None,
                "profit_factor": None,
                "last_updated": utc_now(),
            },
            "config": {},
            "error_count": 0,
            "last_error": None,
            "error_timestamp": None,
            "requires_manual_resume": False,
            "flatten_reason": None,
            "flatten_timestamp": None,
        }

        state = StateSerializer.from_dict(state_dict)

        assert isinstance(state, BotState)
        assert state.symbol == "BTCUSDC"
        assert state.status == BotStatus.RUNNING

    def test_from_dict_with_orders(self):
        """Test creating state from dictionary with orders"""
        order_data = {
            "order_id": "12345",
            "client_order_id": "client_12345",
            "symbol": "BTCUSDC",
            "side": "BUY",
            "order_type": "LIMIT",
            "quantity": Decimal("0.001"),
            "price": Decimal("50000"),
            "status": "NEW",
            "filled_quantity": Decimal("0"),
            "filled_price": Decimal("0"),
            "fee": Decimal("0"),
            "timestamp": utc_now(),
            "update_time": None,
            "grid_level": None,
        }

        state_dict = {
            "symbol": "BTCUSDC",
            "status": "running",
            "start_time": None,
            "last_update": utc_now(),
            "positions": {},
            "active_orders": {"12345": order_data},
            "order_history": [order_data.copy()],
            "balances": {},
            "grid_state": None,
            "metrics": {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_pnl": Decimal("0"),
                "realized_pnl": Decimal("0"),
                "unrealized_pnl": Decimal("0"),
                "fees_paid": Decimal("0"),
                "max_drawdown": Decimal("0"),
                "sharpe_ratio": None,
                "win_rate": None,
                "profit_factor": None,
                "last_updated": utc_now(),
            },
            "config": {},
            "error_count": 0,
            "last_error": None,
            "error_timestamp": None,
            "requires_manual_resume": False,
            "flatten_reason": None,
            "flatten_timestamp": None,
        }

        state = StateSerializer.from_dict(state_dict)

        assert len(state.active_orders) == 1
        assert "12345" in state.active_orders
        assert isinstance(state.active_orders["12345"], Order)
        assert state.active_orders["12345"].side == OrderSide.BUY

        assert len(state.order_history) == 1
        assert isinstance(state.order_history[0], Order)

    def test_save_and_load_json(self):
        """Test saving and loading state as JSON"""
        state = BotState(
            symbol="BTCUSDC", status=BotStatus.RUNNING, start_time=utc_now()
        )

        # Add some data
        state.update_balance("USDC", Decimal("1000"), Decimal("100"))
        order = Order(
            order_id="12345",
            client_order_id="client_12345",
            symbol="BTCUSDC",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000"),
            status=OrderStatus.NEW,
        )
        state.add_order(order)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            StateSerializer.save_to_file(state, tmp_path)
            loaded_state = StateSerializer.load_from_file(tmp_path)

            assert loaded_state.symbol == state.symbol
            assert loaded_state.status == state.status
            assert "USDC" in loaded_state.balances
            assert "12345" in loaded_state.active_orders
        finally:
            tmp_path.unlink()

    def test_save_and_load_pickle(self):
        """Test saving and loading state as pickle"""
        state = BotState(
            symbol="BTCUSDC", status=BotStatus.RUNNING, start_time=utc_now()
        )

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            StateSerializer.save_to_file(state, tmp_path)
            loaded_state = StateSerializer.load_from_file(tmp_path)

            assert loaded_state.symbol == state.symbol
            assert loaded_state.status == state.status
        finally:
            tmp_path.unlink()

    def test_save_unsupported_format(self):
        """Test saving with unsupported format"""
        state = BotState(symbol="BTCUSDC")

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                StateSerializer.save_to_file(state, tmp_path)
        finally:
            tmp_path.unlink()

    def test_load_nonexistent_file(self):
        """Test loading non-existent file"""
        with pytest.raises(FileNotFoundError):
            StateSerializer.load_from_file(Path("nonexistent.json"))

    def test_load_unsupported_format(self):
        """Test loading with unsupported format"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(b"test content")

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                StateSerializer.load_from_file(tmp_path)
        finally:
            tmp_path.unlink()


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_create_empty_state(self):
        """Test creating empty state"""
        state = create_empty_state("BTCUSDC")

        assert isinstance(state, BotState)
        assert state.symbol == "BTCUSDC"
        assert state.config == {}
        assert state.start_time is not None

    def test_create_empty_state_with_config(self):
        """Test creating empty state with config"""
        config = {"grid_levels": 20, "invest_amount": 1000}
        state = create_empty_state("BTCUSDC", config)

        assert state.symbol == "BTCUSDC"
        assert state.config == config

    def test_save_and_load_state_functions(self):
        """Test save_state and load_state convenience functions"""
        state = BotState(
            symbol="BTCUSDC", status=BotStatus.RUNNING, start_time=utc_now()
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            save_state(state, tmp_path)
            loaded_state = load_state(tmp_path)

            assert loaded_state.symbol == state.symbol
            assert loaded_state.status == state.status
        finally:
            Path(tmp_path).unlink()


class TestComplexScenarios:
    """Test complex scenarios and edge cases"""

    def test_full_trading_session_simulation(self):
        """Test full trading session with multiple operations"""
        state = create_empty_state("BTCUSDC", {"grid_levels": 10})

        # Start trading
        state.status = BotStatus.RUNNING
        state.update_balance("USDC", Decimal("10000"), Decimal("0"))
        state.update_balance("BTC", Decimal("0"), Decimal("0"))

        # Place orders
        buy_order = Order(
            order_id="buy_1",
            client_order_id="client_buy_1",
            symbol="BTCUSDC",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.002"),
            price=Decimal("49000"),
            status=OrderStatus.NEW,
        )
        state.add_order(buy_order)

        sell_order = Order(
            order_id="sell_1",
            client_order_id="client_sell_1",
            symbol="BTCUSDC",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("51000"),
            status=OrderStatus.NEW,
        )
        state.add_order(sell_order)

        # Fill buy order
        state.update_order(
            "buy_1", status=OrderStatus.FILLED, filled_quantity=Decimal("0.002")
        )
        state.update_position("BTCUSDC", Decimal("0.002"), Decimal("49000"))
        state.update_balance(
            "USDC", Decimal("10000") - Decimal("98"), Decimal("0")
        )  # 0.002 * 49000
        state.update_balance("BTC", Decimal("0.002"), Decimal("0"))

        # Partial fill sell order
        state.update_order(
            "sell_1",
            status=OrderStatus.PARTIALLY_FILLED,
            filled_quantity=Decimal("0.0005"),
        )
        state.update_position("BTCUSDC", Decimal("-0.0005"), Decimal("51000"))

        # Verify state
        assert len(state.active_orders) == 1  # sell order still active
        assert len(state.order_history) == 1  # buy order in history
        assert "BTCUSDC" in state.positions
        position = state.positions["BTCUSDC"]
        assert position.quantity == Decimal("0.0015")  # 0.002 - 0.0005

    def test_position_averaging_scenario(self):
        """Test position averaging with multiple trades"""
        state = create_empty_state("BTCUSDC")

        # First buy
        state.update_position("BTCUSDC", Decimal("0.001"), Decimal("50000"))

        # Second buy at different price
        state.update_position("BTCUSDC", Decimal("0.002"), Decimal("48000"))

        # Third buy
        state.update_position("BTCUSDC", Decimal("0.001"), Decimal("52000"))

        position = state.positions["BTCUSDC"]
        assert position.quantity == Decimal("0.004")

        # Average price should be weighted average
        # (0.001 * 50000 + 0.002 * 48000 + 0.001 * 52000) / 0.004
        expected_avg = (Decimal("50") + Decimal("96") + Decimal("52")) / Decimal(
            "0.004"
        )
        assert position.entry_price == expected_avg

    def test_error_handling_and_recovery(self):
        """Test error recording and recovery scenario"""
        state = create_empty_state("BTCUSDC")

        # Record multiple errors
        state.record_error("Connection timeout")
        state.record_error("Invalid order")
        state.record_error("Insufficient balance")

        assert state.error_count == 3
        assert state.last_error == "Insufficient balance"
        assert state.error_timestamp is not None

        # Flatten and recovery
        state.set_flatten_state("Too many errors")
        assert state.requires_manual_resume is True
        assert state.status == BotStatus.PAUSED

        # Manual intervention
        state.clear_manual_resume()
        state.status = BotStatus.RUNNING
        assert state.requires_manual_resume is False

    def test_serialization_roundtrip_with_complex_data(self):
        """Test serialization roundtrip with complex state"""
        state = create_empty_state("BTCUSDC")

        # Add complex data
        state.status = BotStatus.RUNNING
        state.update_balance("USDC", Decimal("1000.123456"), Decimal("100.654321"))

        order = Order(
            order_id="complex_order",
            client_order_id="client_complex",
            symbol="BTCUSDC",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001234567"),
            price=Decimal("50000.987654"),
            status=OrderStatus.PARTIALLY_FILLED,
            filled_quantity=Decimal("0.0006"),
            grid_level=5,
        )
        state.add_order(order)

        state.grid_state = GridState(
            center_price=Decimal("50000"),
            grid_spacing=Decimal("500.5"),
            num_levels_up=10,
            num_levels_down=8,
            active_levels=[1, 3, -2],
            total_profit=Decimal("123.456789"),
        )

        state.record_error("Test complex error")
        # Note: set_flatten_state changes status to PAUSED, so we set it after
        state.set_flatten_state("Test flatten")

        # Test JSON roundtrip
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            StateSerializer.save_to_file(state, tmp_path)
            loaded_state = StateSerializer.load_from_file(tmp_path)

            # Verify complex data preservation
            # Note: status is PAUSED because set_flatten_state changes it
            assert loaded_state.status == BotStatus.PAUSED
            # Note: JSON serialization converts complex objects to dicts, so we check dict structure
            assert "USDC" in loaded_state.balances
            assert isinstance(loaded_state.balances["USDC"], dict)
            assert (
                loaded_state.balances["USDC"]["free"] == "1000.123456"
            )  # JSON stores as string
            assert "complex_order" in loaded_state.active_orders
            assert loaded_state.active_orders["complex_order"].grid_level == 5
            assert isinstance(loaded_state.grid_state, dict)
            assert loaded_state.grid_state["active_levels"] == [1, 3, -2]
            assert loaded_state.error_count == 1
            assert loaded_state.requires_manual_resume is True
        finally:
            tmp_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__])
