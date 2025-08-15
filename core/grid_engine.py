"""
Grid Engine - Grid price map, order planner, state transitions

This module handles the core grid trading logic including:
- Grid price level calculation and mapping
- Order planning and placement strategy
- State transitions between grid levels
"""

from typing import List, Dict, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum


class GridDirection(Enum):
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


@dataclass
class GridLevel:
    """Represents a single level in the trading grid"""

    price: Decimal
    level_index: int
    order_id: Optional[str] = None
    filled: bool = False
    side: str = "buy"  # "buy" or "sell"


@dataclass
class GridConfig:
    """Configuration for grid trading strategy"""

    center_price: Decimal
    grid_spacing: Decimal
    num_levels_up: int
    num_levels_down: int
    order_amount: Decimal
    max_position_size: Decimal


class GridEngine:
    """
    Core grid trading engine that manages price levels, order placement,
    and state transitions.
    """

    def __init__(self, config: GridConfig):
        self.config = config
        self.grid_levels: Dict[int, GridLevel] = {}
        self.current_position: Decimal = Decimal("0")
        self.last_price: Optional[Decimal] = None

    def initialize_grid(self) -> List[GridLevel]:
        """Initialize the complete grid structure"""
        self.grid_levels.clear()

        # Create buy levels (below center)
        for i in range(1, self.config.num_levels_down + 1):
            price = self.config.center_price - (self.config.grid_spacing * i)
            level = GridLevel(price=price, level_index=-i, side="buy")
            self.grid_levels[-i] = level

        # Create sell levels (above center)
        for i in range(1, self.config.num_levels_up + 1):
            price = self.config.center_price + (self.config.grid_spacing * i)
            level = GridLevel(price=price, level_index=i, side="sell")
            self.grid_levels[i] = level

        return list(self.grid_levels.values())

    def get_active_orders(self) -> List[GridLevel]:
        """Get all grid levels that should have active orders"""
        return [
            level
            for level in self.grid_levels.values()
            if not level.filled and level.order_id is None
        ]

    def on_fill(
        self, level_index: int, order_id: str, filled_price: Decimal
    ) -> List[GridLevel]:
        """
        Handle order fill and determine next actions

        Returns:
            List of new grid levels to place orders on
        """
        if level_index not in self.grid_levels:
            return []

        level = self.grid_levels[level_index]
        level.filled = True
        level.order_id = order_id

        # Update position
        if level.side == "buy":
            self.current_position += self.config.order_amount
        else:
            self.current_position -= self.config.order_amount

        # Determine new orders to place
        new_orders = []

        if level.side == "buy":
            # After buy fill, place sell order above
            sell_level_index = level_index + 1
            if sell_level_index <= self.config.num_levels_up:
                if sell_level_index in self.grid_levels:
                    sell_level = self.grid_levels[sell_level_index]
                    if not sell_level.filled and sell_level.order_id is None:
                        new_orders.append(sell_level)
        else:
            # After sell fill, place buy order below
            buy_level_index = level_index - 1
            if buy_level_index >= -self.config.num_levels_down:
                if buy_level_index in self.grid_levels:
                    buy_level = self.grid_levels[buy_level_index]
                    if not buy_level.filled and buy_level.order_id is None:
                        new_orders.append(buy_level)

        return new_orders

    def rebalance_grid(
        self, current_price: Decimal
    ) -> Tuple[List[GridLevel], List[GridLevel]]:
        """
        Rebalance grid around current price if needed

        Returns:
            Tuple of (orders_to_cancel, new_orders_to_place)
        """
        self.last_price = current_price

        # Check if current price is too far from center
        price_deviation = abs(current_price - self.config.center_price)
        max_deviation = self.config.grid_spacing * 3  # Arbitrary threshold

        if price_deviation > max_deviation:
            # Recenter grid around current price
            self.config.center_price = current_price
            old_levels = list(self.grid_levels.values())
            new_levels = self.initialize_grid()

            orders_to_cancel = [
                level for level in old_levels if level.order_id and not level.filled
            ]
            return orders_to_cancel, new_levels

        return [], []

    def get_grid_status(self) -> Dict:
        """Get current status of the grid"""
        total_levels = len(self.grid_levels)
        filled_levels = sum(1 for level in self.grid_levels.values() if level.filled)
        active_orders = sum(
            1
            for level in self.grid_levels.values()
            if level.order_id and not level.filled
        )

        return {
            "total_levels": total_levels,
            "filled_levels": filled_levels,
            "active_orders": active_orders,
            "current_position": float(self.current_position),
            "center_price": float(self.config.center_price),
            "last_price": float(self.last_price) if self.last_price else None,
        }
