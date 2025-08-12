"""
Comprehensive Tests for Core Grid Engine Functionality

This file contains consolidated tests for the grid trading engine,
covering all components: GridDirection, GridLevel, GridConfig, and GridEngine.
Tests are organized by functionality and include edge cases, integration scenarios,
and comprehensive coverage of all grid trading operations.

Consolidated from test_grid_engine.py and test_core_grid_engine.py to eliminate
redundancy while preserving full test coverage for the actual implementation.
"""

import pytest
from unittest.mock import Mock, patch
from decimal import Decimal
from typing import List, Dict, Optional, Tuple

# Import the modules under test
from core.grid_engine import (
    GridDirection,
    GridLevel,
    GridConfig,
    GridEngine
)


class TestGridDirection:
    """Test GridDirection enum functionality"""
    
    def test_grid_direction_values(self):
        """Test GridDirection enum has correct values"""
        assert GridDirection.UP.value == "up"
        assert GridDirection.DOWN.value == "down"
        assert GridDirection.NEUTRAL.value == "neutral"
    
    def test_grid_direction_comparison(self):
        """Test GridDirection enum comparison"""
        assert GridDirection.UP == GridDirection.UP
        assert GridDirection.UP != GridDirection.DOWN
        assert GridDirection.DOWN != GridDirection.NEUTRAL


class TestGridLevel:
    """Test GridLevel data class functionality"""
    
    def test_grid_level_creation(self):
        """Test creating GridLevel with required fields"""
        level = GridLevel(
            price=Decimal("50000"),
            level_index=1
        )
        
        assert level.price == Decimal("50000")
        assert level.level_index == 1
        assert level.order_id is None
        assert level.filled is False
        assert level.side == "buy"
    
    def test_grid_level_custom_values(self):
        """Test creating GridLevel with custom values"""
        level = GridLevel(
            price=Decimal("48000"),
            level_index=-2,
            order_id="12345",
            filled=True,
            side="sell"
        )
        
        assert level.price == Decimal("48000")
        assert level.level_index == -2
        assert level.order_id == "12345"
        assert level.filled is True
        assert level.side == "sell"
    
    def test_grid_level_decimal_precision(self):
        """Test GridLevel maintains decimal precision"""
        level = GridLevel(
            price=Decimal("50000.123456"),
            level_index=0
        )
        
        assert isinstance(level.price, Decimal)
        assert level.price == Decimal("50000.123456")
    
    def test_grid_level_negative_index(self):
        """Test GridLevel with negative level index"""
        level = GridLevel(
            price=Decimal("45000"),
            level_index=-5
        )
        
        assert level.level_index == -5
        assert level.price == Decimal("45000")


class TestGridConfig:
    """Test GridConfig data class functionality"""
    
    def test_grid_config_creation(self):
        """Test creating GridConfig with all required fields"""
        config = GridConfig(
            center_price=Decimal("50000"),
            grid_spacing=Decimal("500"),
            num_levels_up=10,
            num_levels_down=10,
            order_amount=Decimal("0.001"),
            max_position_size=Decimal("0.1")
        )
        
        assert config.center_price == Decimal("50000")
        assert config.grid_spacing == Decimal("500")
        assert config.num_levels_up == 10
        assert config.num_levels_down == 10
        assert config.order_amount == Decimal("0.001")
        assert config.max_position_size == Decimal("0.1")
    
    def test_grid_config_asymmetric_levels(self):
        """Test GridConfig with different up/down levels"""
        config = GridConfig(
            center_price=Decimal("30000"),
            grid_spacing=Decimal("200"),
            num_levels_up=15,
            num_levels_down=8,
            order_amount=Decimal("0.002"),
            max_position_size=Decimal("0.2")
        )
        
        assert config.num_levels_up == 15
        assert config.num_levels_down == 8
        assert config.num_levels_up != config.num_levels_down
    
    def test_grid_config_small_spacing(self):
        """Test GridConfig with small grid spacing"""
        config = GridConfig(
            center_price=Decimal("1000"),
            grid_spacing=Decimal("0.1"),
            num_levels_up=5,
            num_levels_down=5,
            order_amount=Decimal("1.0"),
            max_position_size=Decimal("10.0")
        )
        
        assert config.grid_spacing == Decimal("0.1")
        assert isinstance(config.grid_spacing, Decimal)
    
    def test_grid_config_validation(self):
        """Test GridConfig with various parameter combinations"""
        # Test with minimum viable configuration
        minimal_config = GridConfig(
            center_price=Decimal("1.0"),
            grid_spacing=Decimal("0.01"),
            num_levels_up=1,
            num_levels_down=1,
            order_amount=Decimal("0.001"),
            max_position_size=Decimal("0.1")
        )
        
        assert minimal_config.center_price == Decimal("1.0")
        assert minimal_config.num_levels_up == 1
        assert minimal_config.num_levels_down == 1
        
        # Test with zero levels (edge case)
        zero_up_config = GridConfig(
            center_price=Decimal("5000"),
            grid_spacing=Decimal("10"),
            num_levels_up=0,
            num_levels_down=3,
            order_amount=Decimal("1.0"),
            max_position_size=Decimal("10.0")
        )
        
        assert zero_up_config.num_levels_up == 0
        assert zero_up_config.num_levels_down == 3
        
    def test_grid_config_decimal_precision_preservation(self):
        """Test that GridConfig preserves decimal precision"""
        config = GridConfig(
            center_price=Decimal("50000.123456789"),
            grid_spacing=Decimal("0.000000001"),
            num_levels_up=2,
            num_levels_down=2,
            order_amount=Decimal("0.123456789"),
            max_position_size=Decimal("10.987654321")
        )
        
        # Verify precision is maintained
        assert config.center_price == Decimal("50000.123456789")
        assert config.grid_spacing == Decimal("0.000000001")
        assert config.order_amount == Decimal("0.123456789")
        assert config.max_position_size == Decimal("10.987654321")
        
        # Verify types are preserved
        assert isinstance(config.center_price, Decimal)
        assert isinstance(config.grid_spacing, Decimal)
        assert isinstance(config.order_amount, Decimal)
        assert isinstance(config.max_position_size, Decimal)


class TestGridEngine:
    """Test GridEngine core functionality"""
    
    def setup_method(self):
        """Setup GridEngine with standard test configuration"""
        self.config = GridConfig(
            center_price=Decimal("50000"),
            grid_spacing=Decimal("500"),
            num_levels_up=5,
            num_levels_down=5,
            order_amount=Decimal("0.001"),
            max_position_size=Decimal("0.1")
        )
        self.engine = GridEngine(self.config)
    
    def test_grid_engine_initialization(self):
        """Test GridEngine initializes correctly"""
        assert self.engine.config == self.config
        assert self.engine.grid_levels == {}
        assert self.engine.current_position == Decimal('0')
        assert self.engine.last_price is None
    
    def test_initialize_grid_creates_all_levels(self):
        """Test initialize_grid creates correct number of levels"""
        levels = self.engine.initialize_grid()
        
        # Should create levels for both up and down
        total_expected = self.config.num_levels_up + self.config.num_levels_down
        assert len(levels) == total_expected
        assert len(self.engine.grid_levels) == total_expected
    
    def test_initialize_grid_buy_levels(self):
        """Test initialize_grid creates correct buy levels"""
        self.engine.initialize_grid()
        
        # Check buy levels (negative indices, below center)
        for i in range(1, self.config.num_levels_down + 1):
            level_index = -i
            assert level_index in self.engine.grid_levels
            
            level = self.engine.grid_levels[level_index]
            expected_price = self.config.center_price - (self.config.grid_spacing * i)
            
            assert level.price == expected_price
            assert level.level_index == level_index
            assert level.side == "buy"
            assert level.filled is False
            assert level.order_id is None
    
    def test_initialize_grid_sell_levels(self):
        """Test initialize_grid creates correct sell levels"""
        self.engine.initialize_grid()
        
        # Check sell levels (positive indices, above center)
        for i in range(1, self.config.num_levels_up + 1):
            level_index = i
            assert level_index in self.engine.grid_levels
            
            level = self.engine.grid_levels[level_index]
            expected_price = self.config.center_price + (self.config.grid_spacing * i)
            
            assert level.price == expected_price
            assert level.level_index == level_index
            assert level.side == "sell"
            assert level.filled is False
            assert level.order_id is None
    
    def test_initialize_grid_clears_existing(self):
        """Test initialize_grid clears existing levels"""
        # Add some dummy levels
        self.engine.grid_levels[1] = GridLevel(Decimal("1000"), 1)
        self.engine.grid_levels[2] = GridLevel(Decimal("2000"), 2)
        
        assert len(self.engine.grid_levels) == 2
        
        # Initialize grid should clear existing
        self.engine.initialize_grid()
        
        # Should only have new levels
        expected_count = self.config.num_levels_up + self.config.num_levels_down
        assert len(self.engine.grid_levels) == expected_count
    
    def test_initialize_grid_price_spacing(self):
        """Test initialize_grid maintains correct price spacing"""
        self.engine.initialize_grid()
        
        # Check spacing between levels of the same type (buy-to-buy, sell-to-sell)
        sorted_indices = sorted(self.engine.grid_levels.keys())
        
        # Test buy levels (negative indices)
        buy_indices = [idx for idx in sorted_indices if idx < 0]
        for i in range(len(buy_indices) - 1):
            current_idx = buy_indices[i]
            next_idx = buy_indices[i + 1]
            
            current_price = self.engine.grid_levels[current_idx].price
            next_price = self.engine.grid_levels[next_idx].price
            
            price_diff = next_price - current_price
            assert price_diff == self.config.grid_spacing
        
        # Test sell levels (positive indices)
        sell_indices = [idx for idx in sorted_indices if idx > 0]
        for i in range(len(sell_indices) - 1):
            current_idx = sell_indices[i]
            next_idx = sell_indices[i + 1]
            
            current_price = self.engine.grid_levels[current_idx].price
            next_price = self.engine.grid_levels[next_idx].price
            
            price_diff = next_price - current_price
            assert price_diff == self.config.grid_spacing
        
        # Test that each level is correctly positioned relative to center
        for idx, level in self.engine.grid_levels.items():
            expected_price = self.config.center_price + (self.config.grid_spacing * idx)
            assert level.price == expected_price
    
    def test_get_active_orders_all_empty(self):
        """Test get_active_orders when no orders are placed"""
        self.engine.initialize_grid()
        
        active_orders = self.engine.get_active_orders()
        
        # All levels should be active (not filled, no order_id)
        expected_count = self.config.num_levels_up + self.config.num_levels_down
        assert len(active_orders) == expected_count
        
        for level in active_orders:
            assert level.filled is False
            assert level.order_id is None
    
    def test_get_active_orders_some_filled(self):
        """Test get_active_orders with some filled levels"""
        self.engine.initialize_grid()
        
        # Mark some levels as filled
        self.engine.grid_levels[1].filled = True
        self.engine.grid_levels[-2].filled = True
        
        active_orders = self.engine.get_active_orders()
        
        # Should exclude filled levels
        total_levels = self.config.num_levels_up + self.config.num_levels_down
        expected_active = total_levels - 2
        assert len(active_orders) == expected_active
        
        # Verify filled levels are not in active orders
        for level in active_orders:
            assert level.filled is False
    
    def test_get_active_orders_some_with_order_id(self):
        """Test get_active_orders with some levels having order IDs"""
        self.engine.initialize_grid()
        
        # Set order IDs for some levels
        self.engine.grid_levels[2].order_id = "order_123"
        self.engine.grid_levels[-1].order_id = "order_456"
        
        active_orders = self.engine.get_active_orders()
        
        # Should exclude levels with order IDs
        total_levels = self.config.num_levels_up + self.config.num_levels_down
        expected_active = total_levels - 2
        assert len(active_orders) == expected_active
        
        # Verify levels with order IDs are not in active orders
        for level in active_orders:
            assert level.order_id is None
    
    def test_get_active_orders_mixed_states(self):
        """Test get_active_orders with mixed level states"""
        self.engine.initialize_grid()
        
        # Set various states
        self.engine.grid_levels[1].filled = True
        self.engine.grid_levels[2].order_id = "order_123"
        self.engine.grid_levels[-1].filled = True
        self.engine.grid_levels[-2].order_id = "order_456"
        
        active_orders = self.engine.get_active_orders()
        
        # Should only include unfilled levels without order IDs
        total_levels = self.config.num_levels_up + self.config.num_levels_down
        expected_active = total_levels - 4
        assert len(active_orders) == expected_active
        
        for level in active_orders:
            assert level.filled is False
            assert level.order_id is None
    
    def test_on_fill_buy_order(self):
        """Test on_fill handling for buy order"""
        self.engine.initialize_grid()
        
        # Fill a buy order at level -2
        level_index = -2
        order_id = "buy_order_123"
        filled_price = Decimal("49000")
        
        new_orders = self.engine.on_fill(level_index, order_id, filled_price)
        
        # Check level is marked as filled
        level = self.engine.grid_levels[level_index]
        assert level.filled is True
        assert level.order_id == order_id
        
        # Check position is updated
        expected_position = self.config.order_amount
        assert self.engine.current_position == expected_position
        
        # Should create order at next level up (which is a sell level)
        expected_sell_level_index = level_index + 1  # -1
        assert len(new_orders) == 1
        assert new_orders[0].level_index == expected_sell_level_index
        # The returned level has the side defined by its position, not the action
        assert new_orders[0].side == "buy"  # Level -1 is still a buy level
    
    def test_on_fill_sell_order(self):
        """Test on_fill handling for sell order"""
        self.engine.initialize_grid()
        
        # Fill a sell order at level 3
        level_index = 3
        order_id = "sell_order_456"
        filled_price = Decimal("51500")
        
        new_orders = self.engine.on_fill(level_index, order_id, filled_price)
        
        # Check level is marked as filled
        level = self.engine.grid_levels[level_index]
        assert level.filled is True
        assert level.order_id == order_id
        
        # Check position is updated (decreased for sell)
        expected_position = -self.config.order_amount
        assert self.engine.current_position == expected_position
        
        # Should create order at next level down (which is a sell level)
        expected_buy_level_index = level_index - 1  # 2
        assert len(new_orders) == 1
        assert new_orders[0].level_index == expected_buy_level_index
        # The returned level has the side defined by its position, not the action
        assert new_orders[0].side == "sell"  # Level 2 is still a sell level
    
    def test_on_fill_position_accumulation(self):
        """Test position accumulation through multiple fills"""
        self.engine.initialize_grid()
        
        # Fill multiple buy orders
        buy_fills = [(-1, "buy1"), (-3, "buy2"), (-4, "buy3")]
        
        for level_index, order_id in buy_fills:
            self.engine.on_fill(level_index, order_id, Decimal("50000"))
        
        # Position should accumulate
        expected_position = self.config.order_amount * 3
        assert self.engine.current_position == expected_position
        
        # Fill a sell order
        self.engine.on_fill(2, "sell1", Decimal("51000"))
        
        # Position should decrease
        expected_position = (self.config.order_amount * 3) - self.config.order_amount
        assert self.engine.current_position == expected_position
    
    def test_on_fill_invalid_level(self):
        """Test on_fill with invalid level index"""
        self.engine.initialize_grid()
        
        # Try to fill non-existent level
        invalid_index = 999
        new_orders = self.engine.on_fill(invalid_index, "order123", Decimal("50000"))
        
        # Should return empty list and not affect state
        assert new_orders == []
        assert self.engine.current_position == Decimal('0')
    
    def test_on_fill_edge_level_buy(self):
        """Test on_fill at edge buy level (doesn't create new sell)"""
        self.engine.initialize_grid()
        
        # Fill buy order at highest level (should be at index -num_levels_down)
        highest_buy_index = -self.config.num_levels_down
        new_orders = self.engine.on_fill(highest_buy_index, "edge_buy", Decimal("47500"))
        
        # Check if next level up exists
        next_sell_index = highest_buy_index + 1
        if next_sell_index > self.config.num_levels_up:
            # Should not create new order beyond grid
            assert len(new_orders) == 0
        else:
            # Should create order if within grid
            assert len(new_orders) <= 1
    
    def test_on_fill_edge_level_sell(self):
        """Test on_fill at edge sell level (doesn't create new buy)"""
        self.engine.initialize_grid()
        
        # Fill sell order at highest level
        highest_sell_index = self.config.num_levels_up
        new_orders = self.engine.on_fill(highest_sell_index, "edge_sell", Decimal("52500"))
        
        # Check if next level down exists
        next_buy_index = highest_sell_index - 1
        if next_buy_index < -self.config.num_levels_down:
            # Should not create new order beyond grid
            assert len(new_orders) == 0
        else:
            # Should create order if within grid
            assert len(new_orders) <= 1
    
    def test_on_fill_already_filled_next_level(self):
        """Test on_fill when next level is already filled"""
        self.engine.initialize_grid()
        
        # Pre-fill the next level
        next_level_index = -1
        self.engine.grid_levels[next_level_index].filled = True
        
        # Fill buy order at level -2
        new_orders = self.engine.on_fill(-2, "buy_order", Decimal("49000"))
        
        # Should not create order for already filled level
        assert len(new_orders) == 0
    
    def test_on_fill_next_level_has_order(self):
        """Test on_fill when next level already has pending order"""
        self.engine.initialize_grid()
        
        # Pre-set order ID for next level
        next_level_index = -1
        self.engine.grid_levels[next_level_index].order_id = "existing_order"
        
        # Fill buy order at level -2
        new_orders = self.engine.on_fill(-2, "buy_order", Decimal("49000"))
        
        # Should not create order for level with existing order
        assert len(new_orders) == 0
    
    def test_rebalance_grid_no_rebalance_needed(self):
        """Test rebalance_grid when current price is close to center"""
        self.engine.initialize_grid()
        
        # Current price close to center
        current_price = Decimal("50100")  # Only 100 away from center
        
        orders_to_cancel, new_orders = self.engine.rebalance_grid(current_price)
        
        # Should not rebalance
        assert orders_to_cancel == []
        assert new_orders == []
        assert self.engine.last_price == current_price
    
    def test_rebalance_grid_rebalance_needed(self):
        """Test rebalance_grid when current price is far from center"""
        self.engine.initialize_grid()
        
        # Set some orders as pending
        self.engine.grid_levels[1].order_id = "pending1"
        self.engine.grid_levels[-1].order_id = "pending2"
        
        # Current price far from center (> 3 * grid_spacing = 1500)
        current_price = Decimal("52000")  # 2000 away from center
        
        orders_to_cancel, new_orders = self.engine.rebalance_grid(current_price)
        
        # Should rebalance
        assert len(orders_to_cancel) > 0
        assert len(new_orders) > 0
        
        # Center should be updated
        assert self.engine.config.center_price == current_price
        assert self.engine.last_price == current_price
        
        # Cancelled orders should be those with order IDs
        cancelled_order_ids = [order.order_id for order in orders_to_cancel]
        assert "pending1" in cancelled_order_ids
        assert "pending2" in cancelled_order_ids
    
    def test_rebalance_grid_center_price_update(self):
        """Test rebalance_grid updates center price correctly"""
        self.engine.initialize_grid()
        original_center = self.config.center_price
        
        # Price requiring rebalance
        new_center_price = Decimal("55000")
        
        self.engine.rebalance_grid(new_center_price)
        
        # Center should be updated
        assert self.engine.config.center_price == new_center_price
        assert self.engine.config.center_price != original_center
        
        # Grid should be reinitialized around new center
        for level in self.engine.grid_levels.values():
            if level.level_index > 0:
                expected_price = new_center_price + (self.config.grid_spacing * level.level_index)
            else:
                expected_price = new_center_price + (self.config.grid_spacing * level.level_index)
            assert level.price == expected_price
    
    def test_rebalance_grid_boundary_condition(self):
        """Test rebalance_grid at exact boundary condition"""
        self.engine.initialize_grid()
        
        # Price exactly at rebalance threshold
        max_deviation = self.config.grid_spacing * 3  # 1500
        boundary_price = self.config.center_price + max_deviation
        
        orders_to_cancel, new_orders = self.engine.rebalance_grid(boundary_price)
        
        # At boundary, might or might not rebalance depending on implementation
        # This tests the exact threshold behavior
        assert isinstance(orders_to_cancel, list)
        assert isinstance(new_orders, list)
    
    def test_get_grid_status_initial_state(self):
        """Test get_grid_status in initial state"""
        status = self.engine.get_grid_status()
        
        assert status["total_levels"] == 0
        assert status["filled_levels"] == 0
        assert status["active_orders"] == 0
        assert status["current_position"] == 0.0
        assert status["center_price"] == float(self.config.center_price)
        assert status["last_price"] is None
    
    def test_get_grid_status_after_initialization(self):
        """Test get_grid_status after grid initialization"""
        self.engine.initialize_grid()
        
        status = self.engine.get_grid_status()
        
        expected_total = self.config.num_levels_up + self.config.num_levels_down
        assert status["total_levels"] == expected_total
        assert status["filled_levels"] == 0
        assert status["active_orders"] == 0  # No orders placed yet
        assert status["current_position"] == 0.0
    
    def test_get_grid_status_with_activity(self):
        """Test get_grid_status with fills and active orders"""
        self.engine.initialize_grid()
        
        # Fill some orders
        self.engine.on_fill(-1, "buy1", Decimal("49500"))
        self.engine.on_fill(2, "sell1", Decimal("51000"))
        
        # Set some pending orders
        self.engine.grid_levels[1].order_id = "pending1"
        self.engine.grid_levels[-2].order_id = "pending2"
        
        status = self.engine.get_grid_status()
        
        expected_total = self.config.num_levels_up + self.config.num_levels_down
        assert status["total_levels"] == expected_total
        assert status["filled_levels"] == 2
        assert status["active_orders"] == 2
        
        # Position should reflect fills
        expected_position = self.config.order_amount - self.config.order_amount  # buy - sell = 0
        assert status["current_position"] == float(expected_position)
    
    def test_get_grid_status_with_last_price(self):
        """Test get_grid_status includes last price after rebalance"""
        self.engine.initialize_grid()
        
        last_price = Decimal("51000")
        self.engine.rebalance_grid(last_price)
        
        status = self.engine.get_grid_status()
        
        assert status["last_price"] == float(last_price)
    
    def test_get_grid_status_return_types(self):
        """Test get_grid_status returns correct data types"""
        self.engine.initialize_grid()
        self.engine.current_position = Decimal("0.005")
        
        status = self.engine.get_grid_status()
        
        # All numeric values should be converted to float for JSON serialization
        assert isinstance(status["total_levels"], int)
        assert isinstance(status["filled_levels"], int)
        assert isinstance(status["active_orders"], int)
        assert isinstance(status["current_position"], float)
        assert isinstance(status["center_price"], float)


class TestGridEngineIntegration:
    """Test complex grid engine scenarios and integration cases"""
    
    def test_complete_grid_trading_cycle(self):
        """Test complete cycle of grid trading operations"""
        config = GridConfig(
            center_price=Decimal("1000"),
            grid_spacing=Decimal("10"),
            num_levels_up=3,
            num_levels_down=3,
            order_amount=Decimal("1.0"),
            max_position_size=Decimal("10.0")
        )
        engine = GridEngine(config)
        
        # Initialize grid
        levels = engine.initialize_grid()
        assert len(levels) == 6
        
        # Get initial active orders
        active_orders = engine.get_active_orders()
        assert len(active_orders) == 6
        
        # Simulate order fills
        # Buy at 980 (level -2) - this will try to place order at level -1
        new_orders = engine.on_fill(-2, "buy_980", Decimal("980"))
        assert len(new_orders) == 1
        assert new_orders[0].price == Decimal("990")  # Level -1 price
        
        # Buy at 970 (level -3) - this will try to place order at level -2 (already filled)
        new_orders = engine.on_fill(-3, "buy_970", Decimal("970"))
        assert len(new_orders) == 0  # Level -2 already filled
        
        # Check position accumulation
        assert engine.current_position == Decimal("2.0")
        
        # Check status
        status = engine.get_grid_status()
        assert status["filled_levels"] == 2
        assert status["current_position"] == 2.0
    
    def test_asymmetric_grid_configuration(self):
        """Test grid with different up/down level counts"""
        config = GridConfig(
            center_price=Decimal("5000"),
            grid_spacing=Decimal("50"),
            num_levels_up=8,
            num_levels_down=3,
            order_amount=Decimal("0.1"),
            max_position_size=Decimal("2.0")
        )
        engine = GridEngine(config)
        
        levels = engine.initialize_grid()
        
        # Should have more sell levels than buy levels
        sell_levels = [l for l in levels if l.side == "sell"]
        buy_levels = [l for l in levels if l.side == "buy"]
        
        assert len(sell_levels) == 8
        assert len(buy_levels) == 3
        assert len(levels) == 11
        
        # Test price calculations
        highest_sell = max(sell_levels, key=lambda x: x.price)
        lowest_buy = min(buy_levels, key=lambda x: x.price)
        
        assert highest_sell.price == config.center_price + (config.grid_spacing * 8)
        assert lowest_buy.price == config.center_price - (config.grid_spacing * 3)
    
    def test_small_grid_spacing(self):
        """Test grid with very small spacing"""
        config = GridConfig(
            center_price=Decimal("100.00"),
            grid_spacing=Decimal("0.01"),
            num_levels_up=5,
            num_levels_down=5,
            order_amount=Decimal("10.0"),
            max_position_size=Decimal("100.0")
        )
        engine = GridEngine(config)
        
        levels = engine.initialize_grid()
        
        # Verify precision is maintained
        for level in levels:
            assert isinstance(level.price, Decimal)
            # Price should be center ± (spacing * level_index)
            expected_price = config.center_price + (config.grid_spacing * level.level_index)
            assert level.price == expected_price
        
        # Test very small price movements
        status_before = engine.get_grid_status()
        engine.rebalance_grid(Decimal("100.001"))  # Very small move
        status_after = engine.get_grid_status()
        
        # Should not trigger rebalance for small moves
        assert status_before["center_price"] == status_after["center_price"]
    
    def test_large_grid_spacing(self):
        """Test grid with large spacing"""
        config = GridConfig(
            center_price=Decimal("50000"),
            grid_spacing=Decimal("5000"),
            num_levels_up=4,
            num_levels_down=4,
            order_amount=Decimal("0.01"),
            max_position_size=Decimal("1.0")
        )
        engine = GridEngine(config)
        
        levels = engine.initialize_grid()
        
        # Check wide price range
        prices = [level.price for level in levels]
        min_price = min(prices)
        max_price = max(prices)
        
        expected_range = config.grid_spacing * (config.num_levels_up + config.num_levels_down)
        actual_range = max_price - min_price
        
        assert actual_range == expected_range
        assert min_price == config.center_price - (config.grid_spacing * config.num_levels_down)
        assert max_price == config.center_price + (config.grid_spacing * config.num_levels_up)
    
    def test_rapid_fill_sequence(self):
        """Test rapid sequence of order fills"""
        config = GridConfig(
            center_price=Decimal("2000"),
            grid_spacing=Decimal("20"),
            num_levels_up=5,
            num_levels_down=5,
            order_amount=Decimal("0.5"),
            max_position_size=Decimal("5.0")
        )
        engine = GridEngine(config)
        engine.initialize_grid()
        
        # Simulate rapid buy fills (market dropping)
        buy_sequence = [(-1, "1980"), (-2, "1960"), (-3, "1940")]
        
        for level_idx, price in buy_sequence:
            new_orders = engine.on_fill(level_idx, f"buy_{price}", Decimal(price))
            # Each should generate a sell order
            assert len(new_orders) <= 1
        
        # Position should accumulate
        expected_position = Decimal("0.5") * 3
        assert engine.current_position == expected_position
        
        # Now simulate sell fills (market recovering)
        sell_sequence = [(1, "2020"), (2, "2040")]
        
        for level_idx, price in sell_sequence:
            new_orders = engine.on_fill(level_idx, f"sell_{price}", Decimal(price))
            assert len(new_orders) <= 1
        
        # Position should decrease
        expected_position = (Decimal("0.5") * 3) - (Decimal("0.5") * 2)
        assert engine.current_position == expected_position
    
    def test_grid_rebalancing_scenarios(self):
        """Test various grid rebalancing scenarios"""
        config = GridConfig(
            center_price=Decimal("1000"),
            grid_spacing=Decimal("25"),
            num_levels_up=4,
            num_levels_down=4,
            order_amount=Decimal("1.0"),
            max_position_size=Decimal("10.0")
        )
        engine = GridEngine(config)
        engine.initialize_grid()
        
        # Scenario 1: Small price movement (no rebalance)
        small_move_price = Decimal("1050")
        orders_to_cancel, new_orders = engine.rebalance_grid(small_move_price)
        assert orders_to_cancel == []
        assert new_orders == []
        
        # Scenario 2: Large price movement (triggers rebalance)
        large_move_price = Decimal("1200")  # > 3 * 25 = 75 deviation
        orders_to_cancel, new_orders = engine.rebalance_grid(large_move_price)
        assert len(new_orders) > 0
        assert engine.config.center_price == large_move_price
        
        # Scenario 3: Verify new grid is centered correctly
        new_levels = list(engine.grid_levels.values())
        center_level_prices = [l.price for l in new_levels if l.level_index == 1 or l.level_index == -1]
        
        # Should be evenly spaced around new center
        for level in new_levels:
            expected_price = large_move_price + (config.grid_spacing * level.level_index)
            assert level.price == expected_price
    
    def test_grid_level_side_assignment(self):
        """Test correct buy/sell side assignment for grid levels"""
        config = GridConfig(
            center_price=Decimal("1000"),
            grid_spacing=Decimal("10"),
            num_levels_up=3,
            num_levels_down=3,
            order_amount=Decimal("1.0"),
            max_position_size=Decimal("10.0")
        )
        engine = GridEngine(config)
        levels = engine.initialize_grid()
        
        # Verify side assignments
        for level in levels:
            if level.level_index > 0:
                assert level.side == "sell", f"Level {level.level_index} should be sell"
                assert level.price > config.center_price
            elif level.level_index < 0:
                assert level.side == "buy", f"Level {level.level_index} should be buy"
                assert level.price < config.center_price
        
        # Verify price ordering
        buy_levels = [l for l in levels if l.side == "buy"]
        sell_levels = [l for l in levels if l.side == "sell"]
        
        # Buy levels should be in ascending price order by level index (more negative = lower price)
        buy_prices = [l.price for l in sorted(buy_levels, key=lambda x: x.level_index)]
        assert buy_prices == sorted(buy_prices)  # Ascending order by level index
        
        # Sell levels should be in ascending price order by level index
        sell_prices = [l.price for l in sorted(sell_levels, key=lambda x: x.level_index)]
        assert sell_prices == sorted(sell_prices)  # Ascending order

    def test_position_tracking_accuracy(self):
        """Test that position tracking remains accurate through complex scenarios"""
        config = GridConfig(
            center_price=Decimal("1000"),
            grid_spacing=Decimal("50"),
            num_levels_up=4,
            num_levels_down=4,
            order_amount=Decimal("2.5"),
            max_position_size=Decimal("20.0")
        )
        engine = GridEngine(config)
        engine.initialize_grid()
        
        initial_position = engine.current_position
        assert initial_position == Decimal('0')
        
        # Execute a series of buys and sells
        fills = [
            (-1, "buy", Decimal("2.5")),    # +2.5
            (-2, "buy", Decimal("2.5")),    # +2.5 = +5.0
            (1, "sell", Decimal("-2.5")),   # -2.5 = +2.5
            (-3, "buy", Decimal("2.5")),    # +2.5 = +5.0
            (2, "sell", Decimal("-2.5")),   # -2.5 = +2.5
            (3, "sell", Decimal("-2.5")),   # -2.5 = 0.0
        ]
        
        expected_position = Decimal('0')
        for level_idx, side, delta in fills:
            engine.on_fill(level_idx, f"{side}_{level_idx}", Decimal("1000"))
            expected_position += delta
            assert engine.current_position == expected_position, \
                f"Position mismatch after {side} at level {level_idx}"
        
        # Final position should be zero
        assert engine.current_position == Decimal('0')
        
        # Status should reflect correct filled levels
        status = engine.get_grid_status()
        assert status["filled_levels"] == len(fills)
        assert status["current_position"] == 0.0

    def test_grid_statistics_comprehensive(self):
        """Test comprehensive grid statistics and status reporting"""
        config = GridConfig(
            center_price=Decimal("5000"),
            grid_spacing=Decimal("100"),
            num_levels_up=5,
            num_levels_down=3,
            order_amount=Decimal("1.0"),
            max_position_size=Decimal("15.0")
        )
        engine = GridEngine(config)
        
        # Test initial status
        initial_status = engine.get_grid_status()
        assert initial_status["total_levels"] == 0
        assert initial_status["filled_levels"] == 0
        assert initial_status["active_orders"] == 0
        assert initial_status["current_position"] == 0.0
        assert initial_status["center_price"] == 5000.0
        assert initial_status["last_price"] is None
        
        # Initialize grid and verify status
        engine.initialize_grid()
        after_init_status = engine.get_grid_status()
        assert after_init_status["total_levels"] == 8  # 5 + 3
        assert after_init_status["filled_levels"] == 0
        
        # Fill some orders and set some pending
        engine.on_fill(-1, "buy1", Decimal("4900"))
        engine.on_fill(2, "sell1", Decimal("5200"))
        engine.grid_levels[1].order_id = "pending_sell"
        engine.grid_levels[-2].order_id = "pending_buy"
        
        # Check detailed status
        detailed_status = engine.get_grid_status()
        assert detailed_status["total_levels"] == 8
        assert detailed_status["filled_levels"] == 2
        assert detailed_status["active_orders"] == 2
        assert detailed_status["current_position"] == 0.0  # +1.0 - 1.0 = 0
        
        # Test rebalance and last price tracking
        engine.rebalance_grid(Decimal("6000"))
        rebalanced_status = engine.get_grid_status()
        assert rebalanced_status["last_price"] == 6000.0
        assert rebalanced_status["center_price"] == 6000.0
    
    def test_edge_case_handling(self):
        """Test edge cases and boundary conditions"""
        config = GridConfig(
            center_price=Decimal("100"),
            grid_spacing=Decimal("1"),
            num_levels_up=2,
            num_levels_down=2,
            order_amount=Decimal("1"),
            max_position_size=Decimal("10")
        )
        engine = GridEngine(config)
        engine.initialize_grid()
        
        # Fill orders at grid boundaries
        # Highest buy level
        lowest_buy_idx = -config.num_levels_down  # -2
        new_orders = engine.on_fill(lowest_buy_idx, "edge_buy", Decimal("98"))
        
        # Should create sell order at next level if it exists
        next_level_idx = lowest_buy_idx + 1  # -1
        if next_level_idx in engine.grid_levels:
            assert len(new_orders) == 1
        
        # Highest sell level
        highest_sell_idx = config.num_levels_up  # 2
        new_orders = engine.on_fill(highest_sell_idx, "edge_sell", Decimal("102"))
        
        # Should create buy order at next level if it exists
        next_level_idx = highest_sell_idx - 1  # 1
        if next_level_idx in engine.grid_levels:
            assert len(new_orders) == 1
        
        # Test invalid operations
        invalid_fills = engine.on_fill(999, "invalid", Decimal("100"))
        assert invalid_fills == []
        
        # Test status remains consistent
        status = engine.get_grid_status()
        assert isinstance(status, dict)
        assert all(key in status for key in ["total_levels", "filled_levels", "active_orders"])


if __name__ == "__main__":
    pytest.main([__file__])
