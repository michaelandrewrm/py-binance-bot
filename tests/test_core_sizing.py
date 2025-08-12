"""
Test Suite for Core Module - Sizing Component

This module provides comprehensive unit tests for the sizing calculations
including position sizing, fee calculations, exchange constraints, and validation.
"""

import pytest
from unittest.mock import Mock, patch
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Tuple, Dict

# Import the modules under test
from core.sizing import (
    ExchangeInfo,
    FeeConfig,
    SizingCalculator,
    create_binance_exchange_info,
    ensure_min_notional,
    ensure_min_notional_float
)


class TestExchangeInfo:
    """Test ExchangeInfo data class functionality"""
    
    def test_exchange_info_creation(self):
        """Test creating ExchangeInfo with all required fields"""
        exchange_info = ExchangeInfo(
            symbol="BTCUSDC",
            min_qty=Decimal("0.00001"),
            max_qty=Decimal("1000"),
            step_size=Decimal("0.00001"),
            tick_size=Decimal("0.01"),
            min_notional=Decimal("10"),
            base_asset_precision=5,
            quote_asset_precision=2
        )
        
        assert exchange_info.symbol == "BTCUSDC"
        assert exchange_info.min_qty == Decimal("0.00001")
        assert exchange_info.max_qty == Decimal("1000")
        assert exchange_info.step_size == Decimal("0.00001")
        assert exchange_info.tick_size == Decimal("0.01")
        assert exchange_info.min_notional == Decimal("10")
        assert exchange_info.base_asset_precision == 5
        assert exchange_info.quote_asset_precision == 2
    
    def test_exchange_info_decimal_types(self):
        """Test that ExchangeInfo properly handles Decimal types"""
        exchange_info = ExchangeInfo(
            symbol="ETHUSDC",
            min_qty=Decimal("0.001"),
            max_qty=Decimal("10000"),
            step_size=Decimal("0.001"),
            tick_size=Decimal("0.01"),
            min_notional=Decimal("5"),
            base_asset_precision=3,
            quote_asset_precision=2
        )
        
        # All decimal fields should maintain precision
        assert isinstance(exchange_info.min_qty, Decimal)
        assert isinstance(exchange_info.max_qty, Decimal)
        assert isinstance(exchange_info.step_size, Decimal)
        assert isinstance(exchange_info.tick_size, Decimal)
        assert isinstance(exchange_info.min_notional, Decimal)


class TestFeeConfig:
    """Test FeeConfig data class functionality"""
    
    def test_fee_config_defaults(self):
        """Test FeeConfig with default values"""
        fee_config = FeeConfig()
        
        assert fee_config.maker_fee == Decimal('0.001')  # 0.1%
        assert fee_config.taker_fee == Decimal('0.001')  # 0.1%
        assert fee_config.bnb_discount == Decimal('0.75')  # 25% discount
        assert fee_config.use_bnb is False
    
    def test_fee_config_custom_values(self):
        """Test FeeConfig with custom values"""
        fee_config = FeeConfig(
            maker_fee=Decimal('0.0005'),  # 0.05%
            taker_fee=Decimal('0.0008'),  # 0.08%
            bnb_discount=Decimal('0.8'),  # 20% discount
            use_bnb=True
        )
        
        assert fee_config.maker_fee == Decimal('0.0005')
        assert fee_config.taker_fee == Decimal('0.0008')
        assert fee_config.bnb_discount == Decimal('0.8')
        assert fee_config.use_bnb is True
    
    def test_fee_config_high_precision(self):
        """Test FeeConfig with high precision decimal values"""
        fee_config = FeeConfig(
            maker_fee=Decimal('0.00075'),
            taker_fee=Decimal('0.00095'),
            bnb_discount=Decimal('0.75')
        )
        
        # Should maintain decimal precision
        assert fee_config.maker_fee == Decimal('0.00075')
        assert fee_config.taker_fee == Decimal('0.00095')


class TestSizingCalculator:
    """Test SizingCalculator functionality"""
    
    def setup_method(self):
        """Setup SizingCalculator with standard test configuration"""
        self.exchange_info = ExchangeInfo(
            symbol="BTCUSDC",
            min_qty=Decimal("0.00001"),
            max_qty=Decimal("1000"),
            step_size=Decimal("0.00001"),
            tick_size=Decimal("0.01"),
            min_notional=Decimal("10"),
            base_asset_precision=5,
            quote_asset_precision=2
        )
        
        self.fee_config = FeeConfig(
            maker_fee=Decimal('0.001'),
            taker_fee=Decimal('0.001'),
            use_bnb=False
        )
        
        self.calculator = SizingCalculator(self.exchange_info, self.fee_config)
    
    def test_sizing_calculator_initialization(self):
        """Test SizingCalculator initializes correctly"""
        assert self.calculator.exchange_info == self.exchange_info
        assert self.calculator.fee_config == self.fee_config
    
    def test_round_price_down(self):
        """Test price rounding down to tick size"""
        price = Decimal("50123.456")
        rounded = self.calculator.round_price(price, round_up=False)
        
        # Should round down to nearest tick (0.01)
        assert rounded == Decimal("50123.45")
    
    def test_round_price_up(self):
        """Test price rounding up to tick size"""
        price = Decimal("50123.451")
        rounded = self.calculator.round_price(price, round_up=True)
        
        # Should round up to nearest tick (0.01)
        assert rounded == Decimal("50123.46")
    
    def test_round_price_exact_tick(self):
        """Test price rounding when already at exact tick"""
        price = Decimal("50123.45")
        
        rounded_down = self.calculator.round_price(price, round_up=False)
        rounded_up = self.calculator.round_price(price, round_up=True)
        
        # Should remain the same
        assert rounded_down == price
        assert rounded_up == price
    
    def test_round_price_zero_tick_size(self):
        """Test price rounding with zero tick size"""
        # Create calculator with zero tick size
        exchange_info = ExchangeInfo(
            symbol="TEST",
            min_qty=Decimal("0.001"),
            max_qty=Decimal("1000"),
            step_size=Decimal("0.001"),
            tick_size=Decimal("0"),  # Zero tick size
            min_notional=Decimal("10"),
            base_asset_precision=3,
            quote_asset_precision=2
        )
        calculator = SizingCalculator(exchange_info, self.fee_config)
        
        price = Decimal("123.456789")
        rounded = calculator.round_price(price)
        
        # Should return original price when tick_size is 0
        assert rounded == price
    
    def test_round_quantity_down(self):
        """Test quantity rounding down to step size"""
        quantity = Decimal("0.123456")
        rounded = self.calculator.round_quantity(quantity, round_up=False)
        
        # Should round down to nearest step (0.00001)
        assert rounded == Decimal("0.12345")
    
    def test_round_quantity_up(self):
        """Test quantity rounding up to step size"""
        quantity = Decimal("0.123451")
        rounded = self.calculator.round_quantity(quantity, round_up=True)
        
        # Should round up to nearest step (0.00001)
        assert rounded == Decimal("0.12346")
    
    def test_round_quantity_zero_step_size(self):
        """Test quantity rounding with zero step size"""
        exchange_info = ExchangeInfo(
            symbol="TEST",
            min_qty=Decimal("0.001"),
            max_qty=Decimal("1000"),
            step_size=Decimal("0"),  # Zero step size
            tick_size=Decimal("0.01"),
            min_notional=Decimal("10"),
            base_asset_precision=3,
            quote_asset_precision=2
        )
        calculator = SizingCalculator(exchange_info, self.fee_config)
        
        quantity = Decimal("0.123456")
        rounded = calculator.round_quantity(quantity)
        
        # Should return original quantity when step_size is 0
        assert rounded == quantity
    
    def test_calculate_max_quantity_normal_case(self):
        """Test calculating maximum quantity with normal parameters"""
        price = Decimal("50000")
        available_balance = Decimal("1000")
        
        max_qty = self.calculator.calculate_max_quantity(price, available_balance)
        
        # Should account for fees and respect constraints
        # Available / (price * (1 + fee_rate))
        expected_raw = available_balance / (price * Decimal("1.001"))
        expected_rounded = self.calculator.round_quantity(expected_raw, round_up=False)
        
        assert max_qty == expected_rounded
        assert max_qty >= self.exchange_info.min_qty
        assert max_qty <= self.exchange_info.max_qty
    
    def test_calculate_max_quantity_insufficient_balance(self):
        """Test calculating max quantity with insufficient balance"""
        price = Decimal("50000")
        available_balance = Decimal("0.1")  # Very small balance
        
        max_qty = self.calculator.calculate_max_quantity(price, available_balance)
        
        # Should return 0 when balance insufficient for min_qty
        assert max_qty == Decimal("0")
    
    def test_calculate_max_quantity_exceeds_max_qty(self):
        """Test calculating max quantity when result exceeds exchange max"""
        price = Decimal("1")  # Very low price
        available_balance = Decimal("100000")  # Large balance
        
        max_qty = self.calculator.calculate_max_quantity(price, available_balance)
        
        # Should be capped at exchange max_qty
        assert max_qty == self.exchange_info.max_qty
    
    def test_validate_order_valid(self):
        """Test order validation with valid parameters"""
        price = Decimal("50000")
        quantity = Decimal("0.001")
        
        is_valid, message = self.calculator.validate_order(price, quantity)
        
        assert is_valid is True
        assert message == "Order is valid"
    
    def test_validate_order_quantity_too_small(self):
        """Test order validation with quantity below minimum"""
        price = Decimal("50000")
        quantity = Decimal("0.000001")  # Below min_qty
        
        is_valid, message = self.calculator.validate_order(price, quantity)
        
        assert is_valid is False
        assert "below minimum quantity" in message
    
    def test_validate_order_quantity_too_large(self):
        """Test order validation with quantity above maximum"""
        price = Decimal("50000")
        quantity = Decimal("10000")  # Above max_qty
        
        is_valid, message = self.calculator.validate_order(price, quantity)
        
        assert is_valid is False
        assert "above maximum quantity" in message
    
    def test_validate_order_below_min_notional(self):
        """Test order validation below minimum notional value"""
        price = Decimal("1")
        quantity = Decimal("0.001")  # 1 * 0.001 = 0.001 < 10 min_notional
        
        is_valid, message = self.calculator.validate_order(price, quantity)
        
        assert is_valid is False
        assert "below minimum notional" in message
    
    def test_validate_order_quantity_precision(self):
        """Test order validation with incorrect quantity precision"""
        price = Decimal("50000")
        quantity = Decimal("0.000015")  # Not aligned with step_size (0.00001)
        
        is_valid, message = self.calculator.validate_order(price, quantity)
        
        assert is_valid is False
        assert "invalid quantity precision" in message
    
    def test_validate_order_price_precision(self):
        """Test order validation with incorrect price precision"""
        price = Decimal("50000.123")  # Not aligned with tick_size (0.01)
        quantity = Decimal("0.001")
        
        is_valid, message = self.calculator.validate_order(price, quantity)
        
        assert is_valid is False
        assert "invalid price precision" in message
    
    def test_calculate_fees_maker_order(self):
        """Test fee calculation for maker orders"""
        price = Decimal("50000")
        quantity = Decimal("0.001")
        
        fees = self.calculator.calculate_fees(price, quantity, is_maker=True)
        
        expected_total = price * quantity
        expected_fee = expected_total * self.fee_config.maker_fee
        
        assert fees["order_value"] == expected_total
        assert fees["fee_rate"] == self.fee_config.maker_fee
        assert fees["fee_amount"] == expected_fee
        assert fees["net_amount"] == expected_total - expected_fee
        assert fees["is_maker"] is True
        assert fees["used_bnb_discount"] is False
    
    def test_calculate_fees_taker_order(self):
        """Test fee calculation for taker orders"""
        price = Decimal("50000")
        quantity = Decimal("0.001")
        
        fees = self.calculator.calculate_fees(price, quantity, is_maker=False)
        
        expected_total = price * quantity
        expected_fee = expected_total * self.fee_config.taker_fee
        
        assert fees["fee_rate"] == self.fee_config.taker_fee
        assert fees["fee_amount"] == expected_fee
        assert fees["is_maker"] is False
    
    def test_calculate_fees_with_bnb_discount(self):
        """Test fee calculation with BNB discount enabled"""
        # Create calculator with BNB discount enabled
        fee_config = FeeConfig(use_bnb=True)
        calculator = SizingCalculator(self.exchange_info, fee_config)
        
        price = Decimal("50000")
        quantity = Decimal("0.001")
        
        fees = calculator.calculate_fees(price, quantity, is_maker=True)
        
        expected_total = price * quantity
        base_fee = expected_total * fee_config.maker_fee
        discounted_fee = base_fee * fee_config.bnb_discount
        
        assert fees["fee_amount"] == discounted_fee
        assert fees["used_bnb_discount"] is True
        assert fees["discount_saved"] == base_fee - discounted_fee
    
    def test_get_effective_fee_rate_maker_no_bnb(self):
        """Test effective fee rate calculation for maker without BNB"""
        rate = self.calculator.get_effective_fee_rate(is_maker=True)
        assert rate == self.fee_config.maker_fee
    
    def test_get_effective_fee_rate_taker_no_bnb(self):
        """Test effective fee rate calculation for taker without BNB"""
        rate = self.calculator.get_effective_fee_rate(is_maker=False)
        assert rate == self.fee_config.taker_fee
    
    def test_get_effective_fee_rate_with_bnb(self):
        """Test effective fee rate calculation with BNB discount"""
        fee_config = FeeConfig(use_bnb=True)
        calculator = SizingCalculator(self.exchange_info, fee_config)
        
        maker_rate = calculator.get_effective_fee_rate(is_maker=True)
        taker_rate = calculator.get_effective_fee_rate(is_maker=False)
        
        expected_maker = fee_config.maker_fee * fee_config.bnb_discount
        expected_taker = fee_config.taker_fee * fee_config.bnb_discount
        
        assert maker_rate == expected_maker
        assert taker_rate == expected_taker
    
    def test_calculate_grid_order_size_even_distribution(self):
        """Test grid order size calculation with even distribution"""
        total_capital = Decimal("1000")
        num_levels = 10
        
        result = self.calculator.calculate_grid_order_size(
            total_capital=total_capital,
            num_levels=num_levels,
            per_level_percentage=None  # Even distribution
        )
        
        expected_per_level = total_capital / num_levels
        
        assert result["total_capital"] == total_capital
        assert result["num_levels"] == num_levels
        assert result["order_size_quote"] == expected_per_level
        assert result["distribution_type"] == "even"
    
    def test_calculate_grid_order_size_percentage_distribution(self):
        """Test grid order size calculation with percentage distribution"""
        total_capital = Decimal("1000")
        num_levels = 5
        percentage = Decimal("0.15")  # 15% per level
        
        result = self.calculator.calculate_grid_order_size(
            total_capital=total_capital,
            num_levels=num_levels,
            per_level_percentage=percentage
        )
        
        expected_per_level = total_capital * percentage
        
        assert result["order_size_quote"] == expected_per_level
        assert result["distribution_type"] == "percentage"
        assert result["percentage_per_level"] == percentage
    
    def test_calculate_grid_order_size_capital_utilization(self):
        """Test grid order size calculation shows capital utilization"""
        total_capital = Decimal("1000")
        num_levels = 8
        
        result = self.calculator.calculate_grid_order_size(
            total_capital=total_capital,
            num_levels=num_levels
        )
        
        total_allocated = result["order_size_quote"] * num_levels
        expected_utilization = (total_allocated / total_capital) * 100
        
        assert result["total_allocated"] == total_allocated
        assert result["capital_utilization_percent"] == expected_utilization
    
    def test_calculate_position_value(self):
        """Test position value calculation"""
        quantity = Decimal("0.5")
        current_price = Decimal("50000")
        
        result = self.calculator.calculate_position_value(quantity, current_price)
        
        expected_value = quantity * current_price
        
        assert result["quantity"] == quantity
        assert result["current_price"] == current_price
        assert result["total_value"] == expected_value
        assert result["symbol"] == self.exchange_info.symbol
        
        # Verify precision handling
        assert isinstance(result["total_value"], Decimal)


class TestCreateBinanceExchangeInfo:
    """Test create_binance_exchange_info function"""
    
    def test_create_binance_exchange_info_btc(self):
        """Test creating Binance exchange info for BTC"""
        symbol = "BTCUSDC"
        price = Decimal("50000")
        
        exchange_info = create_binance_exchange_info(symbol, price)
        
        assert exchange_info.symbol == symbol
        assert exchange_info.min_qty == Decimal("0.00001")
        assert exchange_info.step_size == Decimal("0.00001")
        assert exchange_info.tick_size == Decimal("0.01")
        assert exchange_info.min_notional == Decimal("10")
        assert exchange_info.base_asset_precision == 5
        assert exchange_info.quote_asset_precision == 2
    
    def test_create_binance_exchange_info_eth(self):
        """Test creating Binance exchange info for ETH"""
        symbol = "ETHUSDC"
        price = Decimal("3000")
        
        exchange_info = create_binance_exchange_info(symbol, price)
        
        assert exchange_info.symbol == symbol
        # Should have appropriate constraints for ETH
        assert exchange_info.min_qty > Decimal("0")
        assert exchange_info.step_size > Decimal("0")
        assert exchange_info.tick_size > Decimal("0")
    
    def test_create_binance_exchange_info_max_qty_calculation(self):
        """Test max quantity calculation based on price"""
        symbol = "TESTUSDC"
        price = Decimal("1")  # Low price
        
        exchange_info = create_binance_exchange_info(symbol, price)
        
        # Max quantity should be reasonable for low price assets
        assert exchange_info.max_qty > Decimal("1000")
        
        # Test with high price
        high_price = Decimal("100000")
        exchange_info_high = create_binance_exchange_info(symbol, high_price)
        
        # Max quantity should be lower for high price assets
        assert exchange_info_high.max_qty < exchange_info.max_qty
    
    def test_create_binance_exchange_info_precision_calculation(self):
        """Test precision calculation based on price levels"""
        # Test different price ranges
        test_cases = [
            (Decimal("0.001"), 6),  # Very small price, high precision
            (Decimal("1"), 3),      # Medium price, medium precision
            (Decimal("1000"), 2),   # High price, lower precision
            (Decimal("100000"), 1)  # Very high price, low precision
        ]
        
        for price, expected_min_precision in test_cases:
            exchange_info = create_binance_exchange_info("TESTUSDC", price)
            assert exchange_info.base_asset_precision >= expected_min_precision


class TestMinNotionalFunctions:
    """Test minimum notional utility functions"""
    
    def test_ensure_min_notional_above_minimum(self):
        """Test ensure_min_notional with value above minimum"""
        price = Decimal("50000")
        qty = Decimal("0.001")
        min_notional = Decimal("10")
        
        # 50000 * 0.001 = 50 > 10
        result = ensure_min_notional(price, qty, min_notional)
        assert result is True
    
    def test_ensure_min_notional_below_minimum(self):
        """Test ensure_min_notional with value below minimum"""
        price = Decimal("1")
        qty = Decimal("0.001")
        min_notional = Decimal("10")
        
        # 1 * 0.001 = 0.001 < 10
        result = ensure_min_notional(price, qty, min_notional)
        assert result is False
    
    def test_ensure_min_notional_exactly_minimum(self):
        """Test ensure_min_notional with value exactly at minimum"""
        price = Decimal("10")
        qty = Decimal("1")
        min_notional = Decimal("10")
        
        # 10 * 1 = 10 = 10
        result = ensure_min_notional(price, qty, min_notional)
        assert result is True
    
    def test_ensure_min_notional_float_version(self):
        """Test ensure_min_notional_float with float inputs"""
        price = 50000.0
        qty = 0.001
        min_notional = 10.0
        
        # 50000 * 0.001 = 50 > 10
        result = ensure_min_notional_float(price, qty, min_notional)
        assert result is True
        
        # Test below minimum
        price = 1.0
        qty = 0.001
        min_notional = 10.0
        
        # 1 * 0.001 = 0.001 < 10
        result = ensure_min_notional_float(price, qty, min_notional)
        assert result is False
    
    def test_ensure_min_notional_zero_values(self):
        """Test ensure_min_notional with zero values"""
        # Zero price
        result = ensure_min_notional(Decimal("0"), Decimal("1"), Decimal("10"))
        assert result is False
        
        # Zero quantity
        result = ensure_min_notional(Decimal("50"), Decimal("0"), Decimal("10"))
        assert result is False
        
        # Zero minimum (should always pass)
        result = ensure_min_notional(Decimal("1"), Decimal("1"), Decimal("0"))
        assert result is True
    
    def test_ensure_min_notional_precision_handling(self):
        """Test ensure_min_notional with high precision values"""
        price = Decimal("12345.6789")
        qty = Decimal("0.0008765")
        min_notional = Decimal("10.0")
        
        notional_value = price * qty  # Should be > 10
        result = ensure_min_notional(price, qty, min_notional)
        
        assert notional_value > min_notional
        assert result is True


class TestIntegrationScenarios:
    """Test complex integration scenarios"""
    
    def test_complete_order_sizing_workflow(self):
        """Test complete order sizing and validation workflow"""
        # Setup realistic exchange constraints
        exchange_info = ExchangeInfo(
            symbol="BTCUSDC",
            min_qty=Decimal("0.00001"),
            max_qty=Decimal("1000"),
            step_size=Decimal("0.00001"),
            tick_size=Decimal("0.01"),
            min_notional=Decimal("10"),
            base_asset_precision=5,
            quote_asset_precision=2
        )
        
        fee_config = FeeConfig()
        calculator = SizingCalculator(exchange_info, fee_config)
        
        # Test workflow: round price -> calculate max qty -> validate order -> calculate fees
        raw_price = Decimal("50123.456")
        available_balance = Decimal("1000")
        
        # Step 1: Round price
        price = calculator.round_price(raw_price)
        assert price == Decimal("50123.46")
        
        # Step 2: Calculate maximum quantity
        max_qty = calculator.calculate_max_quantity(price, available_balance)
        assert max_qty > Decimal("0")
        
        # Step 3: Validate order
        is_valid, message = calculator.validate_order(price, max_qty)
        assert is_valid is True
        
        # Step 4: Calculate fees
        fees = calculator.calculate_fees(price, max_qty, is_maker=True)
        assert fees["fee_amount"] > Decimal("0")
        assert fees["net_amount"] < fees["order_value"]
    
    def test_grid_trading_order_sizing(self):
        """Test order sizing for grid trading scenarios"""
        exchange_info = create_binance_exchange_info("ETHUSDC", Decimal("3000"))
        fee_config = FeeConfig(maker_fee=Decimal("0.0008"))  # VIP level
        calculator = SizingCalculator(exchange_info, fee_config)
        
        # Grid trading parameters
        total_capital = Decimal("5000")
        num_grid_levels = 20
        
        # Calculate order size per grid level
        grid_result = calculator.calculate_grid_order_size(
            total_capital=total_capital,
            num_levels=num_grid_levels
        )
        
        order_size_quote = grid_result["order_size_quote"]
        
        # Test order validation for grid levels
        test_prices = [Decimal("2800"), Decimal("3000"), Decimal("3200")]
        
        for price in test_prices:
            # Round price to exchange constraints
            rounded_price = calculator.round_price(price)
            
            # Calculate quantity for this grid level
            raw_quantity = order_size_quote / rounded_price
            rounded_quantity = calculator.round_quantity(raw_quantity)
            
            # Validate the order
            is_valid, message = calculator.validate_order(rounded_price, rounded_quantity)
            
            if rounded_quantity >= exchange_info.min_qty:
                assert is_valid, f"Order should be valid: {message}"
    
    def test_different_asset_types(self):
        """Test sizing calculations for different asset types"""
        # Test cases for different asset characteristics
        test_assets = [
            ("BTCUSDC", Decimal("50000"), Decimal("0.00001")),  # High price, small step
            ("ETHUSDC", Decimal("3000"), Decimal("0.001")),     # Medium price, medium step
            ("ADAUSDC", Decimal("0.5"), Decimal("0.1")),       # Low price, large step
            ("DOGEUSDC", Decimal("0.08"), Decimal("1")),       # Very low price, very large step
        ]
        
        for symbol, price, step_size in test_assets:
            exchange_info = ExchangeInfo(
                symbol=symbol,
                min_qty=step_size,
                max_qty=Decimal("1000000"),
                step_size=step_size,
                tick_size=Decimal("0.0001"),
                min_notional=Decimal("5"),
                base_asset_precision=8,
                quote_asset_precision=4
            )
            
            calculator = SizingCalculator(exchange_info, FeeConfig())
            
            # Test that basic operations work for all asset types
            rounded_price = calculator.round_price(price)
            assert rounded_price > Decimal("0")
            
            max_qty = calculator.calculate_max_quantity(price, Decimal("100"))
            assert max_qty >= Decimal("0")
            
            if max_qty > Decimal("0"):
                is_valid, _ = calculator.validate_order(rounded_price, max_qty)
                # Should be valid or fail due to min_notional (which is expected for low-value orders)
    
    def test_edge_case_handling(self):
        """Test handling of edge cases and boundary conditions"""
        exchange_info = ExchangeInfo(
            symbol="TESTUSDC",
            min_qty=Decimal("0.001"),
            max_qty=Decimal("1000"),
            step_size=Decimal("0.001"),
            tick_size=Decimal("0.01"),
            min_notional=Decimal("10"),
            base_asset_precision=3,
            quote_asset_precision=2
        )
        
        calculator = SizingCalculator(exchange_info, FeeConfig())
        
        # Test with very small balance
        max_qty = calculator.calculate_max_quantity(Decimal("10000"), Decimal("0.001"))
        assert max_qty == Decimal("0")  # Insufficient for min_qty
        
        # Test with very large balance
        max_qty = calculator.calculate_max_quantity(Decimal("1"), Decimal("10000"))
        assert max_qty == exchange_info.max_qty  # Capped at max_qty
        
        # Test edge case prices
        edge_prices = [
            Decimal("0.01"),      # Minimum tick
            Decimal("0.009"),     # Below minimum tick
            Decimal("999999"),    # Very large price
        ]
        
        for price in edge_prices:
            rounded = calculator.round_price(price)
            assert rounded >= Decimal("0")


if __name__ == "__main__":
    pytest.main([__file__])
