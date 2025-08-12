"""
Sizing - Tick/lot rounding, min notional checks, fees

This module handles position sizing calculations, exchange constraints,
and fee calculations for trading operations.
"""

from typing import Dict, Optional, Tuple
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from dataclasses import dataclass
import math

@dataclass
class ExchangeInfo:
    """Exchange trading rules and constraints"""
    symbol: str
    min_qty: Decimal
    max_qty: Decimal
    step_size: Decimal
    tick_size: Decimal
    min_notional: Decimal
    base_asset_precision: int
    quote_asset_precision: int

@dataclass
class FeeConfig:
    """Fee structure configuration"""
    maker_fee: Decimal = Decimal('0.001')  # 0.1%
    taker_fee: Decimal = Decimal('0.001')  # 0.1%
    bnb_discount: Decimal = Decimal('0.75')  # 25% discount with BNB
    use_bnb: bool = False

class SizingCalculator:
    """
    Handles all position sizing calculations including:
    - Tick and lot size rounding
    - Minimum notional value checks
    - Fee calculations
    - Position size validation
    """
    
    def __init__(self, exchange_info: ExchangeInfo, fee_config: FeeConfig):
        self.exchange_info = exchange_info
        self.fee_config = fee_config
        
    def round_price(self, price: Decimal, round_up: bool = False) -> Decimal:
        """Round price to exchange tick size"""
        if self.exchange_info.tick_size == 0:
            return price
            
        if round_up:
            return (price / self.exchange_info.tick_size).quantize(
                Decimal('1'), rounding=ROUND_UP
            ) * self.exchange_info.tick_size
        else:
            return (price / self.exchange_info.tick_size).quantize(
                Decimal('1'), rounding=ROUND_DOWN
            ) * self.exchange_info.tick_size
    
    def round_quantity(self, quantity: Decimal, round_up: bool = False) -> Decimal:
        """Round quantity to exchange step size"""
        if self.exchange_info.step_size == 0:
            return quantity
            
        if round_up:
            return (quantity / self.exchange_info.step_size).quantize(
                Decimal('1'), rounding=ROUND_UP
            ) * self.exchange_info.step_size
        else:
            return (quantity / self.exchange_info.step_size).quantize(
                Decimal('1'), rounding=ROUND_DOWN
            ) * self.exchange_info.step_size
    
    def calculate_max_quantity(self, price: Decimal, available_balance: Decimal) -> Decimal:
        """Calculate maximum quantity that can be bought with available balance"""
        # Account for fees
        effective_fee = self.get_effective_fee_rate(is_maker=True)
        fee_multiplier = Decimal('1') + effective_fee
        
        # Calculate raw quantity
        raw_quantity = available_balance / (price * fee_multiplier)
        
        # Round down to step size
        max_quantity = self.round_quantity(raw_quantity, round_up=False)
        
        # Ensure minimum quantity
        if max_quantity < self.exchange_info.min_qty:
            return Decimal('0')
            
        # Ensure maximum quantity
        if max_quantity > self.exchange_info.max_qty:
            max_quantity = self.exchange_info.max_qty
            
        return max_quantity
    
    def validate_order(self, price: Decimal, quantity: Decimal) -> Tuple[bool, str]:
        """
        Validate if order meets exchange requirements
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check minimum quantity
        if quantity < self.exchange_info.min_qty:
            return False, f"Quantity {quantity} below minimum {self.exchange_info.min_qty}"
            
        # Check maximum quantity
        if quantity > self.exchange_info.max_qty:
            return False, f"Quantity {quantity} above maximum {self.exchange_info.max_qty}"
            
        # Check step size
        remainder = quantity % self.exchange_info.step_size
        if remainder != 0:
            return False, f"Quantity {quantity} not aligned to step size {self.exchange_info.step_size}"
            
        # Check tick size
        remainder = price % self.exchange_info.tick_size
        if remainder != 0:
            return False, f"Price {price} not aligned to tick size {self.exchange_info.tick_size}"
            
        # Check minimum notional
        notional = price * quantity
        if notional < self.exchange_info.min_notional:
            return False, f"Notional {notional} below minimum {self.exchange_info.min_notional}"
            
        return True, ""
    
    def calculate_fees(self, price: Decimal, quantity: Decimal, is_maker: bool = True) -> Dict[str, Decimal]:
        """
        Calculate trading fees for an order
        
        Returns:
            Dictionary with fee breakdown
        """
        notional = price * quantity
        base_fee_rate = self.fee_config.maker_fee if is_maker else self.fee_config.taker_fee
        
        if self.fee_config.use_bnb:
            effective_fee_rate = base_fee_rate * self.fee_config.bnb_discount
        else:
            effective_fee_rate = base_fee_rate
            
        fee_amount = notional * effective_fee_rate
        
        return {
            "notional": notional,
            "base_fee_rate": base_fee_rate,
            "effective_fee_rate": effective_fee_rate,
            "fee_amount": fee_amount,
            "net_amount": notional - fee_amount,
            "uses_bnb": self.fee_config.use_bnb
        }
    
    def get_effective_fee_rate(self, is_maker: bool = True) -> Decimal:
        """Get the effective fee rate after discounts"""
        base_rate = self.fee_config.maker_fee if is_maker else self.fee_config.taker_fee
        
        if self.fee_config.use_bnb:
            return base_rate * self.fee_config.bnb_discount
        return base_rate
    
    def calculate_grid_order_size(self, total_capital: Decimal, num_levels: int, 
                                  price_range_pct: Decimal = Decimal('0.1')) -> Decimal:
        """
        Calculate optimal order size for grid trading
        
        Args:
            total_capital: Total available capital
            num_levels: Number of grid levels
            price_range_pct: Expected price range as percentage (default 10%)
        """
        # Reserve some capital for price movements
        usable_capital = total_capital * (Decimal('1') - price_range_pct)
        
        # Divide among grid levels
        raw_order_size = usable_capital / num_levels
        
        # Account for fees (assuming average 2 trades per level - buy then sell)
        fee_multiplier = Decimal('1') + (self.get_effective_fee_rate() * 2)
        
        return raw_order_size / fee_multiplier
    
    def calculate_position_value(self, quantity: Decimal, current_price: Decimal) -> Dict[str, Decimal]:
        """Calculate current position value and unrealized PnL"""
        position_value = quantity * current_price
        
        # For grid trading, we'd need entry price to calculate PnL
        # This is a simplified version
        return {
            "quantity": quantity,
            "current_price": current_price,
            "position_value": position_value,
            "notional": position_value
        }

def create_binance_exchange_info(symbol: str, price: Decimal) -> ExchangeInfo:
    """
    Create ExchangeInfo with typical Binance constraints
    Note: In production, this should be fetched from the API
    """
    # Common Binance constraints (these should be fetched from API)
    if "USDC" in symbol.upper():
        return ExchangeInfo(
            symbol=symbol,
            min_qty=Decimal('0.00001'),
            max_qty=Decimal('9000000'),
            step_size=Decimal('0.00001'),
            tick_size=Decimal('0.01'),
            min_notional=Decimal('10.0'),
            base_asset_precision=5,
            quote_asset_precision=2
        )
    else:
        # Default constraints
        return ExchangeInfo(
            symbol=symbol,
            min_qty=Decimal('0.001'),
            max_qty=Decimal('100000'),
            step_size=Decimal('0.001'),
            tick_size=Decimal('0.01'),
            min_notional=Decimal('10.0'),
            base_asset_precision=3,
            quote_asset_precision=2
        )

def ensure_min_notional(price: Decimal, qty: Decimal, min_notional: Decimal) -> bool:
    """
    Helper function to check if an order meets minimum notional requirements.
    
    Args:
        price: Order price
        qty: Order quantity  
        min_notional: Minimum notional value required by exchange
        
    Returns:
        True if price * qty >= min_notional, False otherwise
    """
    return price * qty >= min_notional

# Convenience function with float inputs for easier integration
def ensure_min_notional_float(price: float, qty: float, min_notional: float) -> bool:
    """
    Helper function to check minimum notional requirements with float inputs.
    
    Args:
        price: Order price (float)
        qty: Order quantity (float)
        min_notional: Minimum notional value required by exchange (float)
        
    Returns:
        True if price * qty >= min_notional, False otherwise
    """
    return price * qty >= min_notional
