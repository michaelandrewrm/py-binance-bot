"""
Slippage - Pluggable slippage models

This module provides various slippage models for realistic trade simulation
in backtesting and paper trading environments.
"""

from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math
import random

from core.state import Order, OrderSide, OrderType
from data.schema import KlineData, DepthData, TradeData


@dataclass
class SlippageContext:
    """Context information for slippage calculation"""

    order: Order
    market_price: Decimal
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    volume: Optional[Decimal] = None
    recent_trades: Optional[List[TradeData]] = None
    order_book: Optional[DepthData] = None
    volatility: Optional[Decimal] = None


class SlippageModel(ABC):
    """Base class for slippage models"""

    @abstractmethod
    def calculate_slippage(self, context: SlippageContext) -> Decimal:
        """
        Calculate slippage as a percentage of price

        Args:
            context: SlippageContext with order and market information

        Returns:
            Slippage as decimal (e.g., 0.001 for 0.1% slippage)
            Positive values increase execution price for buys, decrease for sells
        """
        pass

    def get_execution_price(self, context: SlippageContext) -> Decimal:
        """
        Get final execution price including slippage

        Args:
            context: SlippageContext with order and market information

        Returns:
            Final execution price
        """
        slippage = self.calculate_slippage(context)

        if context.order.side == OrderSide.BUY:
            # Positive slippage hurts buyers (higher price)
            return context.market_price * (Decimal("1") + slippage)
        else:
            # Positive slippage hurts sellers (lower price)
            return context.market_price * (Decimal("1") - slippage)


class NoSlippageModel(SlippageModel):
    """No slippage model - perfect execution"""

    def calculate_slippage(self, context: SlippageContext) -> Decimal:
        return Decimal("0")


class FixedSlippageModel(SlippageModel):
    """Fixed slippage model - constant percentage"""

    def __init__(self, slippage_bps: Decimal = Decimal("5")):
        """
        Args:
            slippage_bps: Fixed slippage in basis points (default 5 bps = 0.05%)
        """
        self.slippage_rate = slippage_bps / Decimal("10000")

    def calculate_slippage(self, context: SlippageContext) -> Decimal:
        return self.slippage_rate


class LinearSlippageModel(SlippageModel):
    """Linear slippage model based on order size"""

    def __init__(
        self,
        base_slippage_bps: Decimal = Decimal("2"),
        size_impact_factor: Decimal = Decimal("0.1"),
        max_slippage_bps: Decimal = Decimal("50"),
    ):
        """
        Args:
            base_slippage_bps: Base slippage in basis points
            size_impact_factor: Additional slippage per $1000 order value
            max_slippage_bps: Maximum slippage cap in basis points
        """
        self.base_slippage = base_slippage_bps / Decimal("10000")
        self.size_impact_factor = size_impact_factor / Decimal("100")
        self.max_slippage = max_slippage_bps / Decimal("10000")

    def calculate_slippage(self, context: SlippageContext) -> Decimal:
        # Base slippage
        base_slippage = self.base_slippage

        # Size-based impact
        order_value = context.order.quantity * context.market_price
        size_factor = order_value / Decimal("1000")  # Per $1000
        size_slippage = size_factor * self.size_impact_factor / Decimal("100")

        # Total slippage with cap
        total_slippage = base_slippage + size_slippage
        return min(total_slippage, self.max_slippage)


class SquareRootSlippageModel(SlippageModel):
    """Square root slippage model - more realistic for large orders"""

    def __init__(
        self,
        base_slippage_bps: Decimal = Decimal("3"),
        impact_coefficient: Decimal = Decimal("0.01"),
        max_slippage_bps: Decimal = Decimal("100"),
    ):
        """
        Args:
            base_slippage_bps: Base slippage in basis points
            impact_coefficient: Coefficient for square root impact
            max_slippage_bps: Maximum slippage cap in basis points
        """
        self.base_slippage = base_slippage_bps / Decimal("10000")
        self.impact_coefficient = impact_coefficient
        self.max_slippage = max_slippage_bps / Decimal("10000")

    def calculate_slippage(self, context: SlippageContext) -> Decimal:
        base_slippage = self.base_slippage

        # Square root impact based on order size
        order_value = context.order.quantity * context.market_price
        sqrt_impact = Decimal(str(math.sqrt(float(order_value / Decimal("10000")))))
        size_slippage = self.impact_coefficient * sqrt_impact / Decimal("100")

        total_slippage = base_slippage + size_slippage
        return min(total_slippage, self.max_slippage)


class VolatilityBasedSlippageModel(SlippageModel):
    """Slippage model that adjusts based on market volatility"""

    def __init__(
        self,
        base_slippage_bps: Decimal = Decimal("2"),
        volatility_multiplier: Decimal = Decimal("10"),
        size_impact_factor: Decimal = Decimal("0.05"),
    ):
        """
        Args:
            base_slippage_bps: Base slippage in basis points
            volatility_multiplier: Multiplier for volatility impact
            size_impact_factor: Size impact factor
        """
        self.base_slippage = base_slippage_bps / Decimal("10000")
        self.volatility_multiplier = volatility_multiplier
        self.size_impact_factor = size_impact_factor / Decimal("100")

    def calculate_slippage(self, context: SlippageContext) -> Decimal:
        base_slippage = self.base_slippage

        # Volatility impact
        volatility_slippage = Decimal("0")
        if context.volatility is not None:
            volatility_slippage = (
                context.volatility * self.volatility_multiplier / Decimal("100")
            )

        # Size impact
        order_value = context.order.quantity * context.market_price
        size_factor = order_value / Decimal("10000")
        size_slippage = size_factor * self.size_impact_factor / Decimal("100")

        return base_slippage + volatility_slippage + size_slippage


class BidAskSlippageModel(SlippageModel):
    """Slippage model based on bid-ask spread"""

    def __init__(
        self,
        spread_factor: Decimal = Decimal("0.5"),
        min_slippage_bps: Decimal = Decimal("1"),
        max_slippage_bps: Decimal = Decimal("50"),
    ):
        """
        Args:
            spread_factor: Fraction of spread to use as slippage
            min_slippage_bps: Minimum slippage in basis points
            max_slippage_bps: Maximum slippage in basis points
        """
        self.spread_factor = spread_factor
        self.min_slippage = min_slippage_bps / Decimal("10000")
        self.max_slippage = max_slippage_bps / Decimal("10000")

    def calculate_slippage(self, context: SlippageContext) -> Decimal:
        if context.bid is None or context.ask is None:
            return self.min_slippage

        # Calculate spread as percentage of mid price
        spread = context.ask - context.bid
        mid_price = (context.bid + context.ask) / Decimal("2")
        spread_pct = spread / mid_price

        # Slippage as fraction of spread
        slippage = spread_pct * self.spread_factor

        # Apply bounds
        return max(self.min_slippage, min(slippage, self.max_slippage))


class OrderBookSlippageModel(SlippageModel):
    """Advanced slippage model using order book depth"""

    def __init__(
        self,
        impact_factor: Decimal = Decimal("0.1"),
        base_slippage_bps: Decimal = Decimal("1"),
    ):
        """
        Args:
            impact_factor: Impact factor for order book consumption
            base_slippage_bps: Base slippage in basis points
        """
        self.impact_factor = impact_factor
        self.base_slippage = base_slippage_bps / Decimal("10000")

    def calculate_slippage(self, context: SlippageContext) -> Decimal:
        if context.order_book is None:
            return self.base_slippage

        order = context.order

        # Get relevant side of order book
        if order.side == OrderSide.BUY:
            levels = context.order_book.asks
        else:
            levels = context.order_book.bids

        if not levels:
            return self.base_slippage

        # Calculate how much of the order book we would consume
        remaining_qty = order.quantity
        total_cost = Decimal("0")

        for level in levels:
            if remaining_qty <= 0:
                break

            consumed_qty = min(remaining_qty, level.quantity)
            total_cost += consumed_qty * level.price
            remaining_qty -= consumed_qty

        if remaining_qty > 0:
            # Order would exhaust visible liquidity
            return self.base_slippage * Decimal("10")  # High penalty

        # Calculate average execution price
        avg_price = total_cost / order.quantity

        # Slippage relative to best price
        best_price = levels[0].price
        slippage = abs(avg_price - best_price) / best_price

        return slippage


class RandomSlippageModel(SlippageModel):
    """Random slippage model for stress testing"""

    def __init__(
        self,
        min_slippage_bps: Decimal = Decimal("0"),
        max_slippage_bps: Decimal = Decimal("20"),
        distribution: str = "uniform",
    ):
        """
        Args:
            min_slippage_bps: Minimum slippage in basis points
            max_slippage_bps: Maximum slippage in basis points
            distribution: Random distribution ("uniform" or "normal")
        """
        self.min_slippage = min_slippage_bps / Decimal("10000")
        self.max_slippage = max_slippage_bps / Decimal("10000")
        self.distribution = distribution

    def calculate_slippage(self, context: SlippageContext) -> Decimal:
        if self.distribution == "uniform":
            # Uniform distribution
            range_size = float(self.max_slippage - self.min_slippage)
            random_factor = random.uniform(0, 1)
            slippage = self.min_slippage + Decimal(str(random_factor * range_size))
        else:
            # Normal distribution
            mean = float(self.min_slippage + self.max_slippage) / 2
            std_dev = float(self.max_slippage - self.min_slippage) / 4
            random_value = random.normalvariate(mean, std_dev)
            slippage = Decimal(
                str(
                    max(
                        float(self.min_slippage),
                        min(float(self.max_slippage), random_value),
                    )
                )
            )

        return slippage


class CompositeSlippageModel(SlippageModel):
    """Composite model that combines multiple slippage models"""

    def __init__(self, models: List[Tuple[SlippageModel, Decimal]]):
        """
        Args:
            models: List of (model, weight) tuples where weights sum to 1.0
        """
        self.models = models

        # Validate weights
        total_weight = sum(weight for _, weight in models)
        if abs(total_weight - Decimal("1.0")) > Decimal("0.001"):
            raise ValueError(f"Model weights must sum to 1.0, got {total_weight}")

    def calculate_slippage(self, context: SlippageContext) -> Decimal:
        total_slippage = Decimal("0")

        for model, weight in self.models:
            model_slippage = model.calculate_slippage(context)
            total_slippage += model_slippage * weight

        return total_slippage


class AdaptiveSlippageModel(SlippageModel):
    """Adaptive model that adjusts based on market conditions"""

    def __init__(
        self,
        base_model: SlippageModel,
        volatility_threshold: Decimal = Decimal("0.02"),
        high_vol_multiplier: Decimal = Decimal("2.0"),
        volume_threshold: Decimal = Decimal("1000000"),
    ):
        """
        Args:
            base_model: Base slippage model
            volatility_threshold: Volatility threshold for adaptation
            high_vol_multiplier: Multiplier for high volatility periods
            volume_threshold: Volume threshold for liquidity assessment
        """
        self.base_model = base_model
        self.volatility_threshold = volatility_threshold
        self.high_vol_multiplier = high_vol_multiplier
        self.volume_threshold = volume_threshold

    def calculate_slippage(self, context: SlippageContext) -> Decimal:
        base_slippage = self.base_model.calculate_slippage(context)

        # Adjust for high volatility
        if context.volatility and context.volatility > self.volatility_threshold:
            base_slippage *= self.high_vol_multiplier

        # Adjust for low liquidity
        if context.volume and context.volume < self.volume_threshold:
            liquidity_factor = self.volume_threshold / context.volume
            base_slippage *= min(liquidity_factor, Decimal("5.0"))  # Cap at 5x

        return base_slippage


# Utility functions for creating common slippage models


def create_conservative_slippage_model() -> SlippageModel:
    """Create a conservative slippage model for backtesting"""
    return LinearSlippageModel(
        base_slippage_bps=Decimal("5"),
        size_impact_factor=Decimal("0.2"),
        max_slippage_bps=Decimal("100"),
    )


def create_realistic_slippage_model() -> SlippageModel:
    """Create a realistic slippage model"""
    linear_model = LinearSlippageModel(
        base_slippage_bps=Decimal("3"),
        size_impact_factor=Decimal("0.1"),
        max_slippage_bps=Decimal("50"),
    )

    volatility_model = VolatilityBasedSlippageModel(
        base_slippage_bps=Decimal("1"),
        volatility_multiplier=Decimal("5"),
        size_impact_factor=Decimal("0.05"),
    )

    return CompositeSlippageModel(
        [(linear_model, Decimal("0.7")), (volatility_model, Decimal("0.3"))]
    )


def create_aggressive_slippage_model() -> SlippageModel:
    """Create an aggressive slippage model for stress testing"""
    return RandomSlippageModel(
        min_slippage_bps=Decimal("5"),
        max_slippage_bps=Decimal("50"),
        distribution="normal",
    )
