"""
Baseline Heuristic - Simple parameter calculation fallback

This module provides baseline heuristic methods for calculating grid trading
parameters when AI suggestions are not available or fail.
"""

from typing import Dict, Tuple
from decimal import Decimal
from datetime import datetime, timezone, timedelta
import logging

logger = logging.getLogger(__name__)


class BaselineHeuristic:
    """
    Simple, transparent baseline for grid parameter calculation

    Uses market volatility and price action to suggest reasonable
    grid bounds and spacing without complex ML models.
    """

    def __init__(self):
        self.name = "ATR-Volatility Baseline"
        self.version = "1.0"

    def calculate_grid_parameters(
        self,
        symbol: str,
        current_price: Decimal,
        atr: Decimal,
        volatility: float,
        default_investment: Decimal = Decimal("100"),
        risk_multiplier: float = 2.0,
    ) -> Dict:
        """
        Calculate baseline grid parameters using ATR and volatility

        Args:
            symbol: Trading symbol
            current_price: Current market price
            atr: Average True Range (volatility measure)
            volatility: Daily volatility percentage (0.02 = 2%)
            default_investment: Default investment per grid level
            risk_multiplier: Risk multiplier for bounds (2.0 = ±2*ATR)

        Returns:
            Dictionary with suggested parameters and reasoning
        """
        try:
            # Calculate bounds using ATR
            atr_bound = atr * Decimal(str(risk_multiplier))
            lower_bound = current_price - atr_bound
            upper_bound = current_price + atr_bound

            # Ensure positive bounds
            lower_bound = max(lower_bound, current_price * Decimal("0.5"))

            # Calculate grid count based on volatility
            # Higher volatility = fewer grids to avoid overtrading
            if volatility > 0.05:  # > 5% daily volatility
                grid_count = 8
            elif volatility > 0.03:  # 3-5% volatility
                grid_count = 10
            else:  # < 3% volatility
                grid_count = 12

            # Calculate spacing
            price_range = upper_bound - lower_bound
            grid_spacing = price_range / Decimal(str(grid_count))

            # Reasoning for transparency
            reasoning = {
                "method": "ATR-based bounds with volatility-adjusted grid count",
                "bounds_calculation": f"Current price ± {risk_multiplier}×ATR",
                "atr_value": float(atr),
                "volatility_regime": self._classify_volatility(volatility),
                "grid_count_reason": f"Adjusted for {volatility:.1%} daily volatility",
                "risk_level": (
                    "Conservative" if risk_multiplier <= 2.0 else "Aggressive"
                ),
            }

            return {
                "symbol": symbol,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "grid_count": grid_count,
                "grid_spacing": grid_spacing,
                "investment_per_grid": default_investment,
                "total_investment": default_investment * grid_count,
                "confidence": 0.70,  # Baseline confidence
                "reasoning": reasoning,
                "timestamp": datetime.now(timezone.utc),
                "method": "baseline_heuristic",
            }

        except Exception as e:
            logger.error(f"Error calculating baseline parameters: {e}")
            return self._fallback_parameters(symbol, current_price, default_investment)

    def _classify_volatility(self, volatility: float) -> str:
        """Classify volatility regime"""
        if volatility > 0.08:
            return "Very High (>8%)"
        elif volatility > 0.05:
            return "High (5-8%)"
        elif volatility > 0.03:
            return "Medium (3-5%)"
        elif volatility > 0.01:
            return "Low (1-3%)"
        else:
            return "Very Low (<1%)"

    def _fallback_parameters(
        self, symbol: str, current_price: Decimal, investment: Decimal
    ) -> Dict:
        """Ultra-conservative fallback when all else fails"""
        # Very conservative: ±5% bounds, 10 grids
        bound_pct = Decimal("0.05")
        lower_bound = current_price * (Decimal("1") - bound_pct)
        upper_bound = current_price * (Decimal("1") + bound_pct)

        return {
            "symbol": symbol,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "grid_count": 10,
            "grid_spacing": (upper_bound - lower_bound) / 10,
            "investment_per_grid": investment,
            "total_investment": investment * 10,
            "confidence": 0.50,  # Low confidence fallback
            "reasoning": {
                "method": "Emergency fallback - fixed 5% bounds",
                "note": "Used when market data unavailable",
            },
            "timestamp": datetime.now(timezone.utc),
            "method": "emergency_fallback",
        }


def get_baseline_parameters(
    symbol: str, market_data: Dict, investment: Decimal = Decimal("100")
) -> Dict:
    """
    Convenience function to get baseline parameters

    Args:
        symbol: Trading symbol
        market_data: Dictionary with market data (current_price, atr, volatility)
        investment: Investment per grid level

    Returns:
        Baseline parameter suggestion
    """
    baseline = BaselineHeuristic()

    current_price = market_data.get("current_price", Decimal("50000"))
    atr = market_data.get("atr", current_price * Decimal("0.02"))  # 2% default ATR
    volatility = market_data.get("volatility", 0.03)  # 3% default volatility

    return baseline.calculate_grid_parameters(
        symbol=symbol,
        current_price=current_price,
        atr=atr,
        volatility=volatility,
        default_investment=investment,
    )
