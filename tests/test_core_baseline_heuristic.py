"""
Test Suite for Core Module - Baseline Heuristic Component

This module provides comprehensive unit tests for the baseline heuristic calculations
including parameter generation, volatility classification, and fallback mechanisms.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict

# Import the modules under test
from core.baseline_heuristic import (
    BaselineHeuristic,
    get_baseline_parameters
)


class TestBaselineHeuristic:
    """Test BaselineHeuristic class functionality"""
    
    def setup_method(self):
        """Setup BaselineHeuristic instance for each test"""
        self.heuristic = BaselineHeuristic()
    
    def test_baseline_heuristic_initialization(self):
        """Test BaselineHeuristic initializes with correct attributes"""
        assert self.heuristic.name == "ATR-Volatility Baseline"
        assert self.heuristic.version == "1.0"
    
    def test_calculate_grid_parameters_normal_volatility(self):
        """Test grid parameter calculation with normal volatility"""
        symbol = "BTCUSDC"
        current_price = Decimal("50000")
        atr = Decimal("1000")
        volatility = 0.02  # 2% daily volatility
        investment = Decimal("100")
        
        result = self.heuristic.calculate_grid_parameters(
            symbol=symbol,
            current_price=current_price,
            atr=atr,
            volatility=volatility,
            default_investment=investment,
            risk_multiplier=2.0
        )
        
        # Verify basic structure
        assert result["symbol"] == symbol
        assert result["investment_per_grid"] == investment
        assert result["confidence"] == 0.70
        assert result["method"] == "baseline_heuristic"
        
        # Verify bounds calculation (±2*ATR)
        expected_lower = current_price - (atr * Decimal("2.0"))
        expected_upper = current_price + (atr * Decimal("2.0"))
        assert result["lower_bound"] == expected_lower
        assert result["upper_bound"] == expected_upper
        
        # Verify grid count for normal volatility (< 3%)
        assert result["grid_count"] == 12
        
        # Verify spacing calculation
        price_range = expected_upper - expected_lower
        expected_spacing = price_range / Decimal("12")
        assert result["grid_spacing"] == expected_spacing
        
        # Verify reasoning structure
        assert "reasoning" in result
        assert result["reasoning"]["method"] == "ATR-based bounds with volatility-adjusted grid count"
        assert result["reasoning"]["volatility_regime"] == "Low"
        assert result["reasoning"]["risk_level"] == "Conservative"
    
    def test_calculate_grid_parameters_high_volatility(self):
        """Test grid parameter calculation with high volatility"""
        symbol = "ETHUSDC"
        current_price = Decimal("3000")
        atr = Decimal("150")
        volatility = 0.06  # 6% daily volatility (high)
        
        result = self.heuristic.calculate_grid_parameters(
            symbol=symbol,
            current_price=current_price,
            atr=atr,
            volatility=volatility
        )
        
        # High volatility should result in fewer grids
        assert result["grid_count"] == 8
        assert result["reasoning"]["volatility_regime"] == "High"
    
    def test_calculate_grid_parameters_medium_volatility(self):
        """Test grid parameter calculation with medium volatility"""
        current_price = Decimal("1000")
        atr = Decimal("40")
        volatility = 0.04  # 4% daily volatility (medium)
        
        result = self.heuristic.calculate_grid_parameters(
            symbol="ADAUSDC",
            current_price=current_price,
            atr=atr,
            volatility=volatility
        )
        
        # Medium volatility should result in moderate grid count
        assert result["grid_count"] == 10
        assert result["reasoning"]["volatility_regime"] == "Medium"
    
    def test_calculate_grid_parameters_aggressive_risk(self):
        """Test grid parameter calculation with aggressive risk multiplier"""
        current_price = Decimal("100")
        atr = Decimal("5")
        volatility = 0.02
        risk_multiplier = 3.0  # Aggressive
        
        result = self.heuristic.calculate_grid_parameters(
            symbol="LINKUSDC",
            current_price=current_price,
            atr=atr,
            volatility=volatility,
            risk_multiplier=risk_multiplier
        )
        
        # Verify wider bounds with aggressive multiplier
        expected_lower = current_price - (atr * Decimal("3.0"))
        expected_upper = current_price + (atr * Decimal("3.0"))
        assert result["lower_bound"] == expected_lower
        assert result["upper_bound"] == expected_upper
        assert result["reasoning"]["risk_level"] == "Aggressive"
    
    def test_calculate_grid_parameters_lower_bound_protection(self):
        """Test lower bound protection for very low prices"""
        current_price = Decimal("10")  # Low price
        atr = Decimal("15")  # Large ATR relative to price
        volatility = 0.02
        
        result = self.heuristic.calculate_grid_parameters(
            symbol="DOGEUSDC",
            current_price=current_price,
            atr=atr,
            volatility=volatility,
            risk_multiplier=2.0
        )
        
        # Lower bound should be protected (at least 50% of current price)
        min_lower_bound = current_price * Decimal("0.5")
        assert result["lower_bound"] >= min_lower_bound
    
    def test_calculate_grid_parameters_calculation_error(self):
        """Test fallback when calculation fails"""
        with patch.object(self.heuristic, '_fallback_parameters') as mock_fallback:
            mock_fallback.return_value = {"fallback": True}
            
            # Force an error by passing invalid types
            result = self.heuristic.calculate_grid_parameters(
                symbol="INVALID",
                current_price=None,  # Invalid
                atr=Decimal("100"),
                volatility=0.02
            )
            
            mock_fallback.assert_called_once()
            assert result == {"fallback": True}
    
    def test_classify_volatility_low(self):
        """Test volatility classification for low volatility"""
        result = self.heuristic._classify_volatility(0.01)  # 1%
        assert result == "Low"
        
        result = self.heuristic._classify_volatility(0.025)  # 2.5%
        assert result == "Low"
    
    def test_classify_volatility_medium(self):
        """Test volatility classification for medium volatility"""
        result = self.heuristic._classify_volatility(0.035)  # 3.5%
        assert result == "Medium"
        
        result = self.heuristic._classify_volatility(0.045)  # 4.5%
        assert result == "Medium"
    
    def test_classify_volatility_high(self):
        """Test volatility classification for high volatility"""
        result = self.heuristic._classify_volatility(0.06)  # 6%
        assert result == "High"
        
        result = self.heuristic._classify_volatility(0.10)  # 10%
        assert result == "High"
    
    def test_classify_volatility_edge_cases(self):
        """Test volatility classification edge cases"""
        # Exactly at thresholds
        result = self.heuristic._classify_volatility(0.03)  # Exactly 3%
        assert result == "Low"
        
        result = self.heuristic._classify_volatility(0.05)  # Exactly 5%
        assert result == "Medium"
        
        # Extreme values
        result = self.heuristic._classify_volatility(0.0)  # 0%
        assert result == "Low"
        
        result = self.heuristic._classify_volatility(1.0)  # 100%
        assert result == "High"
    
    def test_fallback_parameters_structure(self):
        """Test fallback parameters structure and values"""
        symbol = "TESTUSDC"
        current_price = Decimal("1000")
        investment = Decimal("50")
        
        result = self.heuristic._fallback_parameters(symbol, current_price, investment)
        
        # Verify basic structure
        assert result["symbol"] == symbol
        assert result["investment_per_grid"] == investment
        assert result["method"] == "fallback_conservative"
        assert result["confidence"] == 0.50  # Lower confidence for fallback
        
        # Verify conservative bounds (±10%)
        expected_lower = current_price * Decimal("0.9")
        expected_upper = current_price * Decimal("1.1")
        assert result["lower_bound"] == expected_lower
        assert result["upper_bound"] == expected_upper
        
        # Verify conservative grid count
        assert result["grid_count"] == 10
        
        # Verify total investment calculation
        assert result["total_investment"] == investment * 10
    
    def test_fallback_parameters_timestamp(self):
        """Test fallback parameters includes current timestamp"""
        result = self.heuristic._fallback_parameters("TEST", Decimal("100"), Decimal("10"))
        
        assert "timestamp" in result
        assert isinstance(result["timestamp"], datetime)
        # Should be recent (within last minute)
        assert (datetime.now(timezone.utc) - result["timestamp"]) < timedelta(minutes=1)


class TestGetBaselineParameters:
    """Test standalone get_baseline_parameters function"""
    
    def test_get_baseline_parameters_complete_data(self):
        """Test get_baseline_parameters with complete market data"""
        symbol = "BTCUSDC"
        market_data = {
            "current_price": 50000.0,
            "atr": 1000.0,
            "volatility": 0.03,
            "price_change_24h": 2.5,
            "volume_24h": 1000000.0
        }
        investment = Decimal("200")
        
        result = get_baseline_parameters(symbol, market_data, investment)
        
        # Verify function delegates to BaselineHeuristic
        assert result["symbol"] == symbol
        assert result["investment_per_grid"] == investment
        assert result["method"] == "baseline_heuristic"
        assert "reasoning" in result
    
    def test_get_baseline_parameters_minimal_data(self):
        """Test get_baseline_parameters with minimal market data"""
        symbol = "ETHUSDC"
        market_data = {
            "current_price": 3000.0
            # Missing ATR and volatility
        }
        
        result = get_baseline_parameters(symbol, market_data)
        
        # Should use default fallback values
        assert result["symbol"] == symbol
        assert result["investment_per_grid"] == Decimal("100")  # Default
        assert "reasoning" in result
    
    def test_get_baseline_parameters_missing_price(self):
        """Test get_baseline_parameters with missing current price"""
        symbol = "ADAUSDC"
        market_data = {
            "atr": 0.05,
            "volatility": 0.02
            # Missing current_price
        }
        
        result = get_baseline_parameters(symbol, market_data)
        
        # Should handle missing price gracefully
        assert result["symbol"] == symbol
        assert "confidence" in result
    
    def test_get_baseline_parameters_invalid_data_types(self):
        """Test get_baseline_parameters with invalid data types"""
        symbol = "LINKUSDC"
        market_data = {
            "current_price": "invalid",  # String instead of number
            "atr": None,
            "volatility": "high"
        }
        
        result = get_baseline_parameters(symbol, market_data)
        
        # Should handle invalid data gracefully
        assert result["symbol"] == symbol
        assert "method" in result
    
    def test_get_baseline_parameters_empty_data(self):
        """Test get_baseline_parameters with empty market data"""
        symbol = "DOGEUSDC"
        market_data = {}
        
        result = get_baseline_parameters(symbol, market_data)
        
        # Should still return valid parameters
        assert result["symbol"] == symbol
        assert "lower_bound" in result
        assert "upper_bound" in result
        assert "grid_count" in result
    
    def test_get_baseline_parameters_extreme_values(self):
        """Test get_baseline_parameters with extreme market values"""
        symbol = "TESTUSDC"
        market_data = {
            "current_price": 0.000001,  # Very small price
            "atr": 1000000.0,  # Very large ATR
            "volatility": 2.0  # 200% volatility
        }
        
        result = get_baseline_parameters(symbol, market_data)
        
        # Should handle extreme values without crashing
        assert result["symbol"] == symbol
        assert result["lower_bound"] > 0  # Should be positive
        assert result["upper_bound"] > result["lower_bound"]  # Should be valid range
    
    def test_get_baseline_parameters_default_investment(self):
        """Test get_baseline_parameters uses default investment when not specified"""
        symbol = "BTCUSDC"
        market_data = {"current_price": 50000.0}
        
        result = get_baseline_parameters(symbol, market_data)
        
        assert result["investment_per_grid"] == Decimal("100")  # Default value
    
    def test_get_baseline_parameters_custom_investment(self):
        """Test get_baseline_parameters uses custom investment when specified"""
        symbol = "BTCUSDC"
        market_data = {"current_price": 50000.0}
        custom_investment = Decimal("500")
        
        result = get_baseline_parameters(symbol, market_data, custom_investment)
        
        assert result["investment_per_grid"] == custom_investment


class TestParameterValidation:
    """Test parameter validation and edge cases"""
    
    def setup_method(self):
        """Setup BaselineHeuristic instance for each test"""
        self.heuristic = BaselineHeuristic()
    
    def test_calculate_grid_parameters_zero_atr(self):
        """Test handling of zero ATR"""
        result = self.heuristic.calculate_grid_parameters(
            symbol="STABLEUSDC",
            current_price=Decimal("1.0"),
            atr=Decimal("0"),  # Zero ATR
            volatility=0.001  # Very low volatility
        )
        
        # Should still generate valid bounds
        assert result["lower_bound"] > 0
        assert result["upper_bound"] > result["lower_bound"]
    
    def test_calculate_grid_parameters_negative_values(self):
        """Test handling of negative input values"""
        # Should gracefully handle or use fallback
        result = self.heuristic.calculate_grid_parameters(
            symbol="TESTUSDC",
            current_price=Decimal("100"),
            atr=Decimal("-10"),  # Negative ATR
            volatility=-0.01  # Negative volatility
        )
        
        # Should either handle gracefully or use fallback
        assert "symbol" in result
        assert "method" in result
    
    def test_calculate_grid_parameters_very_high_risk_multiplier(self):
        """Test with very high risk multiplier"""
        result = self.heuristic.calculate_grid_parameters(
            symbol="VOLATILEUSDC",
            current_price=Decimal("100"),
            atr=Decimal("20"),
            volatility=0.02,
            risk_multiplier=10.0  # Very high
        )
        
        # Should handle extreme multipliers
        assert result["lower_bound"] >= 0  # Should not go negative due to protection
        assert result["upper_bound"] > result["lower_bound"]
    
    def test_calculate_grid_parameters_precision_handling(self):
        """Test decimal precision handling"""
        result = self.heuristic.calculate_grid_parameters(
            symbol="PRECISUSDC",
            current_price=Decimal("1.123456789"),
            atr=Decimal("0.000123456"),
            volatility=0.00123456,
            default_investment=Decimal("123.456789")
        )
        
        # Should handle high precision values
        assert isinstance(result["lower_bound"], Decimal)
        assert isinstance(result["upper_bound"], Decimal)
        assert isinstance(result["grid_spacing"], Decimal)


class TestIntegrationScenarios:
    """Test integration scenarios and complex workflows"""
    
    def test_parameter_generation_workflow(self):
        """Test complete parameter generation workflow"""
        # Test data representing real market conditions
        test_cases = [
            {
                "symbol": "BTCUSDC",
                "current_price": 45000.0,
                "atr": 2000.0,
                "volatility": 0.04,
                "investment": 100.0
            },
            {
                "symbol": "ETHUSDC", 
                "current_price": 2800.0,
                "atr": 150.0,
                "volatility": 0.035,
                "investment": 50.0
            },
            {
                "symbol": "ADAUSDC",
                "current_price": 0.85,
                "atr": 0.05,
                "volatility": 0.06,
                "investment": 25.0
            }
        ]
        
        for case in test_cases:
            market_data = {
                "current_price": case["current_price"],
                "atr": case["atr"],
                "volatility": case["volatility"]
            }
            
            result = get_baseline_parameters(
                case["symbol"], 
                market_data, 
                Decimal(str(case["investment"]))
            )
            
            # Verify all results are valid
            assert result["symbol"] == case["symbol"]
            assert result["lower_bound"] > 0
            assert result["upper_bound"] > result["lower_bound"]
            assert result["grid_count"] > 0
            assert result["investment_per_grid"] > 0
            assert 0 < result["confidence"] <= 1
            assert "reasoning" in result
    
    def test_parameter_consistency(self):
        """Test parameter calculation consistency"""
        # Same inputs should produce same outputs
        symbol = "CONSISTUSDC"
        market_data = {
            "current_price": 1000.0,
            "atr": 50.0,
            "volatility": 0.03
        }
        investment = Decimal("75")
        
        result1 = get_baseline_parameters(symbol, market_data, investment)
        result2 = get_baseline_parameters(symbol, market_data, investment)
        
        # Results should be identical (excluding timestamp)
        for key in ["symbol", "lower_bound", "upper_bound", "grid_count", "investment_per_grid"]:
            assert result1[key] == result2[key]
    
    def test_volatility_impact_on_grid_count(self):
        """Test that volatility properly impacts grid count"""
        base_params = {
            "symbol": "TESTUSDC",
            "current_price": 1000.0,
            "atr": 50.0
        }
        
        # Test low, medium, and high volatility
        low_vol_result = get_baseline_parameters("TEST", {**base_params, "volatility": 0.01})
        med_vol_result = get_baseline_parameters("TEST", {**base_params, "volatility": 0.04})
        high_vol_result = get_baseline_parameters("TEST", {**base_params, "volatility": 0.08})
        
        # Higher volatility should result in fewer grids
        assert high_vol_result["grid_count"] < med_vol_result["grid_count"]
        assert med_vol_result["grid_count"] < low_vol_result["grid_count"]


if __name__ == "__main__":
    pytest.main([__file__])
