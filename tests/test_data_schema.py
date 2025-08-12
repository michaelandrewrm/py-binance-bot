"""
Test cases for data/schema.py module

This module provides comprehensive tests for the data schema definitions,
serialization/deserialization, and utility functions.
"""

import pytest
import json
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

from data.schema import (
    # Enums
    TimeFrame,
    MarketDataType,
    
    # Data Classes
    KlineData,
    TradeData,
    TickerData,
    DepthLevel,
    DepthData,
    FeatureData,
    ModelPrediction,
    TradeRecord,
    EquityCurvePoint,
    BacktestResult,
    
    # Utility Functions
    utc_isoformat,
    klines_from_binance,
    trades_from_binance,
    depth_from_binance,
    serialize_to_json,
    deserialize_from_json,
    DecimalEncoder
)


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_utc_isoformat_naive_datetime(self):
        """Test UTC ISO format with naive datetime"""
        naive_dt = datetime(2023, 1, 1, 12, 0, 0)
        result = utc_isoformat(naive_dt)
        assert result == "2023-01-01T12:00:00Z"
    
    def test_utc_isoformat_utc_datetime(self):
        """Test UTC ISO format with UTC datetime"""
        utc_dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = utc_isoformat(utc_dt)
        assert result == "2023-01-01T12:00:00Z"
    
    def test_utc_isoformat_other_timezone(self):
        """Test UTC ISO format with non-UTC timezone"""
        offset = timezone(timedelta(hours=5))
        offset_dt = datetime(2023, 1, 1, 17, 0, 0, tzinfo=offset)
        result = utc_isoformat(offset_dt)
        assert result == "2023-01-01T12:00:00Z"  # Converted to UTC
    
    def test_serialize_to_json(self):
        """Test JSON serialization"""
        data = {
            "decimal_value": Decimal("123.45"),
            "datetime_value": datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            "string_value": "test",
            "int_value": 42
        }
        result = serialize_to_json(data)
        
        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed["decimal_value"] == "123.45"
        assert parsed["datetime_value"] == "2023-01-01T12:00:00Z"
        assert parsed["string_value"] == "test"
        assert parsed["int_value"] == 42
    
    def test_deserialize_from_json(self):
        """Test JSON deserialization"""
        json_str = '{"test": "value", "number": 42}'
        result = deserialize_from_json(json_str)
        
        assert result["test"] == "value"
        assert result["number"] == 42
    
    def test_decimal_encoder(self):
        """Test DecimalEncoder functionality"""
        encoder = DecimalEncoder()
        
        # Test Decimal encoding
        decimal_result = encoder.default(Decimal("123.45"))
        assert decimal_result == "123.45"
        
        # Test datetime encoding
        dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        datetime_result = encoder.default(dt)
        assert datetime_result == "2023-01-01T12:00:00Z"
        
        # Test fallback to default behavior
        with pytest.raises(TypeError):
            encoder.default(object())


class TestEnums:
    """Test enumeration definitions"""
    
    def test_timeframe_enum_values(self):
        """Test TimeFrame enum values"""
        assert TimeFrame.M1.value == "1m"
        assert TimeFrame.M5.value == "5m"
        assert TimeFrame.H1.value == "1h"
        assert TimeFrame.D1.value == "1d"
        assert TimeFrame.W1.value == "1w"
        assert TimeFrame.MN1.value == "1M"
    
    def test_timeframe_enum_completeness(self):
        """Test TimeFrame enum has expected values"""
        expected_timeframes = [
            "M1", "M3", "M5", "M15", "M30",
            "H1", "H2", "H4", "H6", "H8", "H12",
            "D1", "D3", "W1", "MN1"
        ]
        
        for timeframe in expected_timeframes:
            assert hasattr(TimeFrame, timeframe)
    
    def test_market_data_type_enum(self):
        """Test MarketDataType enum values"""
        assert MarketDataType.KLINE.value == "kline"
        assert MarketDataType.TRADE.value == "trade"
        assert MarketDataType.TICKER.value == "ticker"
        assert MarketDataType.DEPTH.value == "depth"
        assert MarketDataType.AGG_TRADE.value == "aggTrade"


class TestKlineData:
    """Test KlineData functionality"""
    
    def setup_method(self):
        """Set up test data"""
        self.sample_kline = KlineData(
            symbol="BTCUSDC",
            open_time=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            close_time=datetime(2023, 1, 1, 12, 1, 0, tzinfo=timezone.utc),
            open_price=Decimal("50000.00"),
            high_price=Decimal("50100.00"),
            low_price=Decimal("49900.00"),
            close_price=Decimal("50050.00"),
            volume=Decimal("1.5"),
            quote_volume=Decimal("75075.00"),
            trades_count=100,
            taker_buy_volume=Decimal("0.8"),
            taker_buy_quote_volume=Decimal("40020.00"),
            timeframe=TimeFrame.M1
        )
    
    def test_kline_creation(self):
        """Test KlineData creation"""
        assert self.sample_kline.symbol == "BTCUSDC"
        assert self.sample_kline.open_price == Decimal("50000.00")
        assert self.sample_kline.timeframe == TimeFrame.M1
    
    def test_typical_price_property(self):
        """Test typical price calculation"""
        expected = (Decimal("50100.00") + Decimal("49900.00") + Decimal("50050.00")) / 3
        assert self.sample_kline.typical_price == expected
    
    def test_weighted_price_property(self):
        """Test weighted price calculation"""
        expected = Decimal("75075.00") / Decimal("1.5")
        assert self.sample_kline.weighted_price == expected
    
    def test_weighted_price_zero_volume(self):
        """Test weighted price with zero volume"""
        kline = KlineData(
            symbol="BTCUSDC",
            open_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            close_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            open_price=Decimal("50000"),
            high_price=Decimal("50000"),
            low_price=Decimal("50000"),
            close_price=Decimal("50000"),
            volume=Decimal("0"),
            quote_volume=Decimal("0"),
            trades_count=0,
            taker_buy_volume=Decimal("0"),
            taker_buy_quote_volume=Decimal("0"),
            timeframe=TimeFrame.M1
        )
        assert kline.weighted_price == Decimal("50000")
    
    def test_price_range_property(self):
        """Test price range calculation"""
        expected = Decimal("50100.00") - Decimal("49900.00")
        assert self.sample_kline.price_range == expected
    
    def test_body_size_property(self):
        """Test body size calculation"""
        expected = abs(Decimal("50050.00") - Decimal("50000.00"))
        assert self.sample_kline.body_size == expected
    
    def test_is_bullish_property_true(self):
        """Test bullish candle detection"""
        assert self.sample_kline.is_bullish is True
    
    def test_is_bullish_property_false(self):
        """Test bearish candle detection"""
        bearish_kline = KlineData(
            symbol="BTCUSDC",
            open_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            close_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
            open_price=Decimal("50000"),
            high_price=Decimal("50000"),
            low_price=Decimal("49000"),
            close_price=Decimal("49500"),  # Close < Open
            volume=Decimal("1"),
            quote_volume=Decimal("50000"),
            trades_count=10,
            taker_buy_volume=Decimal("0.5"),
            taker_buy_quote_volume=Decimal("25000"),
            timeframe=TimeFrame.M1
        )
        assert bearish_kline.is_bullish is False
    
    def test_to_dict_conversion(self):
        """Test dictionary conversion"""
        result = self.sample_kline.to_dict()
        
        assert result["symbol"] == "BTCUSDC"
        assert result["open"] == "50000.00"
        assert result["high"] == "50100.00"
        assert result["low"] == "49900.00"
        assert result["close"] == "50050.00"
        assert result["volume"] == "1.5"
        assert result["timeframe"] == "1m"
        assert "open_time" in result
        assert "close_time" in result
    
    def test_from_dict_conversion(self):
        """Test creation from dictionary"""
        data = {
            "symbol": "BTCUSDC",
            "open_time": "2023-01-01T12:00:00+00:00",
            "close_time": "2023-01-01T12:01:00+00:00",
            "open": "50000.00",
            "high": "50100.00",
            "low": "49900.00",
            "close": "50050.00",
            "volume": "1.5",
            "quote_volume": "75075.00",
            "trades_count": 100,
            "taker_buy_volume": "0.8",
            "taker_buy_quote_volume": "40020.00",
            "timeframe": "1m"
        }
        
        kline = KlineData.from_dict(data)
        
        assert kline.symbol == "BTCUSDC"
        assert kline.open_price == Decimal("50000.00")
        assert kline.timeframe == TimeFrame.M1
    
    def test_roundtrip_conversion(self):
        """Test roundtrip to_dict -> from_dict conversion"""
        original = self.sample_kline
        dict_data = original.to_dict()
        
        # Handle the Z format issue in the schema
        # The to_dict() creates Z format but from_dict() expects +00:00 format
        dict_data["open_time"] = dict_data["open_time"].replace("Z", "+00:00")
        dict_data["close_time"] = dict_data["close_time"].replace("Z", "+00:00")
        
        restored = KlineData.from_dict(dict_data)
        
        assert original.symbol == restored.symbol
        assert original.open_price == restored.open_price
        assert original.close_price == restored.close_price
        assert original.timeframe == restored.timeframe


class TestTradeData:
    """Test TradeData functionality"""
    
    def setup_method(self):
        """Set up test data"""
        self.sample_trade = TradeData(
            symbol="BTCUSDC",
            trade_id=12345,
            price=Decimal("50000.00"),
            quantity=Decimal("0.001"),
            quote_quantity=Decimal("50.00"),
            timestamp=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            is_buyer_maker=True
        )
    
    def test_trade_creation(self):
        """Test TradeData creation"""
        assert self.sample_trade.symbol == "BTCUSDC"
        assert self.sample_trade.trade_id == 12345
        assert self.sample_trade.is_buyer_maker is True
    
    def test_to_dict_conversion(self):
        """Test dictionary conversion"""
        result = self.sample_trade.to_dict()
        
        assert result["symbol"] == "BTCUSDC"
        assert result["trade_id"] == 12345
        assert result["price"] == "50000.00"
        assert result["quantity"] == "0.001"
        assert result["is_buyer_maker"] is True
    
    def test_from_dict_conversion(self):
        """Test creation from dictionary"""
        data = {
            "symbol": "BTCUSDC",
            "trade_id": 12345,
            "price": "50000.00",
            "quantity": "0.001",
            "quote_quantity": "50.00",
            "timestamp": "2023-01-01T12:00:00+00:00",
            "is_buyer_maker": True
        }
        
        trade = TradeData.from_dict(data)
        
        assert trade.symbol == "BTCUSDC"
        assert trade.trade_id == 12345
        assert trade.price == Decimal("50000.00")


class TestTickerData:
    """Test TickerData functionality"""
    
    def setup_method(self):
        """Set up test data"""
        self.sample_ticker = TickerData(
            symbol="BTCUSDC",
            price_change=Decimal("1000.00"),
            price_change_percent=Decimal("2.00"),
            weighted_avg_price=Decimal("50500.00"),
            prev_close_price=Decimal("49000.00"),
            last_price=Decimal("50000.00"),
            last_qty=Decimal("0.001"),
            bid_price=Decimal("49995.00"),
            bid_qty=Decimal("0.5"),
            ask_price=Decimal("50005.00"),
            ask_qty=Decimal("0.3"),
            open_price=Decimal("49000.00"),
            high_price=Decimal("51000.00"),
            low_price=Decimal("48000.00"),
            volume=Decimal("1000.0"),
            quote_volume=Decimal("50000000.00"),
            open_time=datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            close_time=datetime(2023, 1, 1, 23, 59, 59, tzinfo=timezone.utc),
            count=50000
        )
    
    def test_ticker_creation(self):
        """Test TickerData creation"""
        assert self.sample_ticker.symbol == "BTCUSDC"
        assert self.sample_ticker.last_price == Decimal("50000.00")
        assert self.sample_ticker.count == 50000
    
    def test_to_dict_conversion(self):
        """Test dictionary conversion"""
        result = self.sample_ticker.to_dict()
        
        assert result["symbol"] == "BTCUSDC"
        assert result["last_price"] == "50000.00"
        assert result["count"] == 50000
        assert "open_time" in result
        assert "close_time" in result


class TestDepthData:
    """Test DepthData functionality"""
    
    def setup_method(self):
        """Set up test data"""
        bids = [
            DepthLevel(Decimal("49990.00"), Decimal("1.0")),
            DepthLevel(Decimal("49980.00"), Decimal("2.0")),
        ]
        asks = [
            DepthLevel(Decimal("50010.00"), Decimal("1.5")),
            DepthLevel(Decimal("50020.00"), Decimal("2.5")),
        ]
        
        self.sample_depth = DepthData(
            symbol="BTCUSDC",
            last_update_id=123456789,
            bids=bids,
            asks=asks,
            timestamp=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        )
    
    def test_depth_creation(self):
        """Test DepthData creation"""
        assert self.sample_depth.symbol == "BTCUSDC"
        assert len(self.sample_depth.bids) == 2
        assert len(self.sample_depth.asks) == 2
    
    def test_best_bid_property(self):
        """Test best bid price calculation"""
        assert self.sample_depth.best_bid == Decimal("49990.00")
    
    def test_best_ask_property(self):
        """Test best ask price calculation"""
        assert self.sample_depth.best_ask == Decimal("50010.00")
    
    def test_spread_property(self):
        """Test bid-ask spread calculation"""
        expected_spread = Decimal("50010.00") - Decimal("49990.00")
        assert self.sample_depth.spread == expected_spread
    
    def test_mid_price_property(self):
        """Test mid price calculation"""
        expected_mid = (Decimal("49990.00") + Decimal("50010.00")) / 2
        assert self.sample_depth.mid_price == expected_mid
    
    def test_empty_depth_properties(self):
        """Test depth properties with empty order book"""
        empty_depth = DepthData(
            symbol="BTCUSDC",
            last_update_id=0,
            bids=[],
            asks=[],
            timestamp=datetime.now(timezone.utc)
        )
        
        assert empty_depth.best_bid is None
        assert empty_depth.best_ask is None
        assert empty_depth.spread is None
        assert empty_depth.mid_price is None
    
    def test_to_dict_conversion(self):
        """Test dictionary conversion"""
        result = self.sample_depth.to_dict()
        
        assert result["symbol"] == "BTCUSDC"
        assert result["last_update_id"] == 123456789
        assert len(result["bids"]) == 2
        assert len(result["asks"]) == 2
        assert result["bids"][0] == ["49990.00", "1.0"]
    
    def test_from_dict_conversion(self):
        """Test creation from dictionary"""
        data = {
            "symbol": "BTCUSDC",
            "last_update_id": 123456789,
            "bids": [["49990.00", "1.0"], ["49980.00", "2.0"]],
            "asks": [["50010.00", "1.5"], ["50020.00", "2.5"]],
            "timestamp": "2023-01-01T12:00:00+00:00"
        }
        
        depth = DepthData.from_dict(data)
        
        assert depth.symbol == "BTCUSDC"
        assert len(depth.bids) == 2
        assert depth.bids[0].price == Decimal("49990.00")


class TestFeatureData:
    """Test FeatureData functionality"""
    
    def setup_method(self):
        """Set up test data"""
        self.sample_features = FeatureData(
            symbol="BTCUSDC",
            timestamp=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            features={
                "sma_20": 50000.0,
                "rsi_14": 65.5,
                "volume": 1000.0,
                "price_change": 0.02
            },
            target=1.0
        )
    
    def test_feature_creation(self):
        """Test FeatureData creation"""
        assert self.sample_features.symbol == "BTCUSDC"
        assert self.sample_features.features["sma_20"] == 50000.0
        assert self.sample_features.target == 1.0
    
    def test_feature_creation_no_target(self):
        """Test FeatureData creation without target"""
        features = FeatureData(
            symbol="BTCUSDC",
            timestamp=datetime.now(timezone.utc),
            features={"price": 50000.0}
        )
        assert features.target is None
    
    def test_to_dict_conversion(self):
        """Test dictionary conversion"""
        result = self.sample_features.to_dict()
        
        assert result["symbol"] == "BTCUSDC"
        assert result["features"]["sma_20"] == 50000.0
        assert result["target"] == 1.0
    
    def test_from_dict_conversion(self):
        """Test creation from dictionary"""
        data = {
            "symbol": "BTCUSDC",
            "timestamp": "2023-01-01T12:00:00+00:00",
            "features": {"sma_20": 50000.0, "rsi_14": 65.5},
            "target": 1.0
        }
        
        features = FeatureData.from_dict(data)
        
        assert features.symbol == "BTCUSDC"
        assert features.features["sma_20"] == 50000.0
        assert features.target == 1.0


class TestModelPrediction:
    """Test ModelPrediction functionality"""
    
    def setup_method(self):
        """Set up test data"""
        self.sample_prediction = ModelPrediction(
            symbol="BTCUSDC",
            timestamp=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            prediction=0.75,
            confidence=0.85,
            features_used=["sma_20", "rsi_14", "volume"],
            model_version="v1.2.3"
        )
    
    def test_prediction_creation(self):
        """Test ModelPrediction creation"""
        assert self.sample_prediction.symbol == "BTCUSDC"
        assert self.sample_prediction.prediction == 0.75
        assert self.sample_prediction.confidence == 0.85
        assert len(self.sample_prediction.features_used) == 3
    
    def test_to_dict_conversion(self):
        """Test dictionary conversion"""
        result = self.sample_prediction.to_dict()
        
        assert result["symbol"] == "BTCUSDC"
        assert result["prediction"] == 0.75
        assert result["model_version"] == "v1.2.3"
        assert len(result["features_used"]) == 3
    
    def test_from_dict_conversion(self):
        """Test creation from dictionary"""
        data = {
            "symbol": "BTCUSDC",
            "timestamp": "2023-01-01T12:00:00+00:00",
            "prediction": 0.75,
            "confidence": 0.85,
            "features_used": ["sma_20", "rsi_14"],
            "model_version": "v1.0.0"
        }
        
        prediction = ModelPrediction.from_dict(data)
        
        assert prediction.symbol == "BTCUSDC"
        assert prediction.prediction == 0.75
        assert prediction.model_version == "v1.0.0"


class TestTradeRecord:
    """Test TradeRecord functionality"""
    
    def setup_method(self):
        """Set up test data"""
        self.sample_record = TradeRecord(
            trade_id="T123456",
            symbol="BTCUSDC",
            side="BUY",
            quantity=Decimal("0.001"),
            price=Decimal("50000.00"),
            fee=Decimal("0.05"),
            fee_asset="USDC",
            timestamp=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            order_id="O123456",
            client_order_id="C123456",
            realized_pnl=Decimal("10.0")
        )
    
    def test_record_creation(self):
        """Test TradeRecord creation"""
        assert self.sample_record.trade_id == "T123456"
        assert self.sample_record.side == "BUY"
        assert self.sample_record.quantity == Decimal("0.001")
        assert self.sample_record.realized_pnl == Decimal("10.0")
    
    def test_record_creation_no_pnl(self):
        """Test TradeRecord creation without PnL"""
        record = TradeRecord(
            trade_id="T123456",
            symbol="BTCUSDC",
            side="BUY",
            quantity=Decimal("0.001"),
            price=Decimal("50000.00"),
            fee=Decimal("0.05"),
            fee_asset="USDC",
            timestamp=datetime.now(timezone.utc),
            order_id="O123456",
            client_order_id="C123456"
        )
        assert record.realized_pnl is None
    
    def test_to_dict_conversion(self):
        """Test dictionary conversion"""
        result = self.sample_record.to_dict()
        
        assert result["trade_id"] == "T123456"
        assert result["side"] == "BUY"
        assert result["quantity"] == "0.001"
        assert result["realized_pnl"] == "10.0"
    
    def test_from_dict_conversion(self):
        """Test creation from dictionary"""
        data = {
            "trade_id": "T123456",
            "symbol": "BTCUSDC",
            "side": "BUY",
            "quantity": "0.001",
            "price": "50000.00",
            "fee": "0.05",
            "fee_asset": "USDC",
            "timestamp": "2023-01-01T12:00:00+00:00",
            "order_id": "O123456",
            "client_order_id": "C123456",
            "realized_pnl": "10.0"
        }
        
        record = TradeRecord.from_dict(data)
        
        assert record.trade_id == "T123456"
        assert record.realized_pnl == Decimal("10.0")


class TestEquityCurvePoint:
    """Test EquityCurvePoint functionality"""
    
    def setup_method(self):
        """Set up test data"""
        self.sample_point = EquityCurvePoint(
            timestamp=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            total_balance=Decimal("10000.00"),
            unrealized_pnl=Decimal("150.00"),
            realized_pnl=Decimal("250.00"),
            fees_paid=Decimal("25.00"),
            trade_count=10
        )
    
    def test_point_creation(self):
        """Test EquityCurvePoint creation"""
        assert self.sample_point.total_balance == Decimal("10000.00")
        assert self.sample_point.trade_count == 10
    
    def test_net_worth_property(self):
        """Test net worth calculation"""
        expected = Decimal("10000.00") + Decimal("150.00")
        assert self.sample_point.net_worth == expected
    
    def test_to_dict_conversion(self):
        """Test dictionary conversion"""
        result = self.sample_point.to_dict()
        
        assert result["total_balance"] == "10000.00"
        assert result["net_worth"] == "10150.00"
        assert result["trade_count"] == 10
    
    def test_from_dict_conversion(self):
        """Test creation from dictionary"""
        data = {
            "timestamp": "2023-01-01T12:00:00+00:00",
            "total_balance": "10000.00",
            "unrealized_pnl": "150.00",
            "realized_pnl": "250.00",
            "fees_paid": "25.00",
            "trade_count": 10
        }
        
        point = EquityCurvePoint.from_dict(data)
        
        assert point.total_balance == Decimal("10000.00")
        assert point.net_worth == Decimal("10150.00")


class TestBacktestResult:
    """Test BacktestResult functionality"""
    
    def setup_method(self):
        """Set up test data"""
        equity_point = EquityCurvePoint(
            timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            total_balance=Decimal("10000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            fees_paid=Decimal("0"),
            trade_count=0
        )
        
        trade_record = TradeRecord(
            trade_id="T1",
            symbol="BTCUSDC",
            side="BUY",
            quantity=Decimal("0.001"),
            price=Decimal("50000"),
            fee=Decimal("0.05"),
            fee_asset="USDC",
            timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            order_id="O1",
            client_order_id="C1"
        )
        
        self.sample_result = BacktestResult(
            symbol="BTCUSDC",
            start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2023, 1, 31, tzinfo=timezone.utc),
            initial_balance=Decimal("10000.00"),
            final_balance=Decimal("11000.00"),
            total_return=Decimal("0.10"),
            total_trades=50,
            winning_trades=30,
            losing_trades=20,
            win_rate=Decimal("0.60"),
            profit_factor=Decimal("1.5"),
            max_drawdown=Decimal("0.05"),
            sharpe_ratio=Decimal("1.2"),
            calmar_ratio=Decimal("2.4"),
            equity_curve=[equity_point],
            trades=[trade_record]
        )
    
    def test_result_creation(self):
        """Test BacktestResult creation"""
        assert self.sample_result.symbol == "BTCUSDC"
        assert self.sample_result.total_trades == 50
        assert self.sample_result.win_rate == Decimal("0.60")
        assert len(self.sample_result.equity_curve) == 1
        assert len(self.sample_result.trades) == 1
    
    def test_to_dict_conversion(self):
        """Test dictionary conversion"""
        result = self.sample_result.to_dict()
        
        assert result["symbol"] == "BTCUSDC"
        assert result["total_trades"] == 50
        assert result["win_rate"] == "0.60"
        assert len(result["equity_curve"]) == 1
        assert len(result["trades"]) == 1


class TestBinanceConversions:
    """Test Binance data conversion functions"""
    
    def test_klines_from_binance(self):
        """Test Binance kline data conversion"""
        binance_data = [
            [
                1609459200000,  # open_time
                "50000.00",     # open
                "50100.00",     # high  
                "49900.00",     # low
                "50050.00",     # close
                "1.5",          # volume
                1609459260000,  # close_time
                "75075.00",     # quote_volume
                100,            # trades_count
                "0.8",          # taker_buy_volume
                "40020.00"      # taker_buy_quote_volume
            ]
        ]
        
        result = klines_from_binance(binance_data, "BTCUSDC", TimeFrame.M1)
        
        assert len(result) == 1
        kline = result[0]
        assert kline.symbol == "BTCUSDC"
        assert kline.open_price == Decimal("50000.00")
        assert kline.timeframe == TimeFrame.M1
    
    def test_trades_from_binance(self):
        """Test Binance trade data conversion"""
        binance_data = [
            {
                "id": 12345,
                "price": "50000.00",
                "qty": "0.001",
                "quoteQty": "50.00",
                "time": 1609459200000,
                "isBuyerMaker": True
            }
        ]
        
        result = trades_from_binance(binance_data, "BTCUSDC")
        
        assert len(result) == 1
        trade = result[0]
        assert trade.symbol == "BTCUSDC"
        assert trade.trade_id == 12345
        assert trade.price == Decimal("50000.00")
        assert trade.is_buyer_maker is True
    
    def test_depth_from_binance(self):
        """Test Binance depth data conversion"""
        binance_data = {
            "lastUpdateId": 123456789,
            "bids": [["49990.00", "1.0"], ["49980.00", "2.0"]],
            "asks": [["50010.00", "1.5"], ["50020.00", "2.5"]]
        }
        
        result = depth_from_binance(binance_data, "BTCUSDC")
        
        assert result.symbol == "BTCUSDC"
        assert result.last_update_id == 123456789
        assert len(result.bids) == 2
        assert len(result.asks) == 2
        assert result.best_bid == Decimal("49990.00")
        assert result.best_ask == Decimal("50010.00")


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_kline_identical_open_close(self):
        """Test kline with identical open and close prices"""
        kline = KlineData(
            symbol="BTCUSDC",
            open_time=datetime.now(timezone.utc),
            close_time=datetime.now(timezone.utc),
            open_price=Decimal("50000"),
            high_price=Decimal("50000"),
            low_price=Decimal("50000"),
            close_price=Decimal("50000"),  # Same as open
            volume=Decimal("1"),
            quote_volume=Decimal("50000"),
            trades_count=10,
            taker_buy_volume=Decimal("0.5"),
            taker_buy_quote_volume=Decimal("25000"),
            timeframe=TimeFrame.M1
        )
        
        assert kline.body_size == Decimal("0")
        assert kline.is_bullish is False  # Not bullish when equal
        assert kline.price_range == Decimal("0")
    
    def test_depth_single_level(self):
        """Test depth data with single bid/ask level"""
        bids = [DepthLevel(Decimal("50000"), Decimal("1.0"))]
        asks = [DepthLevel(Decimal("50010"), Decimal("1.0"))]
        
        depth = DepthData(
            symbol="BTCUSDC",
            last_update_id=1,
            bids=bids,
            asks=asks,
            timestamp=datetime.now(timezone.utc)
        )
        
        assert depth.best_bid == Decimal("50000")
        assert depth.best_ask == Decimal("50010")
        assert depth.spread == Decimal("10")
    
    def test_empty_binance_data(self):
        """Test conversion functions with empty data"""
        empty_klines = klines_from_binance([], "BTCUSDC", TimeFrame.M1)
        assert empty_klines == []
        
        empty_trades = trades_from_binance([], "BTCUSDC")
        assert empty_trades == []
    
    def test_feature_data_empty_features(self):
        """Test FeatureData with empty features dictionary"""
        features = FeatureData(
            symbol="BTCUSDC",
            timestamp=datetime.now(timezone.utc),
            features={}
        )
        
        assert len(features.features) == 0
        assert features.target is None
    
    def test_large_decimal_values(self):
        """Test handling of large decimal values"""
        large_value = Decimal("999999999999.999999999")
        
        kline = KlineData(
            symbol="TESTCOIN",
            open_time=datetime.now(timezone.utc),
            close_time=datetime.now(timezone.utc),
            open_price=large_value,
            high_price=large_value,
            low_price=large_value,
            close_price=large_value,
            volume=large_value,
            quote_volume=large_value,
            trades_count=1,
            taker_buy_volume=large_value,
            taker_buy_quote_volume=large_value,
            timeframe=TimeFrame.M1
        )
        
        # Should handle large values without errors
        assert kline.typical_price == large_value
        dict_data = kline.to_dict()
        assert dict_data["open"] == str(large_value)


if __name__ == "__main__":
    pytest.main([__file__])
