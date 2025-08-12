"""
Comprehensive Test Suite for /exec Directory

This module provides comprehensive unit tests for all functions, methods, and classes
in the exec directory, covering binance.py, paper_broker.py, and rate_limit.py.
"""

import pytest
import asyncio
import time
import json
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Optional

# Import modules under test - need to add parent directory to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exec.binance import (
    BinanceConfig, BinanceError, BinanceRateLimitError, BinanceClient,
    mask_api_key, create_binance_client
)
from exec.paper_broker import (
    MarketData, OrderBookLevel, OrderBook, SlippageModel, LinearSlippageModel,
    PaperBroker
)
from exec.rate_limit import (
    RateLimitError, RateLimitConfig, TokenBucket, RateLimiter,
    AdaptiveRateLimiter, create_conservative_rate_limiter, create_aggressive_rate_limiter
)
from core.state import Order, OrderSide, OrderType, OrderStatus, Position, Balance
from core.sizing import ExchangeInfo, FeeConfig, ensure_min_notional


class TestExecBinanceMaskApiKey:
    """Test mask_api_key utility function"""
    
    def test_mask_api_key_normal(self):
        """Test API key masking with normal key"""
        api_key = "abc123def456ghi789"
        result = mask_api_key(api_key)
        assert result == "abc1***i789"
        assert len(result) == 11
    
    def test_mask_api_key_short(self):
        """Test API key masking with short key"""
        api_key = "short"
        result = mask_api_key(api_key)
        assert result == "***"
    
    def test_mask_api_key_empty(self):
        """Test API key masking with empty key"""
        result = mask_api_key("")
        assert result == "***"
    
    def test_mask_api_key_none(self):
        """Test API key masking with None"""
        result = mask_api_key(None)
        assert result == "***"
    
    def test_mask_api_key_minimum_length(self):
        """Test API key masking with minimum valid length"""
        api_key = "12345678"
        result = mask_api_key(api_key)
        assert result == "1234***5678"


class TestExecBinanceConfig:
    """Test BinanceConfig data class"""
    
    def test_binance_config_creation(self):
        """Test BinanceConfig creation with required fields"""
        config = BinanceConfig(
            api_key="test_key",
            api_secret="test_secret"
        )
        
        assert config.api_key == "test_key"
        assert config.api_secret == "test_secret"
        assert config.base_url == "https://api.binance.com"
        assert config.testnet is False
        assert config.timeout == 30
        assert config.max_retries == 3
    
    def test_binance_config_testnet(self):
        """Test BinanceConfig with testnet enabled"""
        config = BinanceConfig(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
        
        assert config.testnet is True
        assert config.base_url == "https://testnet.binance.vision"
    
    def test_binance_config_custom_values(self):
        """Test BinanceConfig with custom values"""
        config = BinanceConfig(
            api_key="test_key",
            api_secret="test_secret",
            base_url="https://custom.api.com",
            timeout=60,
            max_retries=5,
            retry_delay_base=2.0,
            retry_delay_max=120.0,
            retry_jitter=0.2
        )
        
        assert config.base_url == "https://custom.api.com"
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.retry_delay_base == 2.0
        assert config.retry_delay_max == 120.0
        assert config.retry_jitter == 0.2
    
    @patch('exec.binance._unified_config_available', True)
    @patch('exec.binance.get_settings')
    def test_binance_config_from_unified_config(self, mock_get_settings):
        """Test BinanceConfig creation from unified config"""
        mock_config = Mock()
        mock_config.binance.api_key = "unified_key"
        mock_config.binance.api_secret = "unified_secret"
        mock_config.binance.base_url = None
        mock_config.binance.testnet = False
        mock_get_settings.return_value = mock_config
        
        config = BinanceConfig.from_unified_config()
        
        assert config.api_key == "unified_key"
        assert config.api_secret == "unified_secret"
        assert config.testnet is False
    
    @patch('exec.binance._unified_config_available', False)
    def test_binance_config_from_unified_config_unavailable(self):
        """Test BinanceConfig creation when unified config unavailable"""
        with pytest.raises(ImportError):
            BinanceConfig.from_unified_config()


class TestExecBinanceError:
    """Test BinanceError exception classes"""
    
    def test_binance_error_basic(self):
        """Test basic BinanceError creation"""
        error = BinanceError("Test error")
        
        assert str(error) == "Test error"
        assert error.code is None
        assert error.response == {}
    
    def test_binance_error_with_code(self):
        """Test BinanceError with error code"""
        response = {"code": -1021, "msg": "Timestamp out of window"}
        error = BinanceError("Timestamp error", code=-1021, response=response)
        
        assert error.code == -1021
        assert error.response == response
    
    def test_binance_rate_limit_error(self):
        """Test BinanceRateLimitError inheritance"""
        error = BinanceRateLimitError("Rate limit exceeded")
        
        assert isinstance(error, BinanceError)
        assert str(error) == "Rate limit exceeded"


class TestExecBinanceClient:
    """Test BinanceClient functionality"""
    
    @pytest.fixture
    def binance_config(self):
        """Fixture for BinanceConfig"""
        return BinanceConfig(
            api_key="test_api_key",
            api_secret="test_api_secret",
            testnet=True
        )
    
    @pytest.fixture
    def binance_client(self, binance_config):
        """Fixture for BinanceClient"""
        return BinanceClient(binance_config)
    
    def test_binance_client_initialization(self, binance_client, binance_config):
        """Test BinanceClient initialization"""
        assert binance_client.config == binance_config
        assert binance_client.session is None
        assert binance_client.rate_limiter is not None
        assert binance_client.server_time_offset == 0
    
    @pytest.mark.asyncio
    async def test_binance_client_context_manager(self, binance_client):
        """Test BinanceClient as async context manager"""
        with patch.object(binance_client, 'start_session') as mock_start, \
             patch.object(binance_client, 'close_session') as mock_close:
            
            async with binance_client as client:
                assert client == binance_client
                mock_start.assert_called_once()
            
            mock_close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_session(self, binance_client):
        """Test session initialization"""
        with patch('aiohttp.ClientSession') as mock_session_class, \
             patch.object(binance_client, '_sync_server_time') as mock_sync:
            
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session
            
            await binance_client.start_session()
            
            assert binance_client.session == mock_session
            mock_sync.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_close_session(self, binance_client):
        """Test session cleanup"""
        mock_session = AsyncMock()
        binance_client.session = mock_session
        
        await binance_client.close_session()
        
        mock_session.close.assert_called_once()
        assert binance_client.session is None
    
    @pytest.mark.asyncio
    async def test_sync_server_time(self, binance_client):
        """Test server time synchronization"""
        with patch.object(binance_client, '_make_request') as mock_request:
            mock_request.return_value = {"serverTime": 1640995200000}
            
            with patch('time.time', return_value=1640995100.0):
                await binance_client._sync_server_time()
            
            # Server time offset should be calculated
            assert binance_client.server_time_offset == 100000  # 100 seconds in ms
            assert binance_client.last_time_sync > 0
    
    def test_get_timestamp(self, binance_client):
        """Test timestamp generation with server offset"""
        binance_client.server_time_offset = 1000
        binance_client.last_time_sync = time.time()
        
        with patch('time.time', return_value=1640995200.0):
            timestamp = binance_client._get_timestamp()
            
            expected = int(1640995200.0 * 1000) + 1000
            assert timestamp == expected
    
    def test_generate_signature(self, binance_client):
        """Test HMAC signature generation"""
        query_string = "symbol=BTCUSDT&side=BUY&type=LIMIT&quantity=1&price=50000&timestamp=1640995200000"
        signature = binance_client._generate_signature(query_string)
        
        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA256 hex digest length
    
    def test_calculate_retry_delay(self, binance_client):
        """Test retry delay calculation with exponential backoff"""
        # Test first attempt
        delay_0 = binance_client._calculate_retry_delay(0)
        assert delay_0 >= 1.0  # Base delay
        assert delay_0 <= 1.1  # Base + max jitter
        
        # Test exponential growth
        delay_2 = binance_client._calculate_retry_delay(2)
        assert delay_2 >= 4.0  # 1.0 * 2^2
        assert delay_2 <= 4.4  # Base + max jitter
        
        # Test maximum cap
        delay_10 = binance_client._calculate_retry_delay(10)
        assert delay_10 <= 66.0  # 60.0 max + 10% jitter
    
    @pytest.mark.asyncio
    async def test_make_request_single_success(self, binance_client):
        """Test successful single HTTP request"""
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"symbol": "BTCUSDT", "price": "50000"}
        mock_session.request.return_value.__aenter__.return_value = mock_response
        
        binance_client.session = mock_session
        binance_client.rate_limiter = AsyncMock()
        
        result = await binance_client._make_request_single("GET", "/api/v3/ticker/price")
        
        assert result == {"symbol": "BTCUSDT", "price": "50000"}
        binance_client.rate_limiter.acquire.assert_called_once_with(weight=1)
    
    @pytest.mark.asyncio
    async def test_make_request_single_rate_limit_error(self, binance_client):
        """Test single HTTP request with rate limit error"""
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.json.return_value = {"code": 429, "msg": "Rate limit exceeded"}
        mock_session.request.return_value.__aenter__.return_value = mock_response
        
        binance_client.session = mock_session
        binance_client.rate_limiter = AsyncMock()
        
        with pytest.raises(BinanceRateLimitError):
            await binance_client._make_request_single("GET", "/api/v3/ticker/price")
    
    @pytest.mark.asyncio
    async def test_make_request_single_api_error(self, binance_client):
        """Test single HTTP request with API error"""
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.json.return_value = {"code": -1021, "msg": "Invalid timestamp"}
        mock_session.request.return_value.__aenter__.return_value = mock_response
        
        binance_client.session = mock_session
        binance_client.rate_limiter = AsyncMock()
        
        with pytest.raises(BinanceError) as exc_info:
            await binance_client._make_request_single("GET", "/api/v3/ticker/price")
        
        assert exc_info.value.code == -1021
    
    @pytest.mark.asyncio
    async def test_make_request_with_retry_success_after_retry(self, binance_client):
        """Test request success after retry"""
        with patch.object(binance_client, '_make_request_single') as mock_single:
            # First call fails, second succeeds
            mock_single.side_effect = [
                BinanceRateLimitError("Rate limit"),
                {"success": True}
            ]
            
            with patch('asyncio.sleep'):
                result = await binance_client._make_request_with_retry("GET", "/api/v3/test")
            
            assert result == {"success": True}
            assert mock_single.call_count == 2
    
    @pytest.mark.asyncio
    async def test_get_server_time(self, binance_client):
        """Test get_server_time method"""
        with patch.object(binance_client, '_make_request') as mock_request:
            mock_request.return_value = {"serverTime": 1640995200000}
            
            result = await binance_client.get_server_time()
            
            assert result == {"serverTime": 1640995200000}
            mock_request.assert_called_once_with("GET", "/api/v3/time")
    
    @pytest.mark.asyncio
    async def test_get_exchange_info(self, binance_client):
        """Test get_exchange_info method"""
        with patch.object(binance_client, '_make_request') as mock_request:
            mock_request.return_value = {"symbols": [{"symbol": "BTCUSDT"}]}
            
            result = await binance_client.get_exchange_info("BTCUSDT")
            
            mock_request.assert_called_once_with("GET", "/api/v3/exchangeInfo", {"symbol": "BTCUSDT"})
    
    @pytest.mark.asyncio
    async def test_place_order(self, binance_client):
        """Test place_order method"""
        binance_client.rate_limiter = AsyncMock()
        
        with patch.object(binance_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "orderId": 123456,
                "symbol": "BTCUSDT",
                "status": "NEW"
            }
            
            result = await binance_client.place_order(
                symbol="BTCUSDT",
                side="BUY",
                order_type="LIMIT",
                quantity=Decimal("1.0"),
                price=Decimal("50000")
            )
            
            assert result["orderId"] == 123456
            binance_client.rate_limiter.acquire_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cancel_order_with_order_id(self, binance_client):
        """Test cancel_order with order ID"""
        with patch.object(binance_client, '_make_request') as mock_request:
            mock_request.return_value = {"orderId": 123456, "status": "CANCELED"}
            
            result = await binance_client.cancel_order("BTCUSDT", order_id=123456)
            
            expected_params = {"symbol": "BTCUSDT", "orderId": 123456}
            mock_request.assert_called_once_with("DELETE", "/api/v3/order", expected_params, signed=True, weight=1)
    
    @pytest.mark.asyncio
    async def test_cancel_order_missing_id(self, binance_client):
        """Test cancel_order without order ID or client order ID"""
        with pytest.raises(ValueError, match="Either order_id or client_order_id must be provided"):
            await binance_client.cancel_order("BTCUSDT")
    
    def test_convert_order_to_internal(self, binance_client):
        """Test conversion of Binance order to internal Order object"""
        binance_order = {
            "orderId": 123456,
            "clientOrderId": "test_order",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "origQty": "1.0",
            "price": "50000.0",
            "status": "FILLED",
            "executedQty": "1.0",
            "time": 1640995200000,
            "updateTime": 1640995300000
        }
        
        order = binance_client.convert_order_to_internal(binance_order)
        
        assert order.order_id == "123456"
        assert order.symbol == "BTCUSDT"
        assert order.side == OrderSide.BUY
        assert order.quantity == Decimal("1.0")
        assert order.status == OrderStatus.FILLED
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, binance_client):
        """Test successful health check"""
        with patch.object(binance_client, 'get_server_time') as mock_get_time:
            mock_get_time.return_value = {"serverTime": 1640995200000}
            
            result = await binance_client.health_check()
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, binance_client):
        """Test failed health check"""
        with patch.object(binance_client, 'get_server_time') as mock_get_time:
            mock_get_time.side_effect = Exception("Connection failed")
            
            result = await binance_client.health_check()
            
            assert result is False


class TestExecPaperBrokerMarketData:
    """Test MarketData class"""
    
    def test_market_data_creation(self):
        """Test MarketData creation"""
        data = MarketData(
            symbol="BTCUSDT",
            bid=Decimal("49900"),
            ask=Decimal("50100"),
            last_price=Decimal("50000"),
            volume=Decimal("1000")
        )
        
        assert data.symbol == "BTCUSDT"
        assert data.bid == Decimal("49900")
        assert data.ask == Decimal("50100")
        assert data.last_price == Decimal("50000")
        assert data.volume == Decimal("1000")
    
    def test_market_data_mid_price(self):
        """Test mid price calculation"""
        data = MarketData(
            symbol="BTCUSDT",
            bid=Decimal("49900"),
            ask=Decimal("50100"),
            last_price=Decimal("50000"),
            volume=Decimal("1000")
        )
        
        assert data.mid_price == Decimal("50000")
    
    def test_market_data_spread(self):
        """Test spread calculations"""
        data = MarketData(
            symbol="BTCUSDT",
            bid=Decimal("49900"),
            ask=Decimal("50100"),
            last_price=Decimal("50000"),
            volume=Decimal("1000")
        )
        
        assert data.spread == Decimal("200")
        assert data.spread_pct == Decimal("0.4")  # 200/50000 * 100


class TestExecPaperBrokerOrderBook:
    """Test OrderBook and OrderBookLevel classes"""
    
    def test_order_book_level_creation(self):
        """Test OrderBookLevel creation"""
        level = OrderBookLevel(Decimal("50000"), Decimal("1.5"))
        
        assert level.price == Decimal("50000")
        assert level.quantity == Decimal("1.5")
    
    def test_order_book_creation(self):
        """Test OrderBook creation"""
        book = OrderBook(symbol="BTCUSDT")
        
        assert book.symbol == "BTCUSDT"
        assert book.bids == []
        assert book.asks == []
    
    def test_order_book_best_prices(self):
        """Test best bid/ask calculation"""
        bids = [OrderBookLevel(Decimal("49900"), Decimal("1.0"))]
        asks = [OrderBookLevel(Decimal("50100"), Decimal("1.0"))]
        
        book = OrderBook(symbol="BTCUSDT", bids=bids, asks=asks)
        
        assert book.get_best_bid() == Decimal("49900")
        assert book.get_best_ask() == Decimal("50100")
        assert book.get_mid_price() == Decimal("50000")
    
    def test_order_book_empty(self):
        """Test empty order book"""
        book = OrderBook(symbol="BTCUSDT")
        
        assert book.get_best_bid() is None
        assert book.get_best_ask() is None
        assert book.get_mid_price() is None


class TestExecPaperBrokerSlippage:
    """Test slippage models"""
    
    def test_linear_slippage_model_creation(self):
        """Test LinearSlippageModel creation"""
        model = LinearSlippageModel(
            base_slippage_bps=Decimal('5'),
            size_impact_factor=Decimal('0.2')
        )
        
        assert model.base_slippage_bps == Decimal('5')
        assert model.size_impact_factor == Decimal('0.2')
    
    def test_linear_slippage_calculation_buy(self):
        """Test slippage calculation for buy order"""
        model = LinearSlippageModel()
        
        order = Order(
            order_id="test",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=Decimal("50000")
        )
        
        market_data = MarketData(
            symbol="BTCUSDT",
            bid=Decimal("49900"),
            ask=Decimal("50100"),
            last_price=Decimal("50000"),
            volume=Decimal("1000")
        )
        
        slippage = model.calculate_slippage(order, market_data)
        
        assert slippage > 0  # Positive slippage for buy orders
        assert isinstance(slippage, Decimal)
    
    def test_linear_slippage_calculation_sell(self):
        """Test slippage calculation for sell order"""
        model = LinearSlippageModel()
        
        order = Order(
            order_id="test",
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=Decimal("50000")
        )
        
        market_data = MarketData(
            symbol="BTCUSDT",
            bid=Decimal("49900"),
            ask=Decimal("50100"),
            last_price=Decimal("50000"),
            volume=Decimal("1000")
        )
        
        slippage = model.calculate_slippage(order, market_data)
        
        assert slippage < 0  # Negative slippage for sell orders


class TestExecPaperBroker:
    """Test PaperBroker functionality"""
    
    @pytest.fixture
    def exchange_info(self):
        """Fixture for exchange info"""
        return {
            "BTCUSDT": ExchangeInfo(
                symbol="BTCUSDT",
                min_qty=Decimal("0.001"),
                max_qty=Decimal("1000"),
                step_size=Decimal("0.001"),
                tick_size=Decimal("0.01"),
                min_notional=Decimal("10"),
                base_asset_precision=3,
                quote_asset_precision=2
            )
        }
    
    @pytest.fixture
    def fee_config(self):
        """Fixture for fee config"""
        return FeeConfig(
            maker_fee=Decimal('0.001'),
            taker_fee=Decimal('0.001')
        )
    
    @pytest.fixture
    def paper_broker(self, exchange_info, fee_config):
        """Fixture for PaperBroker"""
        initial_balances = {
            "BTC": Decimal("10"),
            "USDT": Decimal("100000")
        }
        
        return PaperBroker(
            initial_balances=initial_balances,
            exchange_info=exchange_info,
            fee_config=fee_config
        )
    
    def test_paper_broker_initialization(self, paper_broker):
        """Test PaperBroker initialization"""
        assert "BTC" in paper_broker.balances
        assert "USDT" in paper_broker.balances
        assert paper_broker.balances["BTC"].free == Decimal("10")
        assert paper_broker.balances["USDT"].free == Decimal("100000")
        assert len(paper_broker.orders) == 0
        assert len(paper_broker.positions) == 0
    
    def test_paper_broker_callbacks(self, paper_broker):
        """Test callback registration"""
        callback = Mock()
        
        paper_broker.register_fill_callback(callback)
        paper_broker.register_order_callback(callback)
        
        assert callback in paper_broker.fill_callbacks
        assert callback in paper_broker.order_callbacks
    
    def test_update_market_data(self, paper_broker):
        """Test market data update"""
        paper_broker.update_market_data(
            symbol="BTCUSDT",
            bid=Decimal("49900"),
            ask=Decimal("50100"),
            last_price=Decimal("50000"),
            volume=Decimal("1000")
        )
        
        assert "BTCUSDT" in paper_broker.market_data
        market = paper_broker.market_data["BTCUSDT"]
        assert market.symbol == "BTCUSDT"
        assert market.bid == Decimal("49900")
        assert market.ask == Decimal("50100")
    
    @pytest.mark.asyncio
    async def test_place_limit_order_success(self, paper_broker):
        """Test successful limit order placement"""
        # Set up market data
        paper_broker.update_market_data(
            symbol="BTCUSDT",
            bid=Decimal("49900"),
            ask=Decimal("50100"),
            last_price=Decimal("50000")
        )
        
        order = await paper_broker.place_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000")
        )
        
        assert order.symbol == "BTCUSDT"
        assert order.side == OrderSide.BUY
        assert order.quantity == Decimal("1.0")
        assert order.price == Decimal("50000")
        assert order.status == OrderStatus.NEW
        
        # Check balance reservation
        assert paper_broker.balances["USDT"].locked == Decimal("50000")
    
    @pytest.mark.asyncio
    async def test_place_market_order_immediate_fill(self, paper_broker):
        """Test market order immediate fill"""
        # Set up market data
        paper_broker.update_market_data(
            symbol="BTCUSDT",
            bid=Decimal("49900"),
            ask=Decimal("50100"),
            last_price=Decimal("50000")
        )
        
        order = await paper_broker.place_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0")
        )
        
        # Market orders should be filled immediately
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == Decimal("1.0")
        assert order.order_id not in paper_broker.orders
        assert len(paper_broker.order_history) == 1
    
    @pytest.mark.asyncio
    async def test_place_order_insufficient_balance(self, paper_broker):
        """Test order placement with insufficient balance"""
        paper_broker.update_market_data(
            symbol="BTCUSDT",
            bid=Decimal("49900"),
            ask=Decimal("50100"),
            last_price=Decimal("50000")
        )
        
        with pytest.raises(ValueError, match="Insufficient balance"):
            await paper_broker.place_order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("10.0"),  # Too large
                price=Decimal("50000")
            )
    
    @pytest.mark.asyncio
    async def test_cancel_order_success(self, paper_broker):
        """Test successful order cancellation"""
        paper_broker.update_market_data(
            symbol="BTCUSDT",
            bid=Decimal("49900"),
            ask=Decimal("50100"),
            last_price=Decimal("50000")
        )
        
        order = await paper_broker.place_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("49000")  # Below market, won't fill
        )
        
        success = await paper_broker.cancel_order(order.order_id)
        
        assert success is True
        assert order.status == OrderStatus.CANCELED
        assert order.order_id not in paper_broker.orders
        assert paper_broker.balances["USDT"].locked == Decimal("0")
    
    @pytest.mark.asyncio
    async def test_cancel_nonexistent_order(self, paper_broker):
        """Test cancellation of non-existent order"""
        success = await paper_broker.cancel_order("nonexistent")
        assert success is False
    
    def test_check_balance_sufficient(self, paper_broker):
        """Test balance check with sufficient funds"""
        result = paper_broker._check_balance("BTCUSDT", OrderSide.BUY, Decimal("1.0"), Decimal("50000"))
        assert result is True
        
        result = paper_broker._check_balance("BTCUSDT", OrderSide.SELL, Decimal("5.0"), Decimal("50000"))
        assert result is True
    
    def test_check_balance_insufficient(self, paper_broker):
        """Test balance check with insufficient funds"""
        result = paper_broker._check_balance("BTCUSDT", OrderSide.BUY, Decimal("10.0"), Decimal("50000"))
        assert result is False
        
        result = paper_broker._check_balance("BTCUSDT", OrderSide.SELL, Decimal("20.0"), Decimal("50000"))
        assert result is False
    
    def test_parse_symbol(self, paper_broker):
        """Test symbol parsing"""
        base, quote = paper_broker._parse_symbol("BTCUSDC")
        assert base == "BTC"
        assert quote == "USDC"
        
        base, quote = paper_broker._parse_symbol("ETHBTC")
        assert base == "ETH"
        assert quote == "BTC"
    
    def test_get_account_info(self, paper_broker):
        """Test account info retrieval"""
        account_info = paper_broker.get_account_info()
        
        assert "balances" in account_info
        assert len(account_info["balances"]) == 2
        
        balance_assets = [b["asset"] for b in account_info["balances"]]
        assert "BTC" in balance_assets
        assert "USDT" in balance_assets


class TestExecRateLimitError:
    """Test RateLimitError exception"""
    
    def test_rate_limit_error_basic(self):
        """Test basic RateLimitError creation"""
        error = RateLimitError("Rate limit exceeded")
        
        assert str(error) == "Rate limit exceeded"
        assert error.retry_after is None
    
    def test_rate_limit_error_with_retry_after(self):
        """Test RateLimitError with retry_after"""
        error = RateLimitError("Rate limit exceeded", retry_after=30.0)
        
        assert error.retry_after == 30.0


class TestExecRateLimitConfig:
    """Test RateLimitConfig data class"""
    
    def test_rate_limit_config_defaults(self):
        """Test RateLimitConfig default values"""
        config = RateLimitConfig()
        
        assert config.requests_per_minute == 1200
        assert config.burst_allowance == 10
        assert config.weight_per_minute == 6000
        assert config.order_limit_per_second == 10
        assert config.order_limit_per_day == 200000
    
    def test_rate_limit_config_custom(self):
        """Test RateLimitConfig with custom values"""
        config = RateLimitConfig(
            requests_per_minute=600,
            weight_per_minute=3000,
            order_limit_per_second=5
        )
        
        assert config.requests_per_minute == 600
        assert config.weight_per_minute == 3000
        assert config.order_limit_per_second == 5


class TestExecTokenBucket:
    """Test TokenBucket implementation"""
    
    def test_token_bucket_initialization(self):
        """Test TokenBucket initialization"""
        bucket = TokenBucket(capacity=100, refill_rate=10.0)
        
        assert bucket.capacity == 100
        assert bucket.tokens == 100
        assert bucket.refill_rate == 10.0
    
    @pytest.mark.asyncio
    async def test_token_bucket_acquire_success(self):
        """Test successful token acquisition"""
        bucket = TokenBucket(capacity=100, refill_rate=10.0)
        
        result = await bucket.acquire(10)
        
        assert result is True
        assert bucket.tokens == 90
    
    @pytest.mark.asyncio
    async def test_token_bucket_acquire_insufficient(self):
        """Test token acquisition with insufficient tokens"""
        bucket = TokenBucket(capacity=100, refill_rate=10.0)
        
        result = await bucket.acquire(150)
        
        assert result is False
        assert bucket.tokens == 100  # No tokens consumed
    
    @pytest.mark.asyncio
    async def test_token_bucket_refill(self):
        """Test token bucket refill over time"""
        bucket = TokenBucket(capacity=100, refill_rate=10.0)
        
        # Consume tokens
        await bucket.acquire(50)
        assert bucket.tokens == 50
        
        # Simulate time passage
        bucket.last_refill -= 1.0  # 1 second ago
        
        # Try to acquire again - should trigger refill
        result = await bucket.acquire(1)
        
        assert result is True
        # Should have refilled 10 tokens (10/second * 1 second)
        assert bucket.tokens == 59  # 50 + 10 - 1
    
    def test_time_to_refill(self):
        """Test time to refill calculation"""
        bucket = TokenBucket(capacity=100, refill_rate=10.0)
        bucket.tokens = 30
        
        # Need 50 tokens, have 30, need 20 more
        # At 10 tokens/second, need 2 seconds
        time_needed = bucket.time_to_refill(50)
        assert time_needed == 2.0
        
        # Already have enough tokens
        time_needed = bucket.time_to_refill(20)
        assert time_needed == 0.0


class TestExecRateLimiter:
    """Test RateLimiter functionality"""
    
    @pytest.fixture
    def rate_config(self):
        """Fixture for rate limit config"""
        return RateLimitConfig(
            requests_per_minute=60,  # 1 per second for testing
            weight_per_minute=300,   # 5 per second for testing
            order_limit_per_second=2,
            order_limit_per_day=100
        )
    
    @pytest.fixture
    def rate_limiter(self, rate_config):
        """Fixture for RateLimiter"""
        return RateLimiter(rate_config)
    
    def test_rate_limiter_initialization(self, rate_limiter, rate_config):
        """Test RateLimiter initialization"""
        assert rate_limiter.config == rate_config
        assert rate_limiter.request_bucket is not None
        assert rate_limiter.weight_bucket is not None
        assert rate_limiter.order_bucket is not None
        assert rate_limiter.daily_orders == 0
    
    @pytest.mark.asyncio
    async def test_acquire_success(self, rate_limiter):
        """Test successful token acquisition"""
        result = await rate_limiter.acquire(weight=1)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_acquire_order_success(self, rate_limiter):
        """Test successful order token acquisition"""
        result = await rate_limiter.acquire_order()
        
        assert result is True
        assert rate_limiter.daily_orders == 1
    
    @pytest.mark.asyncio
    async def test_acquire_order_daily_limit(self, rate_limiter):
        """Test order acquisition with daily limit exceeded"""
        rate_limiter.daily_orders = rate_limiter.config.order_limit_per_day
        
        result = await rate_limiter.acquire_order()
        
        assert result is False
    
    def test_calculate_retry_delay(self, rate_limiter):
        """Test retry delay calculation"""
        delay_0 = rate_limiter._calculate_retry_delay(0)
        delay_2 = rate_limiter._calculate_retry_delay(2)
        
        assert delay_0 >= 1.0
        assert delay_2 >= 4.0  # Exponential backoff
        assert delay_2 > delay_0
    
    @pytest.mark.asyncio
    async def test_wait_for_capacity(self, rate_limiter):
        """Test waiting for capacity"""
        # Exhaust tokens
        rate_limiter.request_bucket.tokens = 0
        rate_limiter.weight_bucket.tokens = 0
        
        start_time = time.time()
        await rate_limiter.wait_for_capacity(weight=1)
        end_time = time.time()
        
        # Should have waited some time for refill
        assert end_time > start_time
    
    def test_get_status(self, rate_limiter):
        """Test status retrieval"""
        status = rate_limiter.get_status()
        
        assert "request_bucket" in status
        assert "weight_bucket" in status
        assert "order_bucket" in status
        assert "daily_orders" in status
        assert "config" in status
        
        assert status["daily_orders"] == 0
        assert status["daily_limit"] == rate_limiter.config.order_limit_per_day
    
    def test_reset_buckets(self, rate_limiter):
        """Test bucket reset functionality"""
        # Consume some tokens
        rate_limiter.request_bucket.tokens = 10
        rate_limiter.weight_bucket.tokens = 50
        rate_limiter.consecutive_retries = 5
        
        rate_limiter.reset_buckets()
        
        assert rate_limiter.request_bucket.tokens == rate_limiter.request_bucket.capacity
        assert rate_limiter.weight_bucket.tokens == rate_limiter.weight_bucket.capacity
        assert rate_limiter.consecutive_retries == 0


class TestExecAdaptiveRateLimiter:
    """Test AdaptiveRateLimiter functionality"""
    
    @pytest.fixture
    def adaptive_limiter(self):
        """Fixture for AdaptiveRateLimiter"""
        config = RateLimitConfig(requests_per_minute=60, weight_per_minute=300)
        return AdaptiveRateLimiter(config)
    
    def test_adaptive_rate_limiter_initialization(self, adaptive_limiter):
        """Test AdaptiveRateLimiter initialization"""
        assert adaptive_limiter.adaptive_factor == 1.0
        assert adaptive_limiter.recent_errors == []
        assert adaptive_limiter.error_window == 300
    
    def test_record_error_high_rate(self, adaptive_limiter):
        """Test error recording with high error rate"""
        # Simulate many errors
        current_time = time.time()
        for i in range(10):
            adaptive_limiter.recent_errors.append((current_time - i, "rate_limit"))
        
        initial_factor = adaptive_limiter.adaptive_factor
        adaptive_limiter.record_error("rate_limit")
        
        # Should reduce adaptive factor
        assert adaptive_limiter.adaptive_factor < initial_factor
    
    def test_record_error_low_rate(self, adaptive_limiter):
        """Test error recording with low error rate"""
        adaptive_limiter.adaptive_factor = 0.8  # Start with reduced factor
        
        # Record single error (low rate)
        adaptive_limiter.record_error("network")
        
        # Should gradually increase factor
        assert adaptive_limiter.adaptive_factor > 0.8
    
    @pytest.mark.asyncio
    async def test_adaptive_acquire(self, adaptive_limiter):
        """Test adaptive acquire method"""
        adaptive_limiter.adaptive_factor = 0.5  # Reduced factor
        
        with patch.object(RateLimiter, 'acquire') as mock_acquire:
            mock_acquire.return_value = True
            
            result = await adaptive_limiter.acquire(weight=10)
            
            # Should call parent with adjusted weight
            mock_acquire.assert_called_once_with(20, 3)  # 10 / 0.5 = 20
            assert result is True


class TestExecUtilityFunctions:
    """Test utility functions in rate_limit module"""
    
    def test_create_conservative_rate_limiter(self):
        """Test creation of conservative rate limiter"""
        limiter = create_conservative_rate_limiter()
        
        assert isinstance(limiter, RateLimiter)
        assert limiter.config.requests_per_minute == 600
        assert limiter.config.weight_per_minute == 3000
        assert limiter.config.order_limit_per_second == 5
    
    def test_create_aggressive_rate_limiter(self):
        """Test creation of aggressive rate limiter"""
        limiter = create_aggressive_rate_limiter()
        
        assert isinstance(limiter, RateLimiter)
        assert limiter.config.requests_per_minute == 1100
        assert limiter.config.weight_per_minute == 5500
        assert limiter.config.order_limit_per_second == 9


class TestExecCreateBinanceClient:
    """Test create_binance_client utility function"""
    
    @pytest.mark.asyncio
    async def test_create_binance_client_with_params(self):
        """Test client creation with explicit parameters"""
        with patch('exec.binance.BinanceClient') as mock_client_class:
            mock_instance = AsyncMock()
            mock_client_class.return_value = mock_instance
            
            client = await create_binance_client(
                api_key="test_key",
                api_secret="test_secret",
                testnet=True
            )
            
            assert client == mock_instance
            mock_instance.start_session.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_binance_client_no_params_no_config(self):
        """Test client creation without params and no unified config"""
        with patch('exec.binance._unified_config_available', False):
            with pytest.raises(ValueError, match="No API credentials provided"):
                await create_binance_client()
    
    @pytest.mark.asyncio
    async def test_create_binance_client_missing_credentials(self):
        """Test client creation with missing credentials"""
        with pytest.raises(ValueError, match="API key and secret are required"):
            await create_binance_client(api_key="test_key")  # Missing secret


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
