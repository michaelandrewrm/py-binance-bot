"""
Test Configuration and Utilities

Configuration for pytest and common test utilities.
"""

import pytest
import asyncio
import tempfile
import os
import shutil
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock

# pytest configuration
pytest_plugins = []


# Test event loop configuration for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Common test fixtures
@pytest.fixture
def mock_binance_client():
    """Create mock Binance client"""
    client = AsyncMock()
    client.get_server_time.return_value = {
        "serverTime": int(datetime.now(timezone.utc).timestamp() * 1000)
    }
    client.get_exchange_info.return_value = {
        "symbols": [
            {
                "symbol": "BTCUSDC",
                "status": "TRADING",
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                    {
                        "filterType": "LOT_SIZE",
                        "stepSize": "0.00001",
                        "minQty": "0.00001",
                        "maxQty": "1000",
                    },
                    {"filterType": "MIN_NOTIONAL", "minNotional": "10"},
                ],
            }
        ]
    }
    client.get_ticker_price.return_value = {"symbol": "BTCUSDC", "price": "50000.00"}
    client.place_order.return_value = {
        "orderId": 12345,
        "symbol": "BTCUSDC",
        "side": "BUY",
        "type": "MARKET",
        "origQty": "0.001",
        "status": "FILLED",
        "executedQty": "0.001",
        "time": int(datetime.now(timezone.utc).timestamp() * 1000),
        "updateTime": int(datetime.now(timezone.utc).timestamp() * 1000),
    }
    return client


@pytest.fixture
def sample_klines():
    """Create sample kline data"""
    from data.schema import KlineData

    base_time = datetime.now(timezone.utc) - timedelta(days=7)
    klines = []

    for i in range(168):  # 7 days of hourly data
        open_price = Decimal("50000") + Decimal(i * 10)
        high_price = open_price + Decimal("100")
        low_price = open_price - Decimal("50")
        close_price = open_price + Decimal(i % 20 - 10)  # Some variation

        kline = KlineData(
            symbol="BTCUSDC",
            interval="1h",
            open_time=base_time + timedelta(hours=i),
            close_time=base_time + timedelta(hours=i + 1),
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            volume=Decimal("100"),
            quote_volume=open_price * Decimal("100"),
            trades=1000,
            taker_buy_volume=Decimal("60"),
            taker_buy_quote_volume=open_price * Decimal("60"),
        )
        klines.append(kline)

    return klines


@pytest.fixture
def sample_trades():
    """Create sample trade data"""
    from data.schema import TradeRecord

    trades = []
    base_time = datetime.now(timezone.utc) - timedelta(days=30)

    for i in range(20):
        trade = TradeRecord(
            symbol="BTCUSDC",
            side="BUY" if i % 2 == 0 else "SELL",
            order_type="MARKET",
            quantity=Decimal("0.001"),
            price=Decimal("50000") + Decimal(i * 100),
            timestamp=base_time + timedelta(days=i),
            pnl=Decimal("10") if i % 3 == 0 else Decimal("-5"),
            fee=Decimal("0.1"),
            strategy_name="test_strategy",
        )
        trades.append(trade)

    return trades


@pytest.fixture
def temp_directory():
    """Create temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_database():
    """Create temporary database file for tests"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def mock_exchange_info():
    """Create mock exchange information"""
    from core.sizing import ExchangeInfo

    return ExchangeInfo(
        symbol="BTCUSDC",
        base_precision=8,
        quote_precision=8,
        min_qty=Decimal("0.00001"),
        max_qty=Decimal("1000"),
        step_size=Decimal("0.00001"),
        tick_size=Decimal("0.01"),
        min_notional=Decimal("10"),
    )


@pytest.fixture
def mock_fee_config():
    """Create mock fee configuration"""
    from core.sizing import FeeConfig

    return FeeConfig(
        maker_fee=Decimal("0.001"),
        taker_fee=Decimal("0.001"),
        withdraw_fee=Decimal("0.0005"),
    )


# Test utilities
class TestDataGenerator:
    """Utility class for generating test data"""

    @staticmethod
    def create_kline(symbol="BTCUSDC", interval="1h", base_price=50000, timestamp=None):
        """Create a single kline for testing"""
        from data.schema import KlineData

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        return KlineData(
            symbol=symbol,
            interval=interval,
            open_time=timestamp,
            close_time=timestamp + timedelta(hours=1),
            open_price=Decimal(str(base_price)),
            high_price=Decimal(str(base_price + 100)),
            low_price=Decimal(str(base_price - 50)),
            close_price=Decimal(str(base_price + 25)),
            volume=Decimal("100"),
            quote_volume=Decimal(str(base_price * 100)),
            trades=1000,
            taker_buy_volume=Decimal("60"),
            taker_buy_quote_volume=Decimal(str(base_price * 60)),
        )

    @staticmethod
    def create_trade(
        symbol="BTCUSDC", side="BUY", price=50000, quantity=0.001, timestamp=None
    ):
        """Create a single trade for testing"""
        from data.schema import TradeRecord

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        return TradeRecord(
            symbol=symbol,
            side=side,
            order_type="MARKET",
            quantity=Decimal(str(quantity)),
            price=Decimal(str(price)),
            timestamp=timestamp,
            pnl=Decimal("10"),
            fee=Decimal("0.1"),
            strategy_name="test_strategy",
        )

    @staticmethod
    def create_order(symbol="BTCUSDC", side="BUY", price=50000, quantity=0.001):
        """Create a single order for testing"""
        from core.state import Order, OrderSide, OrderType, OrderStatus

        return Order(
            order_id="test_order_123",
            client_order_id="test_client_order_123",
            symbol=symbol,
            side=OrderSide(side),
            order_type=OrderType.LIMIT,
            quantity=Decimal(str(quantity)),
            price=Decimal(str(price)),
            status=OrderStatus.NEW,
            timestamp=datetime.now(timezone.utc),
        )


class AsyncMockHelper:
    """Helper for working with async mocks"""

    @staticmethod
    def make_async_mock(**kwargs):
        """Create an async mock with return values"""
        mock = AsyncMock()
        for key, value in kwargs.items():
            setattr(mock, key, AsyncMock(return_value=value))
        return mock


# Custom pytest markers
pytestmark = pytest.mark.asyncio


# Test configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Mark async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)

        # Mark slow tests
        if "test_concurrent" in item.name or "test_performance" in item.name:
            item.add_marker(pytest.mark.slow)

        # Mark integration tests
        if "integration" in item.name or "end_to_end" in item.name:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)


# Environment setup for tests
@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment"""
    # Set test-specific environment variables
    monkeypatch.setenv("TRADING_BOT_ENV", "test")
    monkeypatch.setenv("TRADING_BOT_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("TRADING_BOT_REPO_TYPE", "sqlite")

    # Mock API keys for tests (don't use real ones in tests)
    monkeypatch.setenv("BINANCE_API_KEY", "test_api_key_for_testing")
    monkeypatch.setenv("BINANCE_API_SECRET", "test_api_secret_for_testing")
