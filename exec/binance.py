"""
Binance - Official API client wrapper with throttling

This module provides a wrapper around the official Binance API client
with added throttling, error handling, and retry logic.
"""

# Standard library imports
import asyncio
import hashlib
import hmac
import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable

# Third-party imports
import aiohttp

# Local imports
from exec.rate_limit import RateLimiter, RateLimitConfig
from core.state import Order, OrderSide, OrderType, OrderStatus

# Try to import unified config, fall back to None if not available
try:
    from config import get_settings

    _unified_config_available = True
except ImportError:
    _unified_config_available = False

logger = logging.getLogger(__name__)


def mask_api_key(api_key: str) -> str:
    """Mask API key for safe logging"""
    if not api_key or len(api_key) < 8:
        return "***"
    return f"{api_key[:4]}***{api_key[-4:]}"


@dataclass
class BinanceConfig:
    """Binance API configuration"""

    api_key: str
    api_secret: str
    base_url: str = "https://api.binance.com"
    testnet: bool = False
    timeout: int = 30
    max_retries: int = 3
    retry_delay_base: float = 1.0
    retry_delay_max: float = 60.0
    retry_jitter: float = 0.1

    def __post_init__(self) -> None:
        if self.testnet:
            self.base_url = "https://testnet.binance.vision"
        logger.info(
            f"Binance client configured for {'testnet' if self.testnet else 'mainnet'} with API key {mask_api_key(self.api_key)}"
        )

    @classmethod
    def from_unified_config(cls, config: Optional[Any] = None) -> "BinanceConfig":
        """Create BinanceConfig from unified configuration system

        Args:
            config: Settings instance from unified config system. If None,
                   will get current settings.

        Returns:
            BinanceConfig: Configured instance

        Raises:
            ImportError: If unified config system is not available
        """
        if not _unified_config_available:
            raise ImportError(
                "Unified config system not available. Install required dependencies."
            )

        if config is None:
            config = get_settings()

        return cls(
            api_key=config.binance.api_key or "",
            api_secret=config.binance.api_secret or "",
            base_url=config.binance.base_url
            or (
                "https://testnet.binance.vision"
                if config.binance.testnet
                else "https://api.binance.com"
            ),
            testnet=config.binance.testnet,
            timeout=30,  # Could be added to unified config if needed
            max_retries=3,  # Could be added to unified config if needed
            retry_delay_base=1.0,  # Could be added to unified config if needed
            retry_delay_max=60.0,  # Could be added to unified config if needed
            retry_jitter=0.1,  # Could be added to unified config if needed
        )


class BinanceError(Exception):
    """Base exception for Binance API errors.

    Attributes:
        code: Optional error code from Binance API response
        response: Original response dict from Binance API
    """

    def __init__(
        self, message: str, code: Optional[int] = None, response: Optional[Dict] = None
    ) -> None:
        """Initialize BinanceError.

        Args:
            message: Error message
            code: Optional error code from API response
            response: Optional raw response data from API
        """
        super().__init__(message)
        self.code = code
        self.response = response or {}


class BinanceRateLimitError(BinanceError):
    """Rate limit exceeded error.

    Raised when API rate limits are exceeded and requests need to be throttled.
    """

    pass


class BinanceClient:
    """Binance API client with built-in rate limiting and error handling.

    This client provides async HTTP communication with the Binance API,
    including automatic retry logic, rate limiting, and proper error handling.

    Attributes:
        config: Configuration settings for the client
        session: Async HTTP session for API requests
        rate_limiter: Rate limiting handler
        server_time_offset: Offset to sync with Binance server time
    """

    def __init__(self, config: BinanceConfig) -> None:
        """Initialize Binance API client.

        Args:
            config: Configuration object containing API credentials and settings
        """
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None

        # Rate limiting configuration
        rate_config = RateLimitConfig(
            requests_per_minute=1200,  # Binance limit
            burst_allowance=10,
            weight_per_minute=6000,  # Weight-based limit
            order_limit_per_second=10,
            order_limit_per_day=200000,
        )
        self.rate_limiter = RateLimiter(rate_config)

        # Track server time offset
        self.server_time_offset = 0
        self.last_time_sync = 0

    async def __aenter__(self) -> "BinanceClient":
        """Async context manager entry.

        Returns:
            BinanceClient: Self for context manager usage
        """
        await self.start_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_session()

    async def start_session(self):
        """Initialize HTTP session"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
            await self._sync_server_time()

    async def close_session(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def _sync_server_time(self):
        """Synchronize with Binance server time"""
        try:
            response = await self._make_request("GET", "/api/v3/time")
            server_time = response["serverTime"]
            local_time = int(time.time() * 1000)
            self.server_time_offset = server_time - local_time
            self.last_time_sync = time.time()
            logger.info(f"Synced with server time, offset: {self.server_time_offset}ms")
        except Exception as e:
            logger.warning(f"Failed to sync server time: {e}")

    def _get_timestamp(self) -> int:
        """Get current timestamp adjusted for server time"""
        if time.time() - self.last_time_sync > 3600:  # Resync every hour
            asyncio.create_task(self._sync_server_time())

        return int(time.time() * 1000) + self.server_time_offset

    def _generate_signature(self, query_string: str) -> str:
        """Generate HMAC SHA256 signature"""
        return hmac.new(
            self.config.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter"""
        base_delay = self.config.retry_delay_base * (2**attempt)
        delay = min(base_delay, self.config.retry_delay_max)
        jitter = random.uniform(0, self.config.retry_jitter * delay)
        return delay + jitter

    async def _make_request_with_retry(
        self,
        method: str,
        endpoint: str,
        params: Dict = None,
        signed: bool = False,
        weight: int = 1,
    ) -> Dict:
        """Make HTTP request with bounded retry and jitter"""
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                # Apply rate limiting with retry
                rate_limit_acquired = await self.rate_limiter.acquire(
                    weight=weight, retries=2
                )
                if not rate_limit_acquired:
                    raise BinanceRateLimitError(
                        "Rate limit acquisition failed after retries"
                    )

                return await self._make_request_single(
                    method, endpoint, params, signed, weight
                )

            except BinanceRateLimitError as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(
                        f"Rate limit hit, retrying in {delay:.2f}s (attempt {attempt + 1}/{self.config.max_retries + 1})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Rate limit retries exhausted after {self.config.max_retries} attempts"
                    )
                    raise

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exception = BinanceError(f"Network error: {e}")
                if attempt < self.config.max_retries:
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(
                        f"Network error, retrying in {delay:.2f}s (attempt {attempt + 1}/{self.config.max_retries + 1}): {e}"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Network retries exhausted after {self.config.max_retries} attempts"
                    )
                    raise last_exception

            except BinanceError as e:
                # For API errors, only retry on specific error codes
                if (
                    e.code in [429, 418, -1021] and attempt < self.config.max_retries
                ):  # Rate limit, banned, timestamp errors
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(
                        f"API error {e.code}, retrying in {delay:.2f}s (attempt {attempt + 1}/{self.config.max_retries + 1}): {e}"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"API error not retryable or retries exhausted: {e}")
                    raise

            except Exception as e:
                last_exception = BinanceError(f"Unexpected error: {e}")
                logger.error(
                    f"Unexpected error in request (attempt {attempt + 1}): {e}"
                )
                if attempt >= self.config.max_retries:
                    raise last_exception

        # Should not reach here, but in case
        if last_exception:
            raise last_exception
        raise BinanceError("Request failed after all retry attempts")

    async def _make_request_single(
        self,
        method: str,
        endpoint: str,
        params: Dict = None,
        signed: bool = False,
        weight: int = 1,
    ) -> Dict:
        """Make a single HTTP request to Binance API (no retry logic)"""
        if not self.session:
            await self.start_session()

        # Apply rate limiting
        await self.rate_limiter.acquire(weight=weight)

        url = f"{self.config.base_url}{endpoint}"
        params = params or {}

        headers = {"X-MBX-APIKEY": self.config.api_key}

        if signed:
            params["timestamp"] = self._get_timestamp()
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            params["signature"] = self._generate_signature(query_string)

        try:
            async with self.session.request(
                method, url, params=params, headers=headers
            ) as response:
                data = await response.json()

                if response.status == 429:
                    raise BinanceRateLimitError(
                        "Rate limit exceeded", response.status, data
                    )
                elif response.status >= 400:
                    error_msg = data.get("msg", f"HTTP {response.status}")
                    error_code = data.get("code", response.status)
                    raise BinanceError(error_msg, error_code, data)

                return data

        except aiohttp.ClientError as e:
            raise BinanceError(f"Network error: {e}")
        except json.JSONDecodeError as e:
            raise BinanceError(f"Invalid JSON response: {e}")

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Dict = None,
        signed: bool = False,
        weight: int = 1,
    ) -> Dict:
        """Make HTTP request with retry logic and rate limiting"""
        return await self._make_request_with_retry(
            method, endpoint, params, signed, weight
        )

    # Market Data Methods

    async def get_server_time(self) -> Dict:
        """Get server time"""
        return await self._make_request("GET", "/api/v3/time")

    async def get_exchange_info(self, symbol: str = None) -> Dict:
        """Get exchange trading rules and symbol information"""
        params = {"symbol": symbol} if symbol else {}
        return await self._make_request("GET", "/api/v3/exchangeInfo", params)

    async def get_ticker_price(self, symbol: str = None) -> Dict:
        """Get latest price for symbol(s)"""
        params = {"symbol": symbol} if symbol else {}
        return await self._make_request("GET", "/api/v3/ticker/price", params)

    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book depth"""
        params = {"symbol": symbol, "limit": limit}
        return await self._make_request("GET", "/api/v3/depth", params)

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: int = None,
        end_time: int = None,
        limit: int = 500,
    ) -> List:
        """Get kline/candlestick data"""
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        return await self._make_request("GET", "/api/v3/klines", params)

    # Account Methods

    async def get_account_info(self) -> Dict:
        """Get account information"""
        return await self._make_request(
            "GET", "/api/v3/account", signed=True, weight=10
        )

    async def get_balances(self) -> List[Dict]:
        """Get account balances"""
        account = await self.get_account_info()
        return account.get("balances", [])

    # Trading Methods

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Decimal = None,
        time_in_force: str = "GTC",
        client_order_id: str = None,
    ) -> Dict:
        """Place a new order"""
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(quantity),
        }

        if price is not None:
            params["price"] = str(price)
        if time_in_force:
            params["timeInForce"] = time_in_force
        if client_order_id:
            params["newClientOrderId"] = client_order_id

        # Apply order rate limiting
        await self.rate_limiter.acquire_order()

        return await self._make_request(
            "POST", "/api/v3/order", params, signed=True, weight=1
        )

    async def cancel_order(
        self, symbol: str, order_id: int = None, client_order_id: str = None
    ) -> Dict:
        """Cancel an active order"""
        params = {"symbol": symbol}

        if order_id:
            params["orderId"] = order_id
        elif client_order_id:
            params["origClientOrderId"] = client_order_id
        else:
            raise ValueError("Either order_id or client_order_id must be provided")

        return await self._make_request(
            "DELETE", "/api/v3/order", params, signed=True, weight=1
        )

    async def get_order(
        self, symbol: str, order_id: int = None, client_order_id: str = None
    ) -> Dict:
        """Get order status"""
        params = {"symbol": symbol}

        if order_id:
            params["orderId"] = order_id
        elif client_order_id:
            params["origClientOrderId"] = client_order_id
        else:
            raise ValueError("Either order_id or client_order_id must be provided")

        return await self._make_request(
            "GET", "/api/v3/order", params, signed=True, weight=2
        )

    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get all open orders"""
        params = {"symbol": symbol} if symbol else {}
        weight = 3 if symbol else 40
        return await self._make_request(
            "GET", "/api/v3/openOrders", params, signed=True, weight=weight
        )

    async def cancel_all_orders(self, symbol: str) -> List[Dict]:
        """Cancel all open orders for a symbol"""
        params = {"symbol": symbol}
        return await self._make_request(
            "DELETE", "/api/v3/openOrders", params, signed=True, weight=1
        )

    async def get_order_history(self, symbol: str, limit: int = 500) -> List[Dict]:
        """Get order history"""
        params = {"symbol": symbol, "limit": limit}
        return await self._make_request(
            "GET", "/api/v3/allOrders", params, signed=True, weight=10
        )

    async def get_trades(self, symbol: str, limit: int = 500) -> List[Dict]:
        """Get trade history"""
        params = {"symbol": symbol, "limit": limit}
        return await self._make_request(
            "GET", "/api/v3/myTrades", params, signed=True, weight=10
        )

    # Utility Methods

    def convert_order_to_internal(self, binance_order: Dict) -> Order:
        """Convert Binance order format to internal Order object"""
        return Order(
            order_id=str(binance_order["orderId"]),
            client_order_id=binance_order.get("clientOrderId", ""),
            symbol=binance_order["symbol"],
            side=OrderSide(binance_order["side"]),
            order_type=OrderType(binance_order["type"]),
            quantity=Decimal(binance_order["origQty"]),
            price=Decimal(binance_order.get("price", "0")),
            status=OrderStatus(binance_order["status"]),
            filled_quantity=Decimal(binance_order.get("executedQty", "0")),
            filled_price=Decimal(
                binance_order.get("price", "0")
            ),  # Avg price not always available
            fee=Decimal("0"),  # Would need separate call to get fees
            timestamp=datetime.fromtimestamp(binance_order["time"] / 1000),
            update_time=datetime.fromtimestamp(binance_order["updateTime"] / 1000),
        )

    async def health_check(self) -> bool:
        """Check if API is accessible"""
        try:
            await self.get_server_time()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def get_rate_limit_status(self) -> Dict:
        """Get current rate limit status"""
        return self.rate_limiter.get_status()


# Convenience functions


async def create_binance_client(
    api_key: str = None, api_secret: str = None, testnet: bool = None
) -> BinanceClient:
    """Create and initialize Binance client

    Args:
        api_key: Binance API key (if None, will try to get from unified config)
        api_secret: Binance API secret (if None, will try to get from unified config)
        testnet: Use testnet (if None, will try to get from unified config)
    """
    # If no parameters provided, try to use unified config
    if api_key is None and api_secret is None and testnet is None:
        if _unified_config_available:
            config = BinanceConfig.from_unified_config()
        else:
            raise ValueError(
                "No API credentials provided and unified config not available"
            )
    else:
        # Use provided parameters or fallback to unified config for missing ones
        if _unified_config_available and (
            api_key is None or api_secret is None or testnet is None
        ):
            unified_settings = get_settings()
            api_key = api_key or unified_settings.binance.api_key
            api_secret = api_secret or unified_settings.binance.api_secret
            testnet = (
                testnet if testnet is not None else unified_settings.binance.testnet
            )

        if not api_key or not api_secret:
            raise ValueError("API key and secret are required")

        config = BinanceConfig(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet if testnet is not None else False,
        )

    client = BinanceClient(config)
    await client.start_session()
    return client
