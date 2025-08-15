"""
Rate Limit - Leaky bucket with jitter-retry

This module implements rate limiting using leaky bucket algorithm
with jitter-based retry logic for API throttling.
"""

import asyncio
import time
import random
import logging
from typing import Optional, Dict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Exception raised when rate limits are exceeded"""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""

    requests_per_minute: int = 1200
    burst_allowance: int = 10
    weight_per_minute: int = 6000
    order_limit_per_second: int = 10
    order_limit_per_day: int = 200000
    retry_delay_base: float = 1.0
    retry_delay_max: float = 60.0
    retry_jitter: float = 0.1


class TokenBucket:
    """Token bucket for rate limiting"""

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens from bucket

        Returns:
            True if tokens acquired, False if not enough tokens
        """
        async with self._lock:
            now = time.time()

            # Refill tokens based on time elapsed
            time_elapsed = now - self.last_refill
            tokens_to_add = time_elapsed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now

            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    def time_to_refill(self, tokens: int) -> float:
        """Calculate time needed to refill enough tokens"""
        if self.tokens >= tokens:
            return 0.0

        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate

    def get_status(self) -> Dict:
        """Get current bucket status"""
        return {
            "capacity": self.capacity,
            "current_tokens": self.tokens,
            "refill_rate": self.refill_rate,
            "last_refill": self.last_refill,
        }


class RateLimiter:
    """
    Advanced rate limiter with multiple buckets and retry logic
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config

        # Request rate bucket (requests per minute)
        self.request_bucket = TokenBucket(
            capacity=config.requests_per_minute + config.burst_allowance,
            refill_rate=config.requests_per_minute / 60.0,  # per second
        )

        # Weight bucket (for API weight limits)
        self.weight_bucket = TokenBucket(
            capacity=config.weight_per_minute,
            refill_rate=config.weight_per_minute / 60.0,  # per second
        )

        # Order rate bucket (orders per second)
        self.order_bucket = TokenBucket(
            capacity=config.order_limit_per_second,
            refill_rate=config.order_limit_per_second,  # per second
        )

        # Daily order counter
        self.daily_orders = 0
        self.daily_reset_time = self._get_next_daily_reset()

        # Retry tracking
        self.consecutive_retries = 0
        self.last_retry_time = 0

    def _get_next_daily_reset(self) -> datetime:
        """Get the next daily reset time (midnight UTC)"""
        now = datetime.utcnow()
        next_reset = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
            days=1
        )
        return next_reset

    def _reset_daily_counter_if_needed(self):
        """Reset daily order counter if new day"""
        if datetime.utcnow() >= self.daily_reset_time:
            self.daily_orders = 0
            self.daily_reset_time = self._get_next_daily_reset()
            logger.info("Daily order counter reset")

    async def acquire(self, weight: int = 1, retries: int = 3) -> bool:
        """
        Acquire rate limit tokens with retry logic

        Args:
            weight: API weight of the request
            retries: Number of retries if rate limited

        Returns:
            True if acquired, False if failed after retries
        """
        for attempt in range(retries + 1):
            # Try to acquire both request and weight tokens
            request_acquired = await self.request_bucket.acquire(1)
            weight_acquired = await self.weight_bucket.acquire(weight)

            if request_acquired and weight_acquired:
                self.consecutive_retries = 0
                return True

            # If we couldn't acquire, calculate wait time
            if attempt < retries:
                wait_time = self._calculate_retry_delay(attempt)
                logger.debug(
                    f"Rate limited, waiting {wait_time:.2f}s (attempt {attempt + 1}/{retries + 1})"
                )
                await asyncio.sleep(wait_time)

        # Failed after all retries
        self.consecutive_retries += 1
        logger.warning(f"Failed to acquire rate limit after {retries} retries")
        return False

    async def acquire_order(self, retries: int = 5) -> bool:
        """
        Acquire tokens for order placement with additional checks

        Args:
            retries: Number of retries if rate limited

        Returns:
            True if acquired, False if failed
        """
        self._reset_daily_counter_if_needed()

        # Check daily limit
        if self.daily_orders >= self.config.order_limit_per_day:
            logger.error("Daily order limit exceeded")
            return False

        # Try to acquire order tokens
        for attempt in range(retries + 1):
            if await self.order_bucket.acquire(1):
                self.daily_orders += 1
                self.consecutive_retries = 0
                return True

            if attempt < retries:
                wait_time = self._calculate_retry_delay(attempt)
                logger.debug(f"Order rate limited, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)

        self.consecutive_retries += 1
        logger.warning("Failed to acquire order rate limit")
        return False

    def _calculate_retry_delay(self, attempt: int) -> float:
        """
        Calculate retry delay with exponential backoff and jitter

        Args:
            attempt: Current retry attempt (0-based)

        Returns:
            Delay in seconds
        """
        # Exponential backoff
        base_delay = self.config.retry_delay_base * (2**attempt)

        # Cap at maximum delay
        delay = min(base_delay, self.config.retry_delay_max)

        # Add jitter to avoid thundering herd
        jitter = random.uniform(0, self.config.retry_jitter * delay)

        return delay + jitter

    async def wait_for_capacity(self, weight: int = 1) -> float:
        """
        Wait until there's enough capacity

        Returns:
            Time waited in seconds
        """
        start_time = time.time()

        while True:
            request_time = self.request_bucket.time_to_refill(1)
            weight_time = self.weight_bucket.time_to_refill(weight)

            wait_time = max(request_time, weight_time)

            if wait_time <= 0:
                break

            logger.debug(f"Waiting {wait_time:.2f}s for rate limit capacity")
            await asyncio.sleep(wait_time)

        return time.time() - start_time

    def get_status(self) -> Dict:
        """Get comprehensive rate limiter status"""
        self._reset_daily_counter_if_needed()

        return {
            "request_bucket": self.request_bucket.get_status(),
            "weight_bucket": self.weight_bucket.get_status(),
            "order_bucket": self.order_bucket.get_status(),
            "daily_orders": self.daily_orders,
            "daily_limit": self.config.order_limit_per_day,
            "daily_reset_time": self.daily_reset_time.isoformat(),
            "consecutive_retries": self.consecutive_retries,
            "config": {
                "requests_per_minute": self.config.requests_per_minute,
                "weight_per_minute": self.config.weight_per_minute,
                "order_limit_per_second": self.config.order_limit_per_second,
                "order_limit_per_day": self.config.order_limit_per_day,
            },
        }

    def reset_buckets(self):
        """Reset all buckets to full capacity (use with caution)"""
        self.request_bucket.tokens = self.request_bucket.capacity
        self.weight_bucket.tokens = self.weight_bucket.capacity
        self.order_bucket.tokens = self.order_bucket.capacity
        self.consecutive_retries = 0
        logger.warning("All rate limit buckets manually reset")


class AdaptiveRateLimiter(RateLimiter):
    """
    Rate limiter that adapts to server responses and adjusts limits
    """

    def __init__(self, config: RateLimitConfig):
        super().__init__(config)
        self.adaptive_factor = 1.0
        self.recent_errors = []
        self.error_window = 300  # 5 minutes

    def record_error(self, error_type: str):
        """Record an error for adaptive adjustment"""
        now = time.time()
        self.recent_errors.append((now, error_type))

        # Clean old errors
        cutoff = now - self.error_window
        self.recent_errors = [(t, e) for t, e in self.recent_errors if t > cutoff]

        # Adjust limits based on error rate
        error_rate = (
            len(self.recent_errors) / self.error_window * 60
        )  # errors per minute

        if error_rate > 5:  # More than 5 errors per minute
            self.adaptive_factor = max(0.5, self.adaptive_factor * 0.9)
            logger.warning(
                f"High error rate detected, reducing limits by factor {self.adaptive_factor}"
            )
        elif error_rate < 1:  # Less than 1 error per minute
            self.adaptive_factor = min(1.0, self.adaptive_factor * 1.05)
            logger.info(
                f"Low error rate, increasing limits by factor {self.adaptive_factor}"
            )

    async def acquire(self, weight: int = 1, retries: int = 3) -> bool:
        """Acquire with adaptive rate limiting"""
        # Apply adaptive factor to weight
        adjusted_weight = int(weight / self.adaptive_factor)
        return await super().acquire(adjusted_weight, retries)


# Utility functions


def create_conservative_rate_limiter() -> RateLimiter:
    """Create a conservative rate limiter for production use"""
    config = RateLimitConfig(
        requests_per_minute=600,  # Half of Binance limit
        weight_per_minute=3000,  # Half of weight limit
        order_limit_per_second=5,  # Half of order limit
        retry_delay_base=2.0,  # Longer base delay
        retry_delay_max=120.0,  # Longer max delay
    )
    return RateLimiter(config)


def create_aggressive_rate_limiter() -> RateLimiter:
    """Create an aggressive rate limiter for testing"""
    config = RateLimitConfig(
        requests_per_minute=1100,  # Close to Binance limit
        weight_per_minute=5500,  # Close to weight limit
        order_limit_per_second=9,  # Close to order limit
        retry_delay_base=0.5,  # Shorter delays
        retry_delay_max=30.0,
    )
    return RateLimiter(config)
