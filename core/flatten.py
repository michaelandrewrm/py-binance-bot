"""
Flatten Execution Module

This module handles the execution of flatten operations including:
- Canceling all open orders
- Market/TWAP closing positions with spread guard
- Setting manual resume flags
"""

from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import asyncio
import logging

from core.state import BotState, Order, OrderStatus, OrderSide, OrderType, utc_now
from core.sizing import FeeConfig
from exec.binance import BinanceClient

logger = logging.getLogger(__name__)


@dataclass
class FlattenConfig:
    """Configuration for flatten operations"""

    # TWAP execution parameters
    twap_duration_minutes: int = 5  # Execute over 5 minutes
    twap_slice_count: int = 10  # Break into 10 slices

    # Spread guard parameters
    max_spread_bps: int = 50  # Maximum 50 basis points spread
    price_timeout_seconds: int = 30  # Timeout for price quotes

    # Market order fallback
    market_order_threshold_pct: Decimal = Decimal(
        "0.95"
    )  # Use market if 95% of TWAP time elapsed

    # Retry parameters
    max_retries: int = 3
    retry_delay_seconds: int = 2


@dataclass
class FlattenResult:
    """Result of flatten operation"""

    success: bool
    orders_canceled: int
    positions_closed: Dict[str, Decimal]  # symbol -> quantity closed
    total_proceeds: Decimal
    fees_paid: Decimal
    execution_time_seconds: float
    error_message: Optional[str] = None


class PositionFlattener:
    """Handles position flattening with TWAP and spread guards"""

    def __init__(self, client: BinanceClient, config: FlattenConfig):
        self.client = client
        self.config = config

    async def flatten_position(self, symbol: str, quantity: Decimal) -> FlattenResult:
        """
        Flatten a position using TWAP with spread guard

        Args:
            symbol: Trading symbol (e.g., "BTCUSDC")
            quantity: Position quantity to close (positive for sell, negative for buy)

        Returns:
            FlattenResult with execution details
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Determine order side
            side = OrderSide.SELL if quantity > 0 else OrderSide.BUY
            abs_quantity = abs(quantity)

            # Calculate TWAP parameters
            slice_size = abs_quantity / self.config.twap_slice_count
            slice_interval = (
                self.config.twap_duration_minutes * 60 / self.config.twap_slice_count
            )

            total_proceeds = Decimal("0")
            total_fees = Decimal("0")
            executed_quantity = Decimal("0")

            logger.info(
                f"Starting TWAP flatten for {symbol}: {abs_quantity} {side.value} over {self.config.twap_duration_minutes}m"
            )

            # Execute TWAP slices
            for slice_num in range(self.config.twap_slice_count):
                if executed_quantity >= abs_quantity:
                    break

                # Calculate remaining quantity
                remaining = abs_quantity - executed_quantity
                current_slice = min(slice_size, remaining)

                # Check if we should switch to market order (near timeout)
                elapsed_pct = slice_num / self.config.twap_slice_count
                if elapsed_pct >= self.config.market_order_threshold_pct:
                    logger.info(
                        f"Switching to market order for remaining {remaining} (timeout threshold reached)"
                    )
                    proceeds, fees = await self._execute_market_order(
                        symbol, side, remaining
                    )
                    total_proceeds += proceeds
                    total_fees += fees
                    executed_quantity += remaining
                    break

                # Execute limit order with spread guard
                proceeds, fees, executed = await self._execute_limit_order_with_guard(
                    symbol, side, current_slice
                )

                total_proceeds += proceeds
                total_fees += fees
                executed_quantity += executed

                # Wait for next slice (unless this is the last one)
                if (
                    slice_num < self.config.twap_slice_count - 1
                    and executed_quantity < abs_quantity
                ):
                    await asyncio.sleep(slice_interval)

            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            return FlattenResult(
                success=True,
                orders_canceled=0,  # Set by caller
                positions_closed={symbol: executed_quantity},
                total_proceeds=total_proceeds,
                fees_paid=total_fees,
                execution_time_seconds=execution_time,
            )

        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.error(f"Error flattening position {symbol}: {e}")

            return FlattenResult(
                success=False,
                orders_canceled=0,
                positions_closed={},
                total_proceeds=Decimal("0"),
                fees_paid=Decimal("0"),
                execution_time_seconds=execution_time,
                error_message=str(e),
            )

    async def _execute_limit_order_with_guard(
        self, symbol: str, side: OrderSide, quantity: Decimal
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """Execute limit order with spread guard"""

        # Get current market data
        ticker = await self.client.get_ticker_price(symbol)
        current_price = Decimal(str(ticker["price"]))

        # Get order book for spread check
        order_book = await self.client.get_order_book(symbol, limit=5)

        best_bid = Decimal(str(order_book["bids"][0][0]))
        best_ask = Decimal(str(order_book["asks"][0][0]))

        # Calculate spread
        spread = (best_ask - best_bid) / current_price
        spread_bps = spread * 10000  # Convert to basis points

        if spread_bps > self.config.max_spread_bps:
            logger.warning(
                f"Spread too wide ({spread_bps:.1f}bps > {self.config.max_spread_bps}bps), using market order"
            )
            return await self._execute_market_order(symbol, side, quantity)

        # Place aggressive limit order
        if side == OrderSide.SELL:
            # Sell at bid price for immediate execution
            limit_price = best_bid
        else:
            # Buy at ask price for immediate execution
            limit_price = best_ask

        # Place order
        order_result = await self.client.place_order(
            symbol=symbol,
            side=side.value,
            order_type=OrderType.LIMIT.value,
            quantity=quantity,
            price=limit_price,
            time_in_force="IOC",  # Immediate or Cancel
        )

        # Check execution
        filled_qty = Decimal(str(order_result.get("executedQty", "0")))

        if filled_qty > 0:
            avg_price = Decimal(
                str(order_result.get("fills", [{}])[0].get("price", "0"))
            )
            proceeds = filled_qty * avg_price
            fees = sum(
                Decimal(str(fill.get("commission", "0")))
                for fill in order_result.get("fills", [])
            )

            logger.info(f"Limit order executed: {filled_qty} @ {avg_price}")
            return proceeds, fees, filled_qty
        else:
            # No fill, fall back to market order
            logger.info("Limit order not filled, falling back to market order")
            return await self._execute_market_order(symbol, side, quantity)

    async def _execute_market_order(
        self, symbol: str, side: OrderSide, quantity: Decimal
    ) -> Tuple[Decimal, Decimal]:
        """Execute market order"""

        order_result = await self.client.place_order(
            symbol=symbol,
            side=side.value,
            order_type=OrderType.MARKET.value,
            quantity=quantity,
        )

        filled_qty = Decimal(str(order_result.get("executedQty", "0")))

        if filled_qty > 0:
            avg_price = Decimal(
                str(order_result.get("fills", [{}])[0].get("price", "0"))
            )
            proceeds = filled_qty * avg_price
            fees = sum(
                Decimal(str(fill.get("commission", "0")))
                for fill in order_result.get("fills", [])
            )

            logger.info(f"Market order executed: {filled_qty} @ {avg_price}")
            return proceeds, fees
        else:
            raise Exception(f"Market order failed to execute for {symbol}")


class FlattenExecutor:
    """Main executor for flatten operations"""

    def __init__(self, client: BinanceClient, config: FlattenConfig = None):
        self.client = client
        self.config = config or FlattenConfig()
        self.flattener = PositionFlattener(client, self.config)

    async def execute_flatten(self, state: BotState, reason: str) -> FlattenResult:
        """
        Execute complete flatten operation:
        1. Cancel all open orders
        2. Close all positions with TWAP/spread guard
        3. Set manual resume flag

        Args:
            state: Current bot state
            reason: Reason for flattening

        Returns:
            FlattenResult with execution details
        """
        start_time = datetime.now(timezone.utc)
        logger.critical(f"FLATTEN OPERATION INITIATED: {reason}")

        try:
            # Step 1: Cancel all open orders
            orders_canceled = await self._cancel_all_orders(state)
            logger.info(f"Canceled {orders_canceled} open orders")

            # Step 2: Close all positions
            total_proceeds = Decimal("0")
            total_fees = Decimal("0")
            positions_closed = {}

            for symbol, position in state.positions.items():
                if not position.is_flat:
                    logger.info(f"Closing position: {symbol} {position.quantity}")

                    result = await self.flattener.flatten_position(
                        symbol, position.quantity
                    )

                    if result.success:
                        total_proceeds += result.total_proceeds
                        total_fees += result.fees_paid
                        positions_closed.update(result.positions_closed)
                    else:
                        logger.error(
                            f"Failed to close position {symbol}: {result.error_message}"
                        )
                        # Continue with other positions

            # Step 3: Set manual resume flag
            state.set_flatten_state(reason)

            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            logger.critical(f"FLATTEN OPERATION COMPLETED in {execution_time:.1f}s")
            logger.info(f"Orders canceled: {orders_canceled}")
            logger.info(f"Positions closed: {len(positions_closed)}")
            logger.info(f"Total proceeds: {total_proceeds}")
            logger.info(f"Total fees: {total_fees}")

            return FlattenResult(
                success=True,
                orders_canceled=orders_canceled,
                positions_closed=positions_closed,
                total_proceeds=total_proceeds,
                fees_paid=total_fees,
                execution_time_seconds=execution_time,
            )

        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            error_msg = f"Flatten operation failed: {e}"
            logger.critical(error_msg)

            # Still set manual resume flag even if flatten failed
            state.set_flatten_state(f"{reason} (FAILED: {e})")

            return FlattenResult(
                success=False,
                orders_canceled=0,
                positions_closed={},
                total_proceeds=Decimal("0"),
                fees_paid=Decimal("0"),
                execution_time_seconds=execution_time,
                error_message=error_msg,
            )

    async def _cancel_all_orders(self, state: BotState) -> int:
        """Cancel all active orders"""
        canceled_count = 0

        # Get list of active orders
        active_orders = list(state.active_orders.values())

        # Cancel each order
        for order in active_orders:
            try:
                await self.client.cancel_order(
                    symbol=order.symbol, order_id=int(order.order_id)
                )

                # Update order status in state
                state.update_order(order.order_id, status=OrderStatus.CANCELED)
                canceled_count += 1

                logger.info(f"Canceled order {order.order_id} for {order.symbol}")

            except Exception as e:
                logger.error(f"Failed to cancel order {order.order_id}: {e}")
                # Continue with other orders

        return canceled_count


# Convenience function for CLI/executor integration
async def execute_flatten_operation(
    client: BinanceClient, state: BotState, reason: str, config: FlattenConfig = None
) -> FlattenResult:
    """Execute flatten operation with default configuration"""
    executor = FlattenExecutor(client, config)
    return await executor.execute_flatten(state, reason)
