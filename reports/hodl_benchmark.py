"""
HODL Benchmark Module

This module provides HODL (Hold On for Dear Life) benchmark calculations
for comparing trading strategy performance against a simple buy-and-hold approach.
"""

from typing import List, Dict, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timezone, timedelta
import logging

from data.schema import KlineData, TradeRecord
from core.sizing import FeeConfig

logger = logging.getLogger(__name__)


class HODLBenchmark:
    """
    Calculate HODL benchmark performance for comparison with trading strategies
    """

    def __init__(self, fee_config: Optional[FeeConfig] = None):
        # Use default Binance-like fees if not provided
        self.fee_config = fee_config or FeeConfig(
            maker_fee=Decimal("0.001"),  # 0.1%
            taker_fee=Decimal("0.001"),  # 0.1%
            use_bnb=False,
        )

    def calculate_hodl_performance(
        self,
        klines: List[KlineData],
        initial_capital: Decimal,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict:
        """
        Calculate HODL performance metrics

        Args:
            klines: Historical price data
            initial_capital: Starting capital
            start_date: Strategy start date (optional)
            end_date: Strategy end date (optional)

        Returns:
            Dictionary with HODL performance metrics
        """
        if not klines:
            return self._empty_hodl_metrics()

        # Filter klines to date range if specified
        filtered_klines = self._filter_klines_by_date(klines, start_date, end_date)

        if not filtered_klines:
            return self._empty_hodl_metrics()

        # Calculate HODL metrics with fees
        entry_price = filtered_klines[0].open_price
        exit_price = filtered_klines[-1].close_price

        # Calculate entry transaction with fees
        # When buying, we pay fees on the quote amount (USDC/USD)
        entry_fee_rate = self.fee_config.taker_fee  # Assume market order (taker)
        if self.fee_config.use_bnb:
            entry_fee_rate *= self.fee_config.bnb_discount

        # Available capital after entry fee
        entry_fee_amount = initial_capital * entry_fee_rate
        capital_after_entry_fee = initial_capital - entry_fee_amount

        # Calculate position size (how much crypto we could buy)
        position_size = capital_after_entry_fee / entry_price

        # Calculate exit transaction with fees
        # When selling, we pay fees on the quote amount received
        gross_sale_value = position_size * exit_price
        exit_fee_rate = self.fee_config.taker_fee  # Assume market order (taker)
        if self.fee_config.use_bnb:
            exit_fee_rate *= self.fee_config.bnb_discount

        exit_fee_amount = gross_sale_value * exit_fee_rate
        final_value = gross_sale_value - exit_fee_amount

        # Calculate returns
        total_return = (final_value - initial_capital) / initial_capital

        # Calculate time-based metrics
        start_time = filtered_klines[0].open_time
        end_time = filtered_klines[-1].close_time
        total_days = (end_time - start_time).days

        # Annualized return
        if total_days > 0:
            return_multiplier = final_value / initial_capital
            days_factor = Decimal(str(365 / total_days))
            annualized_return = (float(return_multiplier) ** float(days_factor)) - 1
            annualized_return = Decimal(str(annualized_return))
        else:
            annualized_return = Decimal("0")

        # Calculate volatility and drawdown
        price_series = [float(k.close_price) for k in filtered_klines]
        daily_returns = []
        for i in range(1, len(price_series)):
            daily_return = (price_series[i] - price_series[i - 1]) / price_series[i - 1]
            daily_returns.append(daily_return)

        # Volatility (annualized)
        if len(daily_returns) > 1:
            import math

            mean_return = sum(daily_returns) / len(daily_returns)
            variance = sum((r - mean_return) ** 2 for r in daily_returns) / (
                len(daily_returns) - 1
            )
            daily_vol = math.sqrt(variance)
            annualized_vol = daily_vol * math.sqrt(365)
        else:
            annualized_vol = 0

        # Maximum drawdown
        max_drawdown, max_drawdown_duration = self._calculate_hodl_drawdown(
            filtered_klines, position_size
        )

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = (
            float(annualized_return) / annualized_vol if annualized_vol > 0 else 0
        )

        # Calmar ratio
        calmar_ratio = (
            float(annualized_return) / abs(float(max_drawdown))
            if max_drawdown != 0
            else 0
        )

        return {
            "strategy_type": "HODL",
            "symbol": filtered_klines[0].symbol,
            "start_date": start_time,
            "end_date": end_time,
            "initial_capital": float(initial_capital),
            "final_capital": float(final_value),
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "position_size": float(position_size),
            "entry_fee_amount": float(entry_fee_amount),
            "exit_fee_amount": float(exit_fee_amount),
            "total_fees": float(entry_fee_amount + exit_fee_amount),
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "volatility": annualized_vol,
            "max_drawdown": float(max_drawdown),
            "max_drawdown_duration": max_drawdown_duration,
            "sharpe_ratio": sharpe_ratio,
            "calmar_ratio": calmar_ratio,
            "total_days": total_days,
            "fee_config": {
                "maker_fee": float(self.fee_config.maker_fee),
                "taker_fee": float(self.fee_config.taker_fee),
                "use_bnb": self.fee_config.use_bnb,
                "bnb_discount": (
                    float(self.fee_config.bnb_discount)
                    if self.fee_config.use_bnb
                    else None
                ),
            },
        }

    def compare_with_strategy(
        self,
        strategy_metrics: Dict,
        klines: List[KlineData],
        initial_capital: Decimal,
    ) -> Dict:
        """
        Compare strategy performance with HODL benchmark

        Args:
            strategy_metrics: Strategy performance metrics
            klines: Historical price data
            initial_capital: Initial capital used

        Returns:
            Comparison results
        """
        # Get strategy dates for fair comparison
        start_date = strategy_metrics.get("start_date")
        end_date = strategy_metrics.get("end_date")

        # Calculate HODL performance for same period
        hodl_metrics = self.calculate_hodl_performance(
            klines, initial_capital, start_date, end_date
        )

        # Calculate comparison metrics
        excess_return = (
            strategy_metrics.get("total_return", 0) - hodl_metrics["total_return"]
        )
        excess_annual_return = (
            strategy_metrics.get("annualized_return", 0)
            - hodl_metrics["annualized_return"]
        )

        # Risk-adjusted comparison
        strategy_sharpe = strategy_metrics.get("sharpe_ratio", 0)
        hodl_sharpe = hodl_metrics["sharpe_ratio"]
        sharpe_difference = strategy_sharpe - hodl_sharpe

        # Drawdown comparison
        strategy_dd = strategy_metrics.get("max_drawdown", 0)
        hodl_dd = hodl_metrics["max_drawdown"]
        drawdown_improvement = abs(hodl_dd) - abs(
            strategy_dd
        )  # Positive means strategy had lower drawdown

        # Win/loss analysis
        strategy_won = excess_return > 0

        return {
            "strategy_metrics": strategy_metrics,
            "hodl_metrics": hodl_metrics,
            "comparison": {
                "excess_return": excess_return,
                "excess_annual_return": excess_annual_return,
                "sharpe_difference": sharpe_difference,
                "drawdown_improvement": drawdown_improvement,
                "strategy_outperformed": strategy_won,
                "return_ratio": (
                    strategy_metrics.get("total_return", 0)
                    / hodl_metrics["total_return"]
                    if hodl_metrics["total_return"] != 0
                    else float("inf")
                ),
            },
        }

    def generate_benchmark_report(self, comparison_data: Dict) -> str:
        """
        Generate a text report comparing strategy vs HODL

        Args:
            comparison_data: Output from compare_with_strategy

        Returns:
            Formatted text report
        """
        strategy = comparison_data["strategy_metrics"]
        hodl = comparison_data["hodl_metrics"]
        comp = comparison_data["comparison"]

        report = []
        report.append("=" * 60)
        report.append("STRATEGY vs HODL BENCHMARK COMPARISON")
        report.append("=" * 60)
        report.append("")

        # Basic metrics comparison
        report.append("PERFORMANCE METRICS")
        report.append("-" * 30)
        report.append(f"Metric                  Strategy      HODL        Difference")
        report.append("-" * 60)
        report.append(
            f"Total Return           {strategy.get('total_return', 0):+8.2%}   {hodl['total_return']:+8.2%}   {comp['excess_return']:+8.2%}"
        )
        report.append(
            f"Annualized Return      {strategy.get('annualized_return', 0):+8.2%}   {hodl['annualized_return']:+8.2%}   {comp['excess_annual_return']:+8.2%}"
        )
        report.append(
            f"Sharpe Ratio           {strategy.get('sharpe_ratio', 0):8.3f}   {hodl['sharpe_ratio']:8.3f}   {comp['sharpe_difference']:+8.3f}"
        )
        report.append(
            f"Max Drawdown           {strategy.get('max_drawdown', 0):8.2%}   {hodl['max_drawdown']:8.2%}   {comp['drawdown_improvement']:+8.2%}"
        )
        report.append(
            f"Volatility             {strategy.get('volatility', 0):8.2%}   {hodl['volatility']:8.2%}   N/A"
        )
        report.append("")

        # Capital comparison
        report.append("CAPITAL COMPARISON")
        report.append("-" * 30)
        report.append(
            f"Initial Capital:        ${strategy.get('initial_capital', 0):,.2f}"
        )
        report.append(
            f"Strategy Final:         ${strategy.get('final_capital', 0):,.2f}"
        )
        report.append(f"HODL Final:             ${hodl['final_capital']:,.2f}")
        report.append(
            f"Difference:             ${strategy.get('final_capital', 0) - hodl['final_capital']:+,.2f}"
        )
        report.append("")

        # Fee analysis for HODL
        report.append("HODL FEE BREAKDOWN")
        report.append("-" * 30)
        report.append(
            f"Entry Fee:              ${hodl.get('entry_fee_amount', 0):,.2f}"
        )
        report.append(f"Exit Fee:               ${hodl.get('exit_fee_amount', 0):,.2f}")
        report.append(f"Total HODL Fees:        ${hodl.get('total_fees', 0):,.2f}")
        fee_impact = hodl.get("total_fees", 0) / hodl["initial_capital"] * 100
        report.append(f"Fee Impact on Return:   {fee_impact:.3f}%")
        report.append("")

        # Trading activity vs HODL
        report.append("TRADING ACTIVITY")
        report.append("-" * 30)
        report.append(f"Strategy Trades:        {strategy.get('total_trades', 0)}")
        report.append(f"HODL Trades:            2 (Buy and Hold)")
        report.append(f"Strategy Win Rate:      {strategy.get('win_rate', 0):.1%}")
        report.append(
            f"Trading Frequency:      {strategy.get('total_trades', 0) / max(hodl['total_days'], 1):.2f} trades/day"
        )
        report.append("")

        # Risk analysis
        report.append("RISK ANALYSIS")
        report.append("-" * 30)
        strategy_vol = strategy.get("volatility", 0)
        hodl_vol = hodl["volatility"]
        report.append(f"Strategy Volatility:    {strategy_vol:.2%}")
        report.append(f"HODL Volatility:        {hodl_vol:.2%}")

        if hodl_vol > 0:
            vol_ratio = strategy_vol / hodl_vol
            report.append(f"Volatility Ratio:       {vol_ratio:.2f}x")

        report.append("")

        # Conclusion
        report.append("CONCLUSION")
        report.append("-" * 30)
        if comp["strategy_outperformed"]:
            report.append("✅ STRATEGY OUTPERFORMED HODL")
            report.append(
                f"The strategy achieved {comp['excess_return']:+.2%} excess return"
            )
            if comp["sharpe_difference"] > 0:
                report.append(
                    f"with {comp['sharpe_difference']:+.3f} better risk-adjusted performance"
                )
        else:
            report.append("❌ STRATEGY UNDERPERFORMED HODL")
            report.append(
                f"The strategy underperformed by {comp['excess_return']:+.2%}"
            )
            report.append("Consider revising strategy parameters or approach")

        report.append("")
        report.append(f"Return Ratio: {comp['return_ratio']:.2f}x")

        if comp["drawdown_improvement"] > 0:
            report.append(
                f"✅ Strategy had {comp['drawdown_improvement']:.2%} lower maximum drawdown"
            )
        else:
            report.append(
                f"⚠️  Strategy had {abs(comp['drawdown_improvement']):.2%} higher maximum drawdown"
            )

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)

    def _filter_klines_by_date(
        self,
        klines: List[KlineData],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> List[KlineData]:
        """Filter klines by date range"""
        filtered = klines

        if start_date:
            filtered = [k for k in filtered if k.open_time >= start_date]

        if end_date:
            filtered = [k for k in filtered if k.open_time <= end_date]

        return filtered

    def _calculate_hodl_drawdown(
        self, klines: List[KlineData], position_size: Decimal
    ) -> Tuple[Decimal, int]:
        """Calculate maximum drawdown for HODL strategy (accounting for exit fees)"""
        if not klines:
            return Decimal("0"), 0

        # Calculate portfolio values accounting for potential exit fees
        portfolio_values = []
        exit_fee_rate = self.fee_config.taker_fee
        if self.fee_config.use_bnb:
            exit_fee_rate *= self.fee_config.bnb_discount

        for kline in klines:
            # Calculate value if we were to sell at this point
            gross_value = position_size * kline.close_price
            exit_fee = gross_value * exit_fee_rate
            net_value = gross_value - exit_fee
            portfolio_values.append(float(net_value))

        # Calculate drawdown
        peak = portfolio_values[0]
        max_drawdown = 0
        max_drawdown_duration = 0
        current_drawdown_duration = 0

        for value in portfolio_values:
            if value > peak:
                peak = value
                current_drawdown_duration = 0
            else:
                drawdown = (peak - value) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                current_drawdown_duration += 1
                max_drawdown_duration = max(
                    max_drawdown_duration, current_drawdown_duration
                )

        return Decimal(str(max_drawdown)), max_drawdown_duration

    def _empty_hodl_metrics(self) -> Dict:
        """Return empty HODL metrics"""
        return {
            "strategy_type": "HODL",
            "total_return": 0,
            "annualized_return": 0,
            "volatility": 0,
            "max_drawdown": 0,
            "max_drawdown_duration": 0,
            "sharpe_ratio": 0,
            "calmar_ratio": 0,
            "initial_capital": 0,
            "final_capital": 0,
            "total_days": 0,
        }


class BenchmarkSuite:
    """
    Suite of benchmark strategies for comprehensive comparison
    """

    def __init__(self, fee_config: Optional[FeeConfig] = None):
        self.hodl_benchmark = HODLBenchmark(fee_config)

    def run_all_benchmarks(
        self,
        klines: List[KlineData],
        initial_capital: Decimal,
        strategy_metrics: Dict,
    ) -> Dict:
        """Run all available benchmarks"""

        results = {}

        # HODL Benchmark
        hodl_comparison = self.hodl_benchmark.compare_with_strategy(
            strategy_metrics, klines, initial_capital
        )
        results["hodl"] = hodl_comparison

        # Simple Moving Average Benchmark (placeholder)
        results["sma_benchmark"] = self._calculate_sma_benchmark(
            klines, initial_capital
        )

        # Random Trading Benchmark (placeholder)
        results["random_benchmark"] = self._calculate_random_benchmark(
            klines, initial_capital
        )

        return results

    def _calculate_sma_benchmark(
        self, klines: List[KlineData], initial_capital: Decimal
    ) -> Dict:
        """Calculate simple moving average crossover benchmark"""
        # Placeholder implementation
        # In real implementation, would calculate SMA(20) vs SMA(50) crossover strategy
        return {
            "strategy_type": "SMA_Crossover",
            "total_return": 0.05,  # Placeholder
            "sharpe_ratio": 0.8,  # Placeholder
            "max_drawdown": -0.15,  # Placeholder
        }

    def _calculate_random_benchmark(
        self, klines: List[KlineData], initial_capital: Decimal
    ) -> Dict:
        """Calculate random trading benchmark"""
        # Placeholder implementation
        # In real implementation, would simulate random buy/sell decisions
        return {
            "strategy_type": "Random_Trading",
            "total_return": -0.02,  # Placeholder (typically negative due to fees)
            "sharpe_ratio": -0.1,  # Placeholder
            "max_drawdown": -0.25,  # Placeholder
        }

    def generate_benchmark_summary(self, benchmark_results: Dict) -> str:
        """Generate summary of all benchmark comparisons"""

        report = []
        report.append("=" * 70)
        report.append("COMPREHENSIVE BENCHMARK COMPARISON")
        report.append("=" * 70)
        report.append("")

        # Strategy performance vs all benchmarks
        report.append("BENCHMARK COMPARISON SUMMARY")
        report.append("-" * 40)
        report.append(
            f"{'Benchmark':<20} {'Strategy Return':<15} {'Benchmark Return':<15} {'Outperformed':<12}"
        )
        report.append("-" * 65)

        for benchmark_name, benchmark_data in benchmark_results.items():
            if benchmark_name == "hodl":
                strategy_return = benchmark_data["strategy_metrics"].get(
                    "total_return", 0
                )
                benchmark_return = benchmark_data["hodl_metrics"]["total_return"]
                outperformed = (
                    "✅ Yes" if strategy_return > benchmark_return else "❌ No"
                )
            else:
                # Placeholder for other benchmarks
                strategy_return = 0.1  # Placeholder
                benchmark_return = benchmark_data.get("total_return", 0)
                outperformed = (
                    "✅ Yes" if strategy_return > benchmark_return else "❌ No"
                )

            report.append(
                f"{benchmark_name.upper():<20} {strategy_return:<15.2%} {benchmark_return:<15.2%} {outperformed:<12}"
            )

        report.append("")
        report.append("=" * 70)

        return "\n".join(report)


# Convenience functions


def quick_hodl_comparison(
    strategy_metrics: Dict,
    klines: List[KlineData],
    initial_capital: Decimal,
    fee_config: Optional[FeeConfig] = None,
) -> str:
    """Quick HODL comparison and report generation"""
    benchmark = HODLBenchmark(fee_config)
    comparison = benchmark.compare_with_strategy(
        strategy_metrics, klines, initial_capital
    )
    return benchmark.generate_benchmark_report(comparison)


def calculate_hodl_baseline(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    initial_capital: Decimal = Decimal("10000"),
) -> Dict:
    """Calculate HODL baseline for a symbol and period"""
    # This would need to load actual price data
    # Placeholder implementation
    return {
        "symbol": symbol,
        "period": f"{start_date} to {end_date}",
        "hodl_return": 0.25,  # Placeholder
        "hodl_sharpe": 1.2,  # Placeholder
        "hodl_max_dd": -0.18,  # Placeholder
    }


if __name__ == "__main__":
    # Example usage
    logger.info("HODL Benchmark Module")
    logger.info(
        "Use this module to compare trading strategies against buy-and-hold performance"
    )
