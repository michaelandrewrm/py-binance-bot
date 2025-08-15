"""
Validation - Walk-forward / rolling CV

This module implements validation techniques for trading strategies
including walk-forward analysis and rolling cross-validation.
"""

from typing import List, Dict, Optional, Tuple, Iterator
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import logging
import numpy as np

from data.schema import KlineData, BacktestResult, TradeRecord, EquityCurvePoint
from sim.evaluation import PerformanceEvaluator, PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class ValidationPeriod:
    """Single validation period definition"""

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    period_id: int


@dataclass
class ValidationResult:
    """Result of a single validation period"""

    period: ValidationPeriod
    train_metrics: PerformanceMetrics
    test_metrics: PerformanceMetrics
    parameters: Dict
    overfitting_score: float


class WalkForwardValidator:
    """
    Walk-forward validation for trading strategies
    """

    def __init__(
        self,
        train_window_days: int = 252,
        test_window_days: int = 63,
        step_days: int = 21,
        min_trades: int = 10,
    ):
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days
        self.min_trades = min_trades
        self.evaluator = PerformanceEvaluator()

    def generate_periods(
        self, start_date: datetime, end_date: datetime
    ) -> List[ValidationPeriod]:
        """Generate walk-forward validation periods"""
        periods = []
        period_id = 0

        current_start = start_date

        while True:
            train_start = current_start
            train_end = train_start + timedelta(days=self.train_window_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_window_days)

            if test_end > end_date:
                break

            period = ValidationPeriod(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                period_id=period_id,
            )
            periods.append(period)

            current_start += timedelta(days=self.step_days)
            period_id += 1

        logger.info(f"Generated {len(periods)} walk-forward periods")
        return periods

    async def validate_strategy(
        self, strategy_class, market_data: List[KlineData], optimization_function=None
    ) -> List[ValidationResult]:
        """
        Run walk-forward validation on a strategy

        Args:
            strategy_class: Strategy class to validate
            market_data: Complete market data
            optimization_function: Optional function to optimize parameters

        Returns:
            List of validation results
        """
        # Generate validation periods
        start_date = market_data[0].open_time
        end_date = market_data[-1].close_time
        periods = self.generate_periods(start_date, end_date)

        results = []

        for period in periods:
            logger.info(
                f"Validating period {period.period_id}: {period.test_start} to {period.test_end}"
            )

            # Split data
            train_data = [
                k
                for k in market_data
                if period.train_start <= k.open_time <= period.train_end
            ]
            test_data = [
                k
                for k in market_data
                if period.test_start <= k.open_time <= period.test_end
            ]

            if len(train_data) < 100 or len(test_data) < 20:
                logger.warning(f"Insufficient data for period {period.period_id}")
                continue

            try:
                # Optimize parameters on training data
                if optimization_function:
                    best_params = await optimization_function(train_data)
                else:
                    best_params = {}  # Use default parameters

                # Train on training data
                train_strategy = strategy_class(**best_params)
                train_signals = train_strategy.generate_signals(train_data)
                train_metrics = self._calculate_metrics(train_signals, train_data)

                # Test on out-of-sample data
                test_strategy = strategy_class(**best_params)
                test_signals = test_strategy.generate_signals(test_data)
                test_metrics = self._calculate_metrics(test_signals, test_data)

                # Calculate overfitting score
                overfitting_score = self._calculate_overfitting_score(
                    train_metrics, test_metrics
                )

                result = ValidationResult(
                    period=period,
                    train_metrics=train_metrics,
                    test_metrics=test_metrics,
                    parameters=best_params,
                    overfitting_score=overfitting_score,
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Error in validation period {period.period_id}: {e}")
                continue

        return results

    def _calculate_metrics(
        self, signals: List[Dict], klines: List[KlineData]
    ) -> PerformanceMetrics:
        """Calculate performance metrics from signals"""
        # Simplified metrics calculation
        trade_pnls = [
            signal.get("pnl", Decimal("0"))
            for signal in signals
            if signal.get("pnl") is not None
        ]

        if not trade_pnls:
            return self._empty_metrics()

        total_return = sum(trade_pnls)
        winning_trades = sum(1 for pnl in trade_pnls if pnl > 0)
        total_trades = len(trade_pnls)

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=total_return * 4,  # Simplified
            cumulative_return=total_return,
            volatility=Decimal("0.2"),  # Placeholder
            max_drawdown=Decimal("0.1"),  # Placeholder
            max_drawdown_duration=30,
            sharpe_ratio=(
                total_return / Decimal("0.2") if total_return != 0 else Decimal("0")
            ),
            calmar_ratio=(
                total_return / Decimal("0.1") if total_return != 0 else Decimal("0")
            ),
            sortino_ratio=Decimal("0"),  # Placeholder
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=total_trades - winning_trades,
            win_rate=(
                Decimal(winning_trades / total_trades)
                if total_trades > 0
                else Decimal("0")
            ),
            profit_factor=Decimal("1.5"),  # Placeholder
            average_win=Decimal("0.01"),  # Placeholder
            average_loss=Decimal("0.005"),  # Placeholder
            largest_win=max(trade_pnls) if trade_pnls else Decimal("0"),
            largest_loss=min(trade_pnls) if trade_pnls else Decimal("0"),
            utilization=Decimal("0.8"),  # Placeholder
            turnover=Decimal("2.0"),  # Placeholder
        )

    def _calculate_overfitting_score(
        self, train_metrics: PerformanceMetrics, test_metrics: PerformanceMetrics
    ) -> float:
        """Calculate overfitting score (lower is better)"""
        if train_metrics.total_return == 0:
            return 1.0

        # Performance degradation from train to test
        performance_ratio = float(
            test_metrics.total_return / train_metrics.total_return
        )

        # Overfitting score: 0 = no overfitting, 1 = complete overfitting
        overfitting_score = max(0.0, 1.0 - performance_ratio)

        return overfitting_score

    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics"""
        return PerformanceMetrics(
            total_return=Decimal("0"),
            annualized_return=Decimal("0"),
            cumulative_return=Decimal("0"),
            volatility=Decimal("0"),
            max_drawdown=Decimal("0"),
            max_drawdown_duration=0,
            sharpe_ratio=Decimal("0"),
            calmar_ratio=Decimal("0"),
            sortino_ratio=Decimal("0"),
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=Decimal("0"),
            profit_factor=Decimal("0"),
            average_win=Decimal("0"),
            average_loss=Decimal("0"),
            largest_win=Decimal("0"),
            largest_loss=Decimal("0"),
            utilization=Decimal("0"),
            turnover=Decimal("0"),
        )


class RollingWindowValidator:
    """
    Rolling window cross-validation for parameter stability
    """

    def __init__(self, window_days: int = 126, step_days: int = 21):
        self.window_days = window_days
        self.step_days = step_days

    def validate_parameters(
        self, strategy_class, market_data: List[KlineData], parameter_ranges: Dict
    ) -> Dict:
        """Validate parameter stability across rolling windows"""
        windows = self._generate_windows(market_data)

        parameter_results = {}

        for param_name, param_values in parameter_ranges.items():
            parameter_results[param_name] = {}

            for param_value in param_values:
                window_performances = []

                for window_data in windows:
                    if len(window_data) < 50:
                        continue

                    # Test parameter value on this window
                    strategy = strategy_class(**{param_name: param_value})
                    signals = strategy.generate_signals(window_data)

                    # Calculate performance
                    trade_pnls = [
                        signal.get("pnl", Decimal("0"))
                        for signal in signals
                        if signal.get("pnl") is not None
                    ]
                    performance = sum(trade_pnls) if trade_pnls else Decimal("0")
                    window_performances.append(float(performance))

                # Calculate stability metrics
                if window_performances:
                    parameter_results[param_name][param_value] = {
                        "mean_performance": np.mean(window_performances),
                        "std_performance": np.std(window_performances),
                        "min_performance": min(window_performances),
                        "max_performance": max(window_performances),
                        "stability_score": self._calculate_stability_score(
                            window_performances
                        ),
                    }

        return parameter_results

    def _generate_windows(self, market_data: List[KlineData]) -> List[List[KlineData]]:
        """Generate rolling windows"""
        windows = []
        start_idx = 0

        while start_idx < len(market_data):
            window_start = market_data[start_idx].open_time
            window_end = window_start + timedelta(days=self.window_days)

            window_data = [
                k for k in market_data[start_idx:] if k.open_time <= window_end
            ]

            if len(window_data) >= 50:  # Minimum window size
                windows.append(window_data)

            # Move to next window
            next_start_time = window_start + timedelta(days=self.step_days)
            start_idx = next(
                (
                    i
                    for i, k in enumerate(market_data)
                    if k.open_time >= next_start_time
                ),
                len(market_data),
            )

        return windows

    def _calculate_stability_score(self, performances: List[float]) -> float:
        """Calculate parameter stability score"""
        if len(performances) < 2:
            return 0.0

        # Lower coefficient of variation = higher stability
        mean_perf = np.mean(performances)
        std_perf = np.std(performances)

        if mean_perf == 0:
            return 0.0

        cv = std_perf / abs(mean_perf)
        stability_score = 1.0 / (1.0 + cv)  # Higher score = more stable

        return stability_score


class ValidationReporter:
    """
    Generate reports and analysis from validation results
    """

    def __init__(self):
        pass

    def generate_validation_report(self, results: List[ValidationResult]) -> Dict:
        """Generate comprehensive validation report"""
        if not results:
            return {"error": "No validation results provided"}

        # Aggregate metrics
        train_returns = [float(r.train_metrics.total_return) for r in results]
        test_returns = [float(r.test_metrics.total_return) for r in results]
        overfitting_scores = [r.overfitting_score for r in results]

        # Performance consistency
        test_win_rate = sum(1 for r in test_returns if r > 0) / len(test_returns)

        # Overfitting analysis
        avg_overfitting = np.mean(overfitting_scores)
        high_overfitting_periods = sum(1 for score in overfitting_scores if score > 0.5)

        return {
            "summary": {
                "total_periods": len(results),
                "avg_train_return": np.mean(train_returns),
                "avg_test_return": np.mean(test_returns),
                "test_return_std": np.std(test_returns),
                "test_win_rate": test_win_rate,
                "avg_overfitting_score": avg_overfitting,
                "high_overfitting_periods": high_overfitting_periods,
            },
            "period_details": [
                {
                    "period_id": r.period.period_id,
                    "test_start": r.period.test_start.isoformat(),
                    "test_end": r.period.test_end.isoformat(),
                    "train_return": float(r.train_metrics.total_return),
                    "test_return": float(r.test_metrics.total_return),
                    "overfitting_score": r.overfitting_score,
                    "parameters": r.parameters,
                }
                for r in results
            ],
            "recommendations": self._generate_recommendations(results),
        }

    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []

        test_returns = [float(r.test_metrics.total_return) for r in results]
        overfitting_scores = [r.overfitting_score for r in results]

        # Check consistency
        positive_periods = sum(1 for r in test_returns if r > 0)
        consistency_rate = positive_periods / len(test_returns)

        if consistency_rate < 0.6:
            recommendations.append(
                "Strategy shows low consistency across periods. Consider parameter optimization."
            )

        # Check overfitting
        avg_overfitting = np.mean(overfitting_scores)
        if avg_overfitting > 0.3:
            recommendations.append(
                "High overfitting detected. Consider regularization or simpler models."
            )

        # Check performance degradation
        train_returns = [float(r.train_metrics.total_return) for r in results]
        performance_ratio = np.mean(test_returns) / max(np.mean(train_returns), 0.001)

        if performance_ratio < 0.5:
            recommendations.append(
                "Significant performance degradation from train to test. Review strategy logic."
            )

        if not recommendations:
            recommendations.append("Strategy shows good validation characteristics.")

        return recommendations


# Utility functions


async def quick_walkforward_validation(
    strategy_class, market_data: List[KlineData]
) -> Dict:
    """Quick walk-forward validation with default parameters"""
    validator = WalkForwardValidator(
        train_window_days=180,  # 6 months
        test_window_days=30,  # 1 month
        step_days=15,  # 2 week steps
    )

    results = await validator.validate_strategy(strategy_class, market_data)

    reporter = ValidationReporter()
    return reporter.generate_validation_report(results)


def parameter_stability_analysis(
    strategy_class, market_data: List[KlineData], parameter_ranges: Dict
) -> Dict:
    """Analyze parameter stability across rolling windows"""
    validator = RollingWindowValidator()
    return validator.validate_parameters(strategy_class, market_data, parameter_ranges)
