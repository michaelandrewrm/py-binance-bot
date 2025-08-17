"""
BO Suggester - Bayesian Optimization over simulator

This module implements Bayesian Optimization for hyperparameter tuning
of trading strategies using the simulation engine.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from decimal import Decimal
from datetime import datetime, timezone
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class HyperParameter:
    """Definition of a hyperparameter to optimize"""

    name: str
    param_type: str  # 'float', 'int', 'categorical'
    bounds: Optional[Tuple] = None  # (min, max) for numeric
    choices: Optional[List] = None  # for categorical
    log_scale: bool = False


@dataclass
class OptimizationResult:
    """Result of a single optimization trial"""

    trial_id: int
    parameters: Dict[str, Any]
    objective_value: float
    metrics: Dict[str, float]
    timestamp: datetime


class BayesianOptimizer:
    """
    Bayesian optimizer for trading strategy hyperparameters
    """

    def __init__(
        self,
        hyperparameters: List[HyperParameter],
        objective_function: Callable,
        n_initial_points: int = 10,
    ):
        self.hyperparameters = hyperparameters
        self.objective_function = objective_function
        self.n_initial_points = n_initial_points
        self.trials: List[OptimizationResult] = []
        self.best_result: Optional[OptimizationResult] = None

    async def optimize(
        self, n_trials: int = 50, progress_callback: Optional[Callable] = None
    ) -> OptimizationResult:
        """
        Run Bayesian optimization

        Args:
            n_trials: Number of optimization trials
            progress_callback: Optional callback for progress updates

        Returns:
            Best optimization result
        """
        logger.info(f"Starting Bayesian optimization with {n_trials} trials")

        for trial_id in range(n_trials):
            # Generate next parameters to try
            if trial_id < self.n_initial_points:
                # Random sampling for initial points
                parameters = self._random_sample()
            else:
                # Bayesian optimization
                parameters = self._bayesian_sample()

            # Evaluate objective function
            try:
                objective_value, metrics = await self.objective_function(parameters)

                result = OptimizationResult(
                    trial_id=trial_id,
                    parameters=parameters,
                    objective_value=objective_value,
                    metrics=metrics,
                    timestamp=datetime.now(timezone.utc),
                )

                self.trials.append(result)

                # Update best result
                if (
                    self.best_result is None
                    or objective_value > self.best_result.objective_value
                ):
                    self.best_result = result
                    logger.info(
                        f"New best result: {objective_value:.4f} at trial {trial_id}"
                    )

                # Progress callback
                if progress_callback:
                    await progress_callback(trial_id, n_trials, result)

            except Exception as e:
                logger.error(f"Error in trial {trial_id}: {e}")
                continue

        logger.info(
            f"Optimization completed. Best score: {self.best_result.objective_value:.4f}"
        )
        return self.best_result

    def _random_sample(self) -> Dict[str, Any]:
        """Generate random parameter sample"""
        parameters = {}

        for hp in self.hyperparameters:
            if hp.param_type == "float":
                if hp.log_scale:
                    value = np.exp(
                        np.random.uniform(np.log(hp.bounds[0]), np.log(hp.bounds[1]))
                    )
                else:
                    value = np.random.uniform(hp.bounds[0], hp.bounds[1])
                parameters[hp.name] = float(value)

            elif hp.param_type == "int":
                value = np.random.randint(hp.bounds[0], hp.bounds[1] + 1)
                parameters[hp.name] = int(value)

            elif hp.param_type == "categorical":
                value = np.random.choice(hp.choices)
                parameters[hp.name] = value

        return parameters

    def _bayesian_sample(self) -> Dict[str, Any]:
        """Generate parameter sample using Bayesian optimization"""
        # This is a simplified implementation
        # In practice, you would use a library like scikit-optimize or optuna

        # For now, use random sampling with some bias towards good regions
        if len(self.trials) < 3:
            return self._random_sample()

        # Find best trials
        sorted_trials = sorted(
            self.trials, key=lambda x: x.objective_value, reverse=True
        )
        top_trials = sorted_trials[:3]

        # Sample around best parameters with some noise
        base_params = top_trials[np.random.randint(0, len(top_trials))].parameters
        parameters = {}

        for hp in self.hyperparameters:
            base_value = base_params[hp.name]

            if hp.param_type == "float":
                # Add Gaussian noise
                noise_std = (hp.bounds[1] - hp.bounds[0]) * 0.1
                value = np.random.normal(base_value, noise_std)
                value = np.clip(value, hp.bounds[0], hp.bounds[1])
                parameters[hp.name] = float(value)

            elif hp.param_type == "int":
                # Add integer noise
                noise = np.random.randint(-2, 3)
                value = int(base_value + noise)
                value = np.clip(value, hp.bounds[0], hp.bounds[1])
                parameters[hp.name] = value

            elif hp.param_type == "categorical":
                # Sometimes use base value, sometimes random
                if np.random.random() < 0.7:
                    parameters[hp.name] = base_value
                else:
                    parameters[hp.name] = np.random.choice(hp.choices)

        return parameters

    def get_optimization_summary(self) -> Dict:
        """Get summary of optimization results"""
        if not self.trials:
            return {}

        scores = [trial.objective_value for trial in self.trials]

        return {
            "total_trials": len(self.trials),
            "best_score": max(scores),
            "worst_score": min(scores),
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "best_parameters": (
                self.best_result.parameters if self.best_result else None
            ),
            "improvement_over_trials": scores,
        }


class StrategyOptimizer:
    """
    High-level strategy optimizer that combines BO with simulation
    """

    def __init__(
        self,
        strategy_class,
        market_data,
        initial_capital: Decimal = Decimal("10000"),
    ):
        self.strategy_class = strategy_class
        self.market_data = market_data
        self.initial_capital = initial_capital

    async def optimize_strategy(
        self, hyperparameters: List[HyperParameter], n_trials: int = 50
    ) -> Dict:
        """
        Optimize a trading strategy using Bayesian optimization

        Args:
            hyperparameters: List of hyperparameters to optimize
            n_trials: Number of optimization trials

        Returns:
            Optimization results
        """

        async def objective_function(parameters: Dict) -> Tuple[float, Dict]:
            """Objective function for optimization"""
            try:
                # Create strategy with parameters
                strategy = self.strategy_class(**parameters)

                # Run backtest (simplified)
                # In practice, this would use the full simulation engine
                signals = strategy.generate_signals(self.market_data)

                # Calculate simple performance metric
                total_pnl = sum(
                    signal.get("pnl", Decimal("0"))
                    for signal in signals
                    if signal.get("pnl") is not None
                )

                # Calculate metrics
                total_trades = sum(
                    1 for signal in signals if signal.get("pnl") is not None
                )
                win_rate = sum(
                    1 for signal in signals if signal.get("pnl", 0) > 0
                ) / max(total_trades, 1)

                # Objective: maximize risk-adjusted return
                if total_trades == 0:
                    objective_value = 0.0
                else:
                    # Simple risk-adjusted return
                    avg_return = float(total_pnl) / float(self.initial_capital)
                    risk_penalty = 1.0 / max(total_trades, 1)  # Penalize too few trades
                    objective_value = avg_return * win_rate - risk_penalty

                metrics = {
                    "total_pnl": float(total_pnl),
                    "total_trades": total_trades,
                    "win_rate": win_rate,
                    "avg_return": avg_return if total_trades > 0 else 0.0,
                }

                return objective_value, metrics

            except Exception as e:
                logger.error(f"Error in objective function: {e}")
                return -1.0, {"error": str(e)}

        # Run optimization
        optimizer = BayesianOptimizer(hyperparameters, objective_function)
        best_result = await optimizer.optimize(n_trials)

        return {
            "best_result": best_result,
            "optimization_summary": optimizer.get_optimization_summary(),
            "all_trials": optimizer.trials,
        }


# Utility functions for common optimization scenarios


def create_grid_strategy_hyperparameters() -> List[HyperParameter]:
    """Create hyperparameters for grid strategy optimization"""
    return [
        HyperParameter("grid_spacing", "float", bounds=(0.005, 0.05)),
        HyperParameter("num_levels", "int", bounds=(3, 10)),
        HyperParameter("order_size_pct", "float", bounds=(0.02, 0.1)),
    ]


def create_volatility_strategy_hyperparameters() -> List[HyperParameter]:
    """Create hyperparameters for volatility strategy optimization"""
    return [
        HyperParameter("atr_period", "int", bounds=(10, 30)),
        HyperParameter("atr_multiplier", "float", bounds=(1.0, 4.0)),
        HyperParameter("sma_period", "int", bounds=(10, 50)),
        HyperParameter("min_volatility", "float", bounds=(0.005, 0.02)),
    ]


async def optimize_grid_strategy(market_data, n_trials: int = 30) -> Dict:
    """Quick optimization of grid strategy"""
    from ..ai.baseline import VolatilityBandStrategy

    hyperparameters = create_volatility_strategy_hyperparameters()
    optimizer = StrategyOptimizer(VolatilityBandStrategy, market_data)

    return await optimizer.optimize_strategy(hyperparameters, n_trials)


class MultiObjectiveOptimizer:
    """
    Multi-objective optimizer for trading strategies
    """

    def __init__(self, objectives: List[str], weights: List[float]):
        """
        Args:
            objectives: List of objective names (e.g., ['return', 'sharpe', 'max_drawdown'])
            weights: Weights for each objective
        """
        self.objectives = objectives
        self.weights = weights

        if len(objectives) != len(weights):
            raise ValueError("Number of objectives must match number of weights")

    def calculate_composite_score(self, metrics: Dict) -> float:
        """Calculate weighted composite score"""
        score = 0.0

        for obj, weight in zip(self.objectives, self.weights):
            if obj in metrics:
                # Normalize different metrics appropriately
                if obj in ["return", "sharpe_ratio"]:
                    score += weight * metrics[obj]
                elif obj in ["max_drawdown", "volatility"]:
                    score -= weight * metrics[obj]  # Minimize these
                elif obj == "win_rate":
                    score += weight * metrics[obj]

        return score


# Example usage functions


async def run_strategy_optimization_example():
    """Example of running strategy optimization"""
    # This would be called with real market data
    logger.info("Strategy optimization example - would need real market data to run")

    # Example hyperparameters
    hyperparams = [
        HyperParameter("param1", "float", bounds=(0.1, 1.0)),
        HyperParameter("param2", "int", bounds=(5, 20)),
        HyperParameter(
            "param3", "categorical", choices=["option1", "option2", "option3"]
        ),
    ]

    # Example objective function
    async def example_objective(params):
        # Simulate strategy evaluation
        score = np.random.random() * params["param1"] + params["param2"] * 0.01
        metrics = {"return": score, "trades": params["param2"]}
        return score, metrics

    optimizer = BayesianOptimizer(hyperparams, example_objective)
    result = await optimizer.optimize(n_trials=20)

    return result
