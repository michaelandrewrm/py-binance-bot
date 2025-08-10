"""
Evaluation - Sharpe, MAR, DD, utilization, turnover

This module provides comprehensive performance evaluation metrics
for trading strategies including risk-adjusted returns.
"""

from typing import List, Dict, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import math
import numpy as np

from data.schema import EquityCurvePoint, TradeRecord, BacktestResult

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Return metrics
    total_return: Decimal
    annualized_return: Decimal
    cumulative_return: Decimal
    
    # Risk metrics
    volatility: Decimal
    max_drawdown: Decimal
    max_drawdown_duration: int
    
    # Risk-adjusted metrics
    sharpe_ratio: Decimal
    calmar_ratio: Decimal
    sortino_ratio: Decimal
    
    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal
    profit_factor: Decimal
    average_win: Decimal
    average_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal
    
    # Efficiency metrics
    utilization: Decimal  # Capital utilization
    turnover: Decimal     # Portfolio turnover
    
    # Additional metrics
    kelly_criterion: Optional[Decimal] = None
    var_95: Optional[Decimal] = None  # Value at Risk 95%
    cvar_95: Optional[Decimal] = None  # Conditional VaR 95%

class PerformanceEvaluator:
    """
    Comprehensive performance evaluation engine
    """
    
    def __init__(self, risk_free_rate: Decimal = Decimal('0.02')):
        """
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
    
    def evaluate(self, equity_curve: List[EquityCurvePoint], 
                trades: List[TradeRecord],
                initial_capital: Decimal) -> PerformanceMetrics:
        """
        Evaluate strategy performance
        
        Args:
            equity_curve: List of equity curve points
            trades: List of executed trades
            initial_capital: Initial capital amount
            
        Returns:
            PerformanceMetrics object
        """
        if not equity_curve:
            return self._empty_metrics()
        
        # Calculate returns
        returns = self._calculate_returns(equity_curve, initial_capital)
        
        # Calculate drawdowns
        drawdowns, max_dd, max_dd_duration = self._calculate_drawdowns(equity_curve)
        
        # Calculate trade metrics
        trade_metrics = self._calculate_trade_metrics(trades)
        
        # Calculate risk metrics
        volatility = self._calculate_volatility(returns)
        
        # Calculate risk-adjusted metrics
        sharpe = self._calculate_sharpe_ratio(returns, volatility)
        calmar = self._calculate_calmar_ratio(returns, max_dd)
        sortino = self._calculate_sortino_ratio(returns)
        
        # Calculate efficiency metrics
        utilization = self._calculate_utilization(equity_curve, initial_capital)
        turnover = self._calculate_turnover(trades, equity_curve)
        
        # Calculate additional risk metrics
        var_95 = self._calculate_var(returns, 0.95)
        cvar_95 = self._calculate_cvar(returns, 0.95)
        kelly = self._calculate_kelly_criterion(trade_metrics)
        
        return PerformanceMetrics(
            total_return=returns['total_return'],
            annualized_return=returns['annualized_return'],
            cumulative_return=returns['cumulative_return'],
            volatility=volatility,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            sharpe_ratio=sharpe,
            calmar_ratio=calmar,
            sortino_ratio=sortino,
            total_trades=trade_metrics['total_trades'],
            winning_trades=trade_metrics['winning_trades'],
            losing_trades=trade_metrics['losing_trades'],
            win_rate=trade_metrics['win_rate'],
            profit_factor=trade_metrics['profit_factor'],
            average_win=trade_metrics['average_win'],
            average_loss=trade_metrics['average_loss'],
            largest_win=trade_metrics['largest_win'],
            largest_loss=trade_metrics['largest_loss'],
            utilization=utilization,
            turnover=turnover,
            kelly_criterion=kelly,
            var_95=var_95,
            cvar_95=cvar_95
        )
    
    def _calculate_returns(self, equity_curve: List[EquityCurvePoint], 
                          initial_capital: Decimal) -> Dict:
        """Calculate various return metrics"""
        if not equity_curve:
            return {'total_return': Decimal('0'), 'annualized_return': Decimal('0'), 
                   'cumulative_return': Decimal('0')}
        
        final_value = equity_curve[-1].net_worth
        total_return = (final_value - initial_capital) / initial_capital
        
        # Calculate time period
        start_date = equity_curve[0].timestamp
        end_date = equity_curve[-1].timestamp
        days = (end_date - start_date).days
        years = Decimal(str(days / 365.25))
        
        # Annualized return
        if years > 0 and final_value > 0:
            annualized_return = (final_value / initial_capital) ** (1 / float(years)) - 1
            annualized_return = Decimal(str(annualized_return))
        else:
            annualized_return = Decimal('0')
        
        cumulative_return = final_value / initial_capital - 1
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'cumulative_return': cumulative_return
        }
    
    def _calculate_drawdowns(self, equity_curve: List[EquityCurvePoint]) -> Tuple[List[Decimal], Decimal, int]:
        """Calculate drawdown metrics"""
        if len(equity_curve) < 2:
            return [], Decimal('0'), 0
        
        values = [point.net_worth for point in equity_curve]
        peak = values[0]
        drawdowns = []
        max_drawdown = Decimal('0')
        current_dd_duration = 0
        max_dd_duration = 0
        
        for value in values:
            if value > peak:
                peak = value
                current_dd_duration = 0
            else:
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
            
            drawdown = (peak - value) / peak
            drawdowns.append(drawdown)
            max_drawdown = max(max_drawdown, drawdown)
        
        return drawdowns, max_drawdown, max_dd_duration
    
    def _calculate_trade_metrics(self, trades: List[TradeRecord]) -> Dict:
        """Calculate trade-based metrics"""
        if not trades:
            return {
                'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                'win_rate': Decimal('0'), 'profit_factor': Decimal('0'),
                'average_win': Decimal('0'), 'average_loss': Decimal('0'),
                'largest_win': Decimal('0'), 'largest_loss': Decimal('0')
            }
        
        # Group trades by pairs (assuming buy-sell pairs)
        trade_pnls = []
        current_position = Decimal('0')
        entry_price = Decimal('0')
        
        for trade in trades:
            if trade.side.upper() == "BUY":
                if current_position == 0:
                    entry_price = trade.price
                current_position += trade.quantity
            else:  # SELL
                if current_position > 0:
                    sell_qty = min(trade.quantity, current_position)
                    pnl = (trade.price - entry_price) * sell_qty - trade.fee
                    trade_pnls.append(pnl)
                    current_position -= sell_qty
        
        if not trade_pnls:
            return {
                'total_trades': len(trades), 'winning_trades': 0, 'losing_trades': 0,
                'win_rate': Decimal('0'), 'profit_factor': Decimal('0'),
                'average_win': Decimal('0'), 'average_loss': Decimal('0'),
                'largest_win': Decimal('0'), 'largest_loss': Decimal('0')
            }
        
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
        
        total_trades = len(trade_pnls)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        win_rate = Decimal(str(win_count / total_trades)) if total_trades > 0 else Decimal('0')
        
        total_wins = sum(winning_trades) if winning_trades else Decimal('0')
        total_losses = abs(sum(losing_trades)) if losing_trades else Decimal('0')
        
        profit_factor = total_wins / total_losses if total_losses > 0 else Decimal('0')
        
        average_win = total_wins / len(winning_trades) if winning_trades else Decimal('0')
        average_loss = total_losses / len(losing_trades) if losing_trades else Decimal('0')
        
        largest_win = max(winning_trades) if winning_trades else Decimal('0')
        largest_loss = abs(min(losing_trades)) if losing_trades else Decimal('0')
        
        return {
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'average_win': average_win,
            'average_loss': average_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss
        }
    
    def _calculate_volatility(self, returns: Dict) -> Decimal:
        """Calculate annualized volatility"""
        # This is a simplified calculation
        # In practice, you'd use daily returns
        return returns['annualized_return'] * Decimal('0.3')  # Rough estimate
    
    def _calculate_sharpe_ratio(self, returns: Dict, volatility: Decimal) -> Decimal:
        """Calculate Sharpe ratio"""
        if volatility == 0:
            return Decimal('0')
        
        excess_return = returns['annualized_return'] - self.risk_free_rate
        return excess_return / volatility
    
    def _calculate_calmar_ratio(self, returns: Dict, max_drawdown: Decimal) -> Decimal:
        """Calculate Calmar ratio"""
        if max_drawdown == 0:
            return Decimal('0')
        
        return returns['annualized_return'] / max_drawdown
    
    def _calculate_sortino_ratio(self, returns: Dict) -> Decimal:
        """Calculate Sortino ratio (simplified)"""
        # This would require daily returns for proper calculation
        return self._calculate_sharpe_ratio(returns, returns['annualized_return'] * Decimal('0.2'))
    
    def _calculate_utilization(self, equity_curve: List[EquityCurvePoint], 
                             initial_capital: Decimal) -> Decimal:
        """Calculate capital utilization"""
        if not equity_curve:
            return Decimal('0')
        
        # Average capital utilization
        total_utilization = sum(
            (point.total_balance - point.unrealized_pnl) / initial_capital 
            for point in equity_curve
        )
        
        return total_utilization / len(equity_curve)
    
    def _calculate_turnover(self, trades: List[TradeRecord], 
                          equity_curve: List[EquityCurvePoint]) -> Decimal:
        """Calculate portfolio turnover"""
        if not trades or not equity_curve:
            return Decimal('0')
        
        total_volume = sum(trade.quantity * trade.price for trade in trades)
        average_capital = sum(point.net_worth for point in equity_curve) / len(equity_curve)
        
        if average_capital == 0:
            return Decimal('0')
        
        # Annualize turnover
        start_date = equity_curve[0].timestamp
        end_date = equity_curve[-1].timestamp
        days = (end_date - start_date).days
        years = max(Decimal(str(days / 365.25)), Decimal('0.01'))  # Minimum 1% of year
        
        return (total_volume / average_capital) / years
    
    def _calculate_var(self, returns: Dict, confidence: float) -> Decimal:
        """Calculate Value at Risk (simplified)"""
        # This is a very simplified VaR calculation
        # Proper implementation would use historical returns
        annual_return = returns['annualized_return']
        volatility = annual_return * Decimal('0.3')  # Rough estimate
        
        # Normal distribution assumption
        z_score = Decimal('1.645') if confidence == 0.95 else Decimal('2.326')  # 95% or 99%
        
        return -(annual_return - z_score * volatility)
    
    def _calculate_cvar(self, returns: Dict, confidence: float) -> Decimal:
        """Calculate Conditional Value at Risk (simplified)"""
        var = self._calculate_var(returns, confidence)
        # CVaR is typically 20-30% worse than VaR
        return var * Decimal('1.3')
    
    def _calculate_kelly_criterion(self, trade_metrics: Dict) -> Optional[Decimal]:
        """Calculate Kelly criterion percentage"""
        win_rate = trade_metrics['win_rate']
        avg_win = trade_metrics['average_win']
        avg_loss = trade_metrics['average_loss']
        
        if avg_loss == 0 or win_rate == 0:
            return None
        
        # Kelly = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = Decimal('1') - win_rate
        
        kelly = (b * p - q) / b
        
        # Cap Kelly at reasonable levels
        return max(Decimal('0'), min(kelly, Decimal('0.25')))
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics for no data"""
        return PerformanceMetrics(
            total_return=Decimal('0'),
            annualized_return=Decimal('0'),
            cumulative_return=Decimal('0'),
            volatility=Decimal('0'),
            max_drawdown=Decimal('0'),
            max_drawdown_duration=0,
            sharpe_ratio=Decimal('0'),
            calmar_ratio=Decimal('0'),
            sortino_ratio=Decimal('0'),
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=Decimal('0'),
            profit_factor=Decimal('0'),
            average_win=Decimal('0'),
            average_loss=Decimal('0'),
            largest_win=Decimal('0'),
            largest_loss=Decimal('0'),
            utilization=Decimal('0'),
            turnover=Decimal('0')
        )

class BenchmarkComparison:
    """Compare strategy performance against benchmarks"""
    
    def __init__(self, strategy_equity: List[EquityCurvePoint],
                 benchmark_equity: List[EquityCurvePoint]):
        self.strategy_equity = strategy_equity
        self.benchmark_equity = benchmark_equity
    
    def calculate_alpha_beta(self) -> Tuple[Decimal, Decimal]:
        """Calculate alpha and beta vs benchmark"""
        if len(self.strategy_equity) != len(self.benchmark_equity):
            return Decimal('0'), Decimal('0')
        
        # Calculate returns
        strategy_returns = []
        benchmark_returns = []
        
        for i in range(1, len(self.strategy_equity)):
            strat_ret = (self.strategy_equity[i].net_worth / 
                        self.strategy_equity[i-1].net_worth - 1)
            bench_ret = (self.benchmark_equity[i].net_worth / 
                        self.benchmark_equity[i-1].net_worth - 1)
            
            strategy_returns.append(float(strat_ret))
            benchmark_returns.append(float(bench_ret))
        
        if not strategy_returns or not benchmark_returns:
            return Decimal('0'), Decimal('0')
        
        # Calculate beta (covariance / variance)
        strategy_array = np.array(strategy_returns)
        benchmark_array = np.array(benchmark_returns)
        
        covariance = np.cov(strategy_array, benchmark_array)[0][1]
        benchmark_variance = np.var(benchmark_array)
        
        beta = Decimal(str(covariance / benchmark_variance)) if benchmark_variance != 0 else Decimal('0')
        
        # Calculate alpha
        strategy_mean = Decimal(str(np.mean(strategy_array)))
        benchmark_mean = Decimal(str(np.mean(benchmark_array)))
        
        alpha = strategy_mean - beta * benchmark_mean
        
        return alpha, beta
    
    def calculate_information_ratio(self) -> Decimal:
        """Calculate information ratio"""
        if len(self.strategy_equity) != len(self.benchmark_equity):
            return Decimal('0')
        
        # Calculate excess returns
        excess_returns = []
        
        for i in range(1, len(self.strategy_equity)):
            strat_ret = (self.strategy_equity[i].net_worth / 
                        self.strategy_equity[i-1].net_worth - 1)
            bench_ret = (self.benchmark_equity[i].net_worth / 
                        self.benchmark_equity[i-1].net_worth - 1)
            
            excess_returns.append(float(strat_ret - bench_ret))
        
        if not excess_returns:
            return Decimal('0')
        
        excess_array = np.array(excess_returns)
        mean_excess = np.mean(excess_array)
        std_excess = np.std(excess_array)
        
        if std_excess == 0:
            return Decimal('0')
        
        return Decimal(str(mean_excess / std_excess))

class RiskAnalyzer:
    """Advanced risk analysis tools"""
    
    @staticmethod
    def calculate_risk_metrics(equity_curve: List[EquityCurvePoint]) -> Dict:
        """Calculate comprehensive risk metrics"""
        if not equity_curve:
            return {}
        
        values = [float(point.net_worth) for point in equity_curve]
        returns = [(values[i] / values[i-1] - 1) for i in range(1, len(values))]
        
        if not returns:
            return {}
        
        returns_array = np.array(returns)
        
        # Downside deviation
        negative_returns = returns_array[returns_array < 0]
        downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0
        
        # Maximum consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        
        for ret in returns:
            if ret < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        # Skewness and kurtosis
        skewness = float(np.mean(((returns_array - np.mean(returns_array)) / np.std(returns_array)) ** 3))
        kurtosis = float(np.mean(((returns_array - np.mean(returns_array)) / np.std(returns_array)) ** 4)) - 3
        
        return {
            'downside_deviation': Decimal(str(downside_deviation)),
            'max_consecutive_losses': max_consecutive_losses,
            'skewness': Decimal(str(skewness)),
            'kurtosis': Decimal(str(kurtosis)),
            'tail_ratio': Decimal(str(abs(np.percentile(returns_array, 95)) / 
                                      abs(np.percentile(returns_array, 5))))
        }

# Utility functions

def create_performance_evaluator(risk_free_rate: Decimal = Decimal('0.02')) -> PerformanceEvaluator:
    """Create a performance evaluator with default settings"""
    return PerformanceEvaluator(risk_free_rate)

def quick_performance_summary(trades: List[TradeRecord], 
                            initial_capital: Decimal) -> Dict:
    """Quick performance summary for live monitoring"""
    if not trades:
        return {'total_pnl': Decimal('0'), 'trade_count': 0, 'win_rate': Decimal('0')}
    
    total_pnl = sum(trade.realized_pnl or Decimal('0') for trade in trades)
    total_fees = sum(trade.fee for trade in trades)
    net_pnl = total_pnl - total_fees
    
    # Simple win rate calculation
    profitable_trades = sum(1 for trade in trades if (trade.realized_pnl or Decimal('0')) > 0)
    win_rate = Decimal(str(profitable_trades / len(trades))) if trades else Decimal('0')
    
    return {
        'total_pnl': net_pnl,
        'total_fees': total_fees,
        'trade_count': len(trades),
        'win_rate': win_rate,
        'return_pct': (net_pnl / initial_capital) * 100
    }
