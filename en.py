"""
Advanced Multi-Domain Simulation Engine
Quantitative Financial Engineering & Formula One Strategy Analytics

Mathematical Backend - Decoupled from UI
Fully vectorized using NumPy and Pandas for optimal performance
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


@dataclass
class HestonParams:
    """Heston Stochastic Volatility Model Parameters"""
    kappa: float      # Speed of mean reversion
    theta: float      # Long-term variance level
    sigma: float      # Volatility of volatility
    rho: float        # Correlation between asset and variance shocks (-0.7 typical)
    v0: float         # Initial variance


class QuantFinanceEngine:
    """Quantitative Financial Engineering Simulation Engine"""
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
        
    def heston_monte_carlo(
        self,
        S0: np.ndarray,
        T: float = 1.0,
        steps: int = 252,
        paths: int = 1000,
        params: Optional[HestonParams] = None,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate asset price paths using Heston Stochastic Volatility model.
        
        Args:
            S0: Initial asset prices (n_assets,)
            T: Time to maturity (years)
            steps: Number of time steps
            paths: Number of simulation paths
            params: HestonParams object with (kappa, theta, sigma, rho, v0)
            seed: Random seed for reproducibility
            
        Returns:
            price_paths: (steps, paths, n_assets)
            variance_paths: (steps, paths, n_assets)
        """
        if seed is not None:
            np.random.seed(seed)
            
        if params is None:
            params = HestonParams(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04)
        
        n_assets = len(S0) if isinstance(S0, np.ndarray) else 1
        S0 = np.atleast_1d(S0)
        dt = T / steps
        
        # Initialize arrays
        price_paths = np.zeros((steps, paths, n_assets))
        variance_paths = np.zeros((steps, paths, n_assets))
        
        price_paths[0] = S0
        variance_paths[0] = params.v0
        
        # Generate correlated Brownian motions
        # W_S: asset price shocks, W_v: variance shocks
        Z_S = np.random.standard_normal((steps - 1, paths, n_assets))
        Z_v = np.random.standard_normal((steps - 1, paths, n_assets))
        
        # Enforce correlation structure: W_v = rho * W_S + sqrt(1-rho^2) * Z_v
        W_v = params.rho * Z_S + np.sqrt(1 - params.rho**2) * Z_v
        
        # Discretize using Euler scheme
        sqrt_dt = np.sqrt(dt)
        
        for t in range(1, steps):
            # Current state
            S_prev = price_paths[t-1]
            v_prev = variance_paths[t-1]
            
            # Ensure variance stays non-negative (Feller condition)
            v_prev = np.maximum(v_prev, 1e-6)
            
            # Variance evolution: dv_t = kappa(theta - v_t)dt + sigma*sqrt(v_t)dW_v
            variance_paths[t] = v_prev + params.kappa * (params.theta - v_prev) * dt + \
                               params.sigma * np.sqrt(v_prev) * W_v[t-1] * sqrt_dt
            variance_paths[t] = np.maximum(variance_paths[t], 1e-6)
            
            # Asset price evolution: dS_t/S_t = r*dt + sqrt(v_t)dW_S
            price_paths[t] = S_prev * np.exp(
                (self.risk_free_rate - 0.5 * variance_paths[t]) * dt + 
                np.sqrt(variance_paths[t]) * Z_S[t-1] * sqrt_dt
            )
        
        return price_paths, variance_paths
    
    def compute_portfolio_metrics(
        self,
        returns: np.ndarray,
        weights: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute portfolio risk metrics from simulation paths.
        
        Args:
            returns: Log returns (n_time_steps, n_paths, n_assets)
            weights: Portfolio weights (n_assets,)
            
        Returns:
            Dictionary with VaR, ES, Kurtosis metrics
        """
        # Portfolio returns across all paths
        portfolio_returns = np.dot(returns, weights)  # (n_time_steps, n_paths)
        final_returns = portfolio_returns[-1]  # Final period returns
        
        # Value at Risk (95% VaR)
        var_95 = np.percentile(final_returns, 5)
        
        # Expected Shortfall / CVaR (average of returns below 5th percentile)
        tail_returns = final_returns[final_returns <= var_95]
        es_95 = np.mean(tail_returns) if len(tail_returns) > 0 else var_95
        
        # Excess Kurtosis (measure of fat tails)
        excess_kurtosis = self._compute_excess_kurtosis(final_returns)
        
        # Annualized metrics
        annual_return = np.mean(final_returns)
        annual_volatility = np.std(final_returns)
        
        sharpe_ratio = (annual_return - self.risk_free_rate) / (annual_volatility + 1e-6)
        
        return {
            'return': annual_return,
            'volatility': annual_volatility,
            'var_95': var_95,
            'es_95': es_95,
            'excess_kurtosis': excess_kurtosis,
            'sharpe_ratio': sharpe_ratio
        }
    
    def generate_random_weights(self, n_assets: int, n_portfolios: int = 1000) -> np.ndarray:
        """
        Generate random portfolio weight allocations.
        
        Args:
            n_assets: Number of assets
            n_portfolios: Number of random portfolios
            
        Returns:
            Weights array (n_portfolios, n_assets)
        """
        weights = np.random.dirichlet(np.ones(n_assets), n_portfolios)
        return weights
    
    def efficient_frontier(
        self,
        price_paths: np.ndarray,
        n_portfolios: int = 500
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute efficient frontier via Monte Carlo random weight sampling.
        
        Args:
            price_paths: (steps, paths, n_assets)
            n_portfolios: Number of portfolios to simulate
            
        Returns:
            volatilities, returns, sharpe_ratios, expected_shortfalls
        """
        n_assets = price_paths.shape[2]
        
        # Compute log returns from price paths
        log_returns = np.diff(np.log(price_paths), axis=0)  # (steps-1, paths, n_assets)
        
        # Generate random weights
        weights = self.generate_random_weights(n_assets, n_portfolios)
        
        volatilities = np.zeros(n_portfolios)
        returns = np.zeros(n_portfolios)
        sharpe_ratios = np.zeros(n_portfolios)
        expected_shortfalls = np.zeros(n_portfolios)
        
        for i in range(n_portfolios):
            metrics = self.compute_portfolio_metrics(log_returns, weights[i])
            returns[i] = metrics['return']
            volatilities[i] = metrics['volatility']
            sharpe_ratios[i] = metrics['sharpe_ratio']
            expected_shortfalls[i] = metrics['es_95']
        
        return volatilities, returns, sharpe_ratios, expected_shortfalls
    
    @staticmethod
    def _compute_excess_kurtosis(data: np.ndarray) -> float:
        """Compute excess kurtosis of data"""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            return 0.0
        
        m4 = np.mean((data - mean) ** 4)
        m2 = np.mean((data - mean) ** 2)
        kurtosis = m4 / (m2 ** 2) - 3
        return float(kurtosis)


class F1StrategyEngine:
    """Formula One Strategy Simulation Engine"""
    
    def __init__(self):
        pass
    
    def simulate_race(
        self,
        n_laps: int = 52,
        base_lap_time: float = 90.0,
        tire_wear_rate: float = 0.15,
        fuel_effect: float = -0.06,
        pit_loss: float = 25.0,
        strategy: str = '1-stop',
        sc_probability: float = 0.3,
        n_simulations: int = 1000,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate F1 race with either 1-stop or 2-stop strategy.
        
        Args:
            n_laps: Total race laps
            base_lap_time: Baseline lap time (seconds)
            tire_wear_rate: Tire degradation per lap (seconds)
            fuel_effect: Fuel burn effect per lap (seconds)
            pit_loss: Pit stop time loss (seconds)
            strategy: '1-stop' or '2-stop'
            sc_probability: Probability of safety car occurrence
            n_simulations: Number of race simulations
            seed: Random seed
            
        Returns:
            race_times array (n_simulations,)
        """
        if seed is not None:
            np.random.seed(seed)
        
        race_times = np.zeros(n_simulations)
        sc_events = np.random.rand(n_simulations) < sc_probability
        
        if strategy == '1-stop':
            # One pit stop roughly at mid-point (lap 25-27)
            pit_lap = np.random.randint(25, 28, n_simulations)
            
            for sim in range(n_simulations):
                time = 0.0
                current_tire_age = 0
                sc_occurred = sc_events[sim]
                
                for lap in range(1, n_laps + 1):
                    # Base lap time with tire wear
                    lap_time = base_lap_time + (current_tire_age * tire_wear_rate)
                    
                    # Exponential tire degradation (compounding effect)
                    lap_time += self._exponential_degradation(current_tire_age, rate=0.008)
                    
                    # Fuel effect: lighter car as fuel burns
                    fuel_time_benefit = (lap - 1) * fuel_effect
                    lap_time += fuel_time_benefit
                    
                    # Traffic/overtaking delay if tires too worn (>40 laps on current compound)
                    if current_tire_age > 40:
                        overtaking_prob = min(0.8, current_tire_age / 50)
                        if np.random.rand() < overtaking_prob:
                            lap_time += np.random.uniform(0.5, 2.0)
                    
                    time += lap_time
                    current_tire_age += 1
                    
                    # Safety car pit stop opportunity
                    if lap == pit_lap[sim]:
                        pit_time = pit_loss
                        if sc_occurred:
                            pit_time *= 0.5  # VSC/Safety car reduces pit cost
                            sc_occurred = False  # One-time event
                        
                        time += pit_time
                        current_tire_age = 0  # Fresh tires
                
                race_times[sim] = time
        
        else:  # 2-stop strategy
            pit_lap_1 = np.random.randint(15, 20, n_simulations)
            pit_lap_2 = np.random.randint(35, 42, n_simulations)
            
            for sim in range(n_simulations):
                time = 0.0
                current_tire_age = 0
                sc_occurred = sc_events[sim]
                pit_count = 0
                
                for lap in range(1, n_laps + 1):
                    # Base lap time with tire wear
                    lap_time = base_lap_time + (current_tire_age * tire_wear_rate)
                    
                    # Exponential degradation
                    lap_time += self._exponential_degradation(current_tire_age, rate=0.008)
                    
                    # Fuel effect
                    fuel_time_benefit = (lap - 1) * fuel_effect
                    lap_time += fuel_time_benefit
                    
                    # Traffic delay
                    if current_tire_age > 30:
                        overtaking_prob = min(0.7, current_tire_age / 40)
                        if np.random.rand() < overtaking_prob:
                            lap_time += np.random.uniform(0.3, 1.5)
                    
                    time += lap_time
                    current_tire_age += 1
                    
                    # Pit stops
                    if lap == pit_lap_1[sim] or lap == pit_lap_2[sim]:
                        pit_time = pit_loss
                        if sc_occurred and pit_count == 0:
                            pit_time *= 0.5
                            sc_occurred = False
                        
                        time += pit_time
                        current_tire_age = 0
                        pit_count += 1
                
                race_times[sim] = time
        
        return race_times
    
    def compare_strategies(
        self,
        n_laps: int = 52,
        base_lap_time: float = 90.0,
        tire_wear_rate: float = 0.15,
        fuel_effect: float = -0.06,
        pit_loss: float = 25.0,
        sc_probability: float = 0.3,
        n_simulations: int = 1000,
        seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compare 1-stop vs 2-stop strategies across thousands of simulations.
        
        Returns:
            Dictionary with race_times for each strategy
        """
        times_1stop = self.simulate_race(
            n_laps, base_lap_time, tire_wear_rate, fuel_effect, pit_loss,
            strategy='1-stop', sc_probability=sc_probability,
            n_simulations=n_simulations, seed=seed
        )
        
        times_2stop = self.simulate_race(
            n_laps, base_lap_time, tire_wear_rate, fuel_effect, pit_loss,
            strategy='2-stop', sc_probability=sc_probability,
            n_simulations=n_simulations, seed=seed if seed is None else seed + 1
        )
        
        return {
            '1-stop': times_1stop,
            '2-stop': times_2stop
        }
    
    @staticmethod
    def _exponential_degradation(tire_age: int, rate: float = 0.008) -> float:
        """Compute exponential tire degradation"""
        return rate * (np.exp(tire_age / 20) - 1)


class SimulationFactory:
    """Factory class for creating simulation engines"""
    
    @staticmethod
    def get_finance_engine(risk_free_rate: float = 0.05) -> QuantFinanceEngine:
        return QuantFinanceEngine(risk_free_rate)
    
    @staticmethod
    def get_f1_engine() -> F1StrategyEngine:
        return F1StrategyEngine()
