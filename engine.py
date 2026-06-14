"""
Advanced Multi-Domain Simulation Engine
Quantitative Financial Engineering & Formula One Strategy Analytics

Mathematical backend for the Streamlit dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class HestonParams:
	"""Heston stochastic volatility model parameters."""

	kappa: float
	theta: float
	sigma: float
	rho: float
	v0: float


class QuantFinanceEngine:
	"""Quantitative finance simulation engine."""

	def __init__(self, risk_free_rate: float = 0.05):
		self.risk_free_rate = risk_free_rate

	def heston_monte_carlo(
		self,
		S0: np.ndarray,
		T: float = 1.0,
		steps: int = 252,
		paths: int = 1000,
		params: Optional[HestonParams] = None,
		seed: Optional[int] = None,
	) -> Tuple[np.ndarray, np.ndarray]:
		"""Simulate asset and variance paths with a Heston-style Euler scheme."""

		if seed is not None:
			np.random.seed(seed)

		if params is None:
			params = HestonParams(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04)

		S0 = np.atleast_1d(np.asarray(S0, dtype=float))
		n_assets = S0.shape[0]
		dt = T / max(steps, 1)

		price_paths = np.zeros((steps, paths, n_assets), dtype=float)
		variance_paths = np.zeros((steps, paths, n_assets), dtype=float)
		price_paths[0] = S0
		variance_paths[0] = params.v0

		z_s = np.random.standard_normal((steps - 1, paths, n_assets))
		z_v = np.random.standard_normal((steps - 1, paths, n_assets))
		w_v = params.rho * z_s + np.sqrt(max(1.0 - params.rho**2, 0.0)) * z_v
		sqrt_dt = np.sqrt(dt)

		for t in range(1, steps):
			s_prev = price_paths[t - 1]
			v_prev = np.maximum(variance_paths[t - 1], 1e-8)

			v_next = v_prev + params.kappa * (params.theta - v_prev) * dt
			v_next += params.sigma * np.sqrt(v_prev) * w_v[t - 1] * sqrt_dt
			v_next = np.maximum(v_next, 1e-8)

			price_paths[t] = s_prev * np.exp(
				(self.risk_free_rate - 0.5 * v_next) * dt
				+ np.sqrt(v_next) * z_s[t - 1] * sqrt_dt
			)
			variance_paths[t] = v_next

		return price_paths, variance_paths

	def generate_random_weights(self, n_assets: int, n_portfolios: int = 1000) -> np.ndarray:
		"""Generate Dirichlet-distributed portfolio weights."""

		return np.random.dirichlet(np.ones(n_assets), size=n_portfolios)

	@staticmethod
	def _compute_excess_kurtosis(data: np.ndarray) -> float:
		centered = data - np.mean(data)
		std = np.std(centered)
		if std < 1e-12:
			return 0.0
		fourth_moment = np.mean(centered**4)
		return float(fourth_moment / (std**4) - 3.0)

	def compute_portfolio_metrics(self, returns: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
		"""Compute return, volatility, Sharpe, VaR, ES, and kurtosis."""

		portfolio_returns = np.tensordot(returns, weights, axes=([2], [0]))
		final_returns = portfolio_returns[-1]

		var_95 = np.percentile(final_returns, 5)
		tail_returns = final_returns[final_returns <= var_95]
		es_95 = float(np.mean(tail_returns)) if tail_returns.size else float(var_95)

		annual_return = float(np.mean(final_returns))
		annual_volatility = float(np.std(final_returns))
		sharpe_ratio = (annual_return - self.risk_free_rate) / (annual_volatility + 1e-8)

		return {
			"return": annual_return,
			"volatility": annual_volatility,
			"var_95": float(var_95),
			"es_95": es_95,
			"excess_kurtosis": self._compute_excess_kurtosis(final_returns),
			"sharpe_ratio": float(sharpe_ratio),
		}

	def efficient_frontier(
		self,
		price_paths: np.ndarray,
		n_portfolios: int = 500,
	) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		"""Sample random portfolios and compute frontier-style metrics."""

		log_returns = np.diff(np.log(price_paths), axis=0)
		n_assets = price_paths.shape[2]
		weights = self.generate_random_weights(n_assets, n_portfolios)

		volatilities = np.zeros(n_portfolios, dtype=float)
		returns = np.zeros(n_portfolios, dtype=float)
		sharpe_ratios = np.zeros(n_portfolios, dtype=float)
		expected_shortfalls = np.zeros(n_portfolios, dtype=float)

		for i in range(n_portfolios):
			metrics = self.compute_portfolio_metrics(log_returns, weights[i])
			volatilities[i] = metrics["volatility"]
			returns[i] = metrics["return"]
			sharpe_ratios[i] = metrics["sharpe_ratio"]
			expected_shortfalls[i] = metrics["es_95"]

		return volatilities, returns, sharpe_ratios, expected_shortfalls


class F1StrategyEngine:
	"""Formula One strategy simulation engine."""

	def compare_strategies(
		self,
		n_laps: int,
		base_lap_time: float,
		tire_wear_rate: float,
		fuel_effect: float,
		pit_loss: float,
		sc_probability: float,
		n_simulations: int = 1000,
		seed: Optional[int] = None,
	) -> Dict[str, np.ndarray]:
		"""Compare simple 1-stop and 2-stop race strategies."""

		if seed is not None:
			np.random.seed(seed)

		def simulate(strategy_stops: int) -> np.ndarray:
			pit_laps = np.array([], dtype=int)
			if strategy_stops == 1:
				pit_laps = np.array([n_laps // 2])
			elif strategy_stops == 2:
				pit_laps = np.array([n_laps // 3, 2 * n_laps // 3])

			finish_times = np.zeros(n_simulations, dtype=float)
			for sim in range(n_simulations):
				total_time = 0.0
				wear_multiplier = 1.0
				fuel_adjustment = 0.0

				for lap in range(1, n_laps + 1):
					if lap in pit_laps:
						total_time += pit_loss
						wear_multiplier = 1.0

					sc_delay = 0.0
					if np.random.rand() < sc_probability:
						sc_delay = np.random.uniform(0.5, 3.0)

					lap_time = base_lap_time
					lap_time += tire_wear_rate * wear_multiplier * lap
					lap_time += fuel_adjustment
					lap_time += sc_delay

					total_time += lap_time
					wear_multiplier += 0.02
					fuel_adjustment += fuel_effect

				total_time += np.random.normal(0.0, 1.5)
				finish_times[sim] = max(total_time, 0.0)

			return finish_times

		return {
			"1-stop": simulate(1),
			"2-stop": simulate(2),
		}


class SimulationFactory:
	"""Factory for simulation engines."""

	@staticmethod
	def get_finance_engine(risk_free_rate: float = 0.05) -> QuantFinanceEngine:
		return QuantFinanceEngine(risk_free_rate=risk_free_rate)

	@staticmethod
	def get_f1_engine() -> F1StrategyEngine:
		return F1StrategyEngine()

