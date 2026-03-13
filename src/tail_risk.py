from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .covariance import make_covariance_model


@dataclass
class TailRiskDiagnostics:
    model: str
    observations: int
    horizon: int
    confidence: float
    mean_included: bool
    sparse_rows_dropped: int
    random_seed: int | None = None
    simulation_count: int | None = None
    covariance_method: str | None = None


class BaseTailRiskModel(ABC):
    @abstractmethod
    def fit(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        exposures: dict[str, float] | None = None,
    ) -> "BaseTailRiskModel":
        raise NotImplementedError

    @abstractmethod
    def loss_distribution(self) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def var(self, confidence: float, horizon: int = 1) -> float:
        raise NotImplementedError

    @abstractmethod
    def es(self, confidence: float, horizon: int = 1) -> float:
        raise NotImplementedError

    @abstractmethod
    def diagnostics(self) -> dict[str, float | int | str | bool | None]:
        raise NotImplementedError


class HistoricalTailRiskModel(BaseTailRiskModel):
    def __init__(self, *, lookback: int | None = None, include_mean: bool = False):
        self.lookback = lookback
        self.include_mean = include_mean
        self._losses: pd.Series | None = None
        self._diagnostics: TailRiskDiagnostics | None = None

    def fit(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        exposures: dict[str, float] | None = None,
    ) -> "HistoricalTailRiskModel":
        del exposures
        clean = returns.dropna(how="any")
        if clean.empty:
            raise ValueError("Returns matrix is empty after dropping missing values.")
        if self.lookback is not None and self.lookback > 0:
            clean = clean.tail(self.lookback)
        w = np.asarray(weights, dtype=float)
        if clean.shape[1] != w.shape[0]:
            raise ValueError("Weights length does not match returns columns.")
        pnl = clean @ w
        centered = pnl if self.include_mean else pnl - pnl.mean()
        losses = -centered
        losses.name = "loss"
        self._losses = losses
        self._diagnostics = TailRiskDiagnostics(
            model="historical",
            observations=int(clean.shape[0]),
            horizon=1,
            confidence=0.95,
            mean_included=self.include_mean,
            sparse_rows_dropped=int(returns.shape[0] - clean.shape[0]),
        )
        return self

    def loss_distribution(self) -> pd.Series:
        if self._losses is None:
            raise ValueError("Tail-risk model has not been fit.")
        return self._losses.copy()

    def var(self, confidence: float, horizon: int = 1) -> float:
        losses = self.loss_distribution()
        scaled = losses * np.sqrt(max(horizon, 1))
        return float(np.quantile(scaled, confidence))

    def es(self, confidence: float, horizon: int = 1) -> float:
        losses = self.loss_distribution() * np.sqrt(max(horizon, 1))
        threshold = float(np.quantile(losses, confidence))
        tail = losses[losses >= threshold]
        if tail.empty:
            return threshold
        return float(tail.mean())

    def diagnostics(self) -> dict[str, float | int | str | bool | None]:
        if self._diagnostics is None:
            raise ValueError("Tail-risk model has not been fit.")
        return self._diagnostics.__dict__.copy()


class MonteCarloTailRiskModel(BaseTailRiskModel):
    def __init__(
        self,
        *,
        covariance_method: str = "ledoit_wolf",
        simulation_count: int = 10000,
        random_seed: int = 42,
        include_mean: bool = False,
        ewma_decay: float = 0.94,
        ewma_mode: str = "expanding",
    ):
        self.covariance_method = covariance_method
        self.simulation_count = simulation_count
        self.random_seed = random_seed
        self.include_mean = include_mean
        self.ewma_decay = ewma_decay
        self.ewma_mode = ewma_mode
        self._losses: pd.Series | None = None
        self._diagnostics: TailRiskDiagnostics | None = None

    def fit(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        exposures: dict[str, float] | None = None,
    ) -> "MonteCarloTailRiskModel":
        del exposures
        clean = returns.dropna(how="any")
        if clean.empty:
            raise ValueError("Returns matrix is empty after dropping missing values.")
        w = np.asarray(weights, dtype=float)
        if clean.shape[1] != w.shape[0]:
            raise ValueError("Weights length does not match returns columns.")

        mean = clean.mean().values if self.include_mean else np.zeros(clean.shape[1], dtype=float)
        cov = make_covariance_model(
            self.covariance_method,
            periods_per_year=252,
            ewma_decay=self.ewma_decay,
            ewma_mode=self.ewma_mode,
        ).fit(clean).get_cov()
        rng = np.random.default_rng(self.random_seed)
        simulated = rng.multivariate_normal(mean=mean, cov=cov, size=self.simulation_count)
        losses = -(simulated @ w)
        self._losses = pd.Series(losses, name="loss")
        self._diagnostics = TailRiskDiagnostics(
            model="monte_carlo",
            observations=int(clean.shape[0]),
            horizon=1,
            confidence=0.95,
            mean_included=self.include_mean,
            sparse_rows_dropped=int(returns.shape[0] - clean.shape[0]),
            random_seed=self.random_seed,
            simulation_count=self.simulation_count,
            covariance_method=self.covariance_method,
        )
        return self

    def loss_distribution(self) -> pd.Series:
        if self._losses is None:
            raise ValueError("Tail-risk model has not been fit.")
        return self._losses.copy()

    def var(self, confidence: float, horizon: int = 1) -> float:
        losses = self.loss_distribution() * np.sqrt(max(horizon, 1))
        return float(np.quantile(losses, confidence))

    def es(self, confidence: float, horizon: int = 1) -> float:
        losses = self.loss_distribution() * np.sqrt(max(horizon, 1))
        threshold = float(np.quantile(losses, confidence))
        tail = losses[losses >= threshold]
        if tail.empty:
            return threshold
        return float(tail.mean())

    def diagnostics(self) -> dict[str, float | int | str | bool | None]:
        if self._diagnostics is None:
            raise ValueError("Tail-risk model has not been fit.")
        return self._diagnostics.__dict__.copy()


def make_tail_risk_model(
    method: str,
    *,
    lookback: int | None = None,
    include_mean: bool = False,
    covariance_method: str = "ledoit_wolf",
    simulation_count: int = 10000,
    random_seed: int = 42,
    ewma_decay: float = 0.94,
    ewma_mode: str = "expanding",
) -> BaseTailRiskModel:
    normalized = method.strip().lower()
    if normalized == "historical":
        return HistoricalTailRiskModel(lookback=lookback, include_mean=include_mean)
    if normalized == "monte_carlo":
        return MonteCarloTailRiskModel(
            covariance_method=covariance_method,
            simulation_count=simulation_count,
            random_seed=random_seed,
            include_mean=include_mean,
            ewma_decay=ewma_decay,
            ewma_mode=ewma_mode,
        )
    raise ValueError(f"Unsupported tail-risk method: {method}")


def rolling_tail_forecasts(
    returns: pd.DataFrame,
    *,
    weights: np.ndarray,
    method: str,
    confidence: float,
    window: int,
    horizon: int = 1,
    include_mean: bool = False,
    covariance_method: str = "ledoit_wolf",
    simulation_count: int = 10000,
    random_seed: int = 42,
    ewma_decay: float = 0.94,
    ewma_mode: str = "expanding",
) -> pd.DataFrame:
    if window < 2:
        raise ValueError("Rolling tail forecasts require window >= 2.")

    records: list[dict[str, float | pd.Timestamp]] = []
    for end_idx in range(window, len(returns) + 1):
        sample = returns.iloc[end_idx - window:end_idx]
        model = make_tail_risk_model(
            method,
            lookback=window,
            include_mean=include_mean,
            covariance_method=covariance_method,
            simulation_count=simulation_count,
            random_seed=random_seed,
            ewma_decay=ewma_decay,
            ewma_mode=ewma_mode,
        ).fit(sample, weights)
        realized_loss = float(-(returns.iloc[end_idx - 1] @ weights))
        var_value = model.var(confidence, horizon=horizon)
        records.append(
            {
                "date": returns.index[end_idx - 1],
                "var": var_value,
                "es": model.es(confidence, horizon=horizon),
                "realized_loss": realized_loss,
                "exception": float(realized_loss > var_value),
            }
        )
    return pd.DataFrame.from_records(records).set_index("date")
