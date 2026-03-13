from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def _zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=1))
    if std <= 0 or not np.isfinite(std):
        return pd.Series(0.0, index=series.index)
    return (series - float(series.mean())) / std


def _ols_loadings(returns: pd.DataFrame, factor_data: pd.DataFrame) -> pd.DataFrame:
    x = factor_data.values
    x = np.column_stack([np.ones(len(factor_data)), x])
    betas: list[np.ndarray] = []
    for col in returns.columns:
        y = returns[col].values
        coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        betas.append(coef[1:])
    return pd.DataFrame(betas, index=returns.columns, columns=factor_data.columns)


def build_macro_factor_proxies(returns: pd.DataFrame, volumes: pd.DataFrame | None = None) -> pd.DataFrame:
    clean = returns.dropna(how="any")
    market = clean.mean(axis=1)
    volatility = clean.std(axis=1, ddof=1)
    downside = clean.clip(upper=0).mean(axis=1).abs()
    data = {
        "macro_growth": market,
        "macro_volatility": volatility,
        "macro_stress": downside,
    }
    if volumes is not None and not volumes.empty:
        aligned = volumes.reindex(clean.index).ffill()
        liquidity = aligned.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).mean(axis=1)
        data["macro_liquidity"] = liquidity
    return pd.DataFrame(data, index=clean.index).fillna(0.0)


def build_style_descriptor_proxies(prices: pd.DataFrame, returns: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    clean_returns = returns.dropna(how="any")
    latest_prices = prices.reindex(clean_returns.index).ffill().iloc[-1]
    adv_dollars = (prices * volumes).reindex(clean_returns.index).tail(min(20, len(clean_returns))).mean()
    market = clean_returns.mean(axis=1)

    momentum = clean_returns.tail(min(60, len(clean_returns))).mean()
    volatility = clean_returns.std(ddof=1)
    liquidity = np.log(np.clip(adv_dollars, 1.0, None))
    size = np.log(np.clip(latest_prices * volumes.reindex(clean_returns.index).ffill().iloc[-1], 1.0, None))
    beta = clean_returns.apply(
        lambda col: np.cov(col.values, market.values, ddof=1)[0, 1] / max(np.var(market.values, ddof=1), 1e-10)
    )
    quality = momentum / np.clip(volatility, 1e-10, None)
    value = -np.log(np.clip(latest_prices, 1e-10, None))

    descriptors = pd.DataFrame(
        {
            "value": _zscore(value),
            "momentum": _zscore(momentum),
            "size": _zscore(size),
            "volatility": _zscore(-volatility),
            "quality": _zscore(quality),
            "liquidity": _zscore(liquidity),
            "beta": _zscore(beta),
        }
    )
    return descriptors.fillna(0.0)


@dataclass
class FactorDiagnostics:
    model: str
    observations: int
    n_factors: int
    explained_variance_ratio: float


class BaseFactorModel(ABC):
    @abstractmethod
    def fit(self, returns: pd.DataFrame, **kwargs) -> "BaseFactorModel":
        raise NotImplementedError

    @abstractmethod
    def exposures(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def factor_returns(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def factor_covariance(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def specific_risk(self) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def diagnostics(self) -> dict[str, float | int | str]:
        raise NotImplementedError

    def factor_risk_decomposition(self, weights: np.ndarray) -> dict[str, float]:
        w = np.asarray(weights, dtype=float)
        exposures = self.exposures().values
        factor_cov = self.factor_covariance()
        specific = np.diag(self.specific_risk().values)
        factor_var = float(w @ exposures @ factor_cov @ exposures.T @ w)
        specific_var = float(w @ specific @ w)
        total = max(factor_var + specific_var, 1e-12)
        return {
            "factor_variance_share": factor_var / total,
            "specific_variance_share": specific_var / total,
        }


class PCAFactorModel(BaseFactorModel):
    def __init__(self, *, n_factors: int | None = None, explained_variance_target: float = 0.8):
        self.n_factors = n_factors
        self.explained_variance_target = explained_variance_target
        self._exposures: pd.DataFrame | None = None
        self._factor_returns: pd.DataFrame | None = None
        self._specific_var: pd.Series | None = None
        self._factor_cov: np.ndarray | None = None
        self._diagnostics: FactorDiagnostics | None = None

    def fit(self, returns: pd.DataFrame, **kwargs) -> "PCAFactorModel":
        del kwargs
        clean = returns.dropna(how="any")
        if clean.empty:
            raise ValueError("Returns matrix is empty after dropping missing values.")
        centered = clean - clean.mean(axis=0)
        max_components = min(centered.shape)
        if self.n_factors is None:
            probe = PCA(n_components=max_components)
            probe.fit(centered.values)
            cumulative = np.cumsum(probe.explained_variance_ratio_)
            chosen = int(np.searchsorted(cumulative, self.explained_variance_target) + 1)
            n_components = max(1, min(chosen, max_components))
        else:
            n_components = max(1, min(self.n_factors, max_components))

        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(centered.values)
        factor_names = [f"PC{idx + 1}" for idx in range(n_components)]
        exposures = pd.DataFrame(pca.components_.T, index=clean.columns, columns=factor_names)
        factor_returns = pd.DataFrame(scores, index=clean.index, columns=factor_names)
        reconstructed = factor_returns.values @ pca.components_
        residuals = centered.values - reconstructed
        specific_var = pd.Series(np.var(residuals, axis=0, ddof=1), index=clean.columns, name="specific_var")

        self._exposures = exposures
        self._factor_returns = factor_returns
        self._specific_var = specific_var
        self._factor_cov = np.cov(factor_returns.values.T, ddof=1)
        if np.isscalar(self._factor_cov):
            self._factor_cov = np.array([[float(self._factor_cov)]], dtype=float)
        self._diagnostics = FactorDiagnostics(
            model="pca",
            observations=int(clean.shape[0]),
            n_factors=n_components,
            explained_variance_ratio=float(np.sum(pca.explained_variance_ratio_)),
        )
        return self

    def exposures(self) -> pd.DataFrame:
        if self._exposures is None:
            raise ValueError("Factor model has not been fit.")
        return self._exposures.copy()

    def factor_returns(self) -> pd.DataFrame:
        if self._factor_returns is None:
            raise ValueError("Factor model has not been fit.")
        return self._factor_returns.copy()

    def factor_covariance(self) -> np.ndarray:
        if self._factor_cov is None:
            raise ValueError("Factor model has not been fit.")
        return self._factor_cov.copy()

    def specific_risk(self) -> pd.Series:
        if self._specific_var is None:
            raise ValueError("Factor model has not been fit.")
        return self._specific_var.copy()

    def diagnostics(self) -> dict[str, float | int | str]:
        if self._diagnostics is None:
            raise ValueError("Factor model has not been fit.")
        return self._diagnostics.__dict__.copy()


class RegressionFactorModel(BaseFactorModel):
    def __init__(self, *, model_name: str):
        self.model_name = model_name
        self._exposures: pd.DataFrame | None = None
        self._factor_returns: pd.DataFrame | None = None
        self._specific_var: pd.Series | None = None
        self._factor_cov: np.ndarray | None = None
        self._diagnostics: FactorDiagnostics | None = None

    def fit(self, returns: pd.DataFrame, *, factor_data: pd.DataFrame) -> "RegressionFactorModel":
        clean = returns.dropna(how="any")
        aligned = factor_data.reindex(clean.index).dropna(how="any")
        clean = clean.reindex(aligned.index)
        if clean.empty or aligned.empty:
            raise ValueError("Aligned factor inputs are empty.")
        exposures = _ols_loadings(clean, aligned)
        predicted = aligned.values @ exposures.T.values
        residuals = clean.values - predicted
        self._exposures = exposures
        self._factor_returns = aligned.copy()
        self._specific_var = pd.Series(np.var(residuals, axis=0, ddof=1), index=clean.columns, name="specific_var")
        self._factor_cov = np.cov(aligned.values.T, ddof=1)
        if np.isscalar(self._factor_cov):
            self._factor_cov = np.array([[float(self._factor_cov)]], dtype=float)
        total_var = float(clean.var(ddof=1).sum())
        residual_var = float(np.var(residuals, axis=0, ddof=1).sum())
        explained = 0.0 if total_var <= 0 else max(0.0, min(1.0, 1.0 - residual_var / total_var))
        self._diagnostics = FactorDiagnostics(
            model=self.model_name,
            observations=int(clean.shape[0]),
            n_factors=int(aligned.shape[1]),
            explained_variance_ratio=explained,
        )
        return self

    def exposures(self) -> pd.DataFrame:
        if self._exposures is None:
            raise ValueError("Factor model has not been fit.")
        return self._exposures.copy()

    def factor_returns(self) -> pd.DataFrame:
        if self._factor_returns is None:
            raise ValueError("Factor model has not been fit.")
        return self._factor_returns.copy()

    def factor_covariance(self) -> np.ndarray:
        if self._factor_cov is None:
            raise ValueError("Factor model has not been fit.")
        return self._factor_cov.copy()

    def specific_risk(self) -> pd.Series:
        if self._specific_var is None:
            raise ValueError("Factor model has not been fit.")
        return self._specific_var.copy()

    def diagnostics(self) -> dict[str, float | int | str]:
        if self._diagnostics is None:
            raise ValueError("Factor model has not been fit.")
        return self._diagnostics.__dict__.copy()


class MacroFactorModel(RegressionFactorModel):
    def __init__(self):
        super().__init__(model_name="macro")

    def fit(
        self,
        returns: pd.DataFrame,
        *,
        factor_data: pd.DataFrame | None = None,
        volumes: pd.DataFrame | None = None,
    ) -> "MacroFactorModel":
        factors = factor_data if factor_data is not None else build_macro_factor_proxies(returns, volumes)
        super().fit(returns, factor_data=factors)
        return self


class StyleFactorModel(RegressionFactorModel):
    def __init__(self):
        super().__init__(model_name="style")

    def fit(
        self,
        returns: pd.DataFrame,
        *,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
    ) -> "StyleFactorModel":
        descriptors = build_style_descriptor_proxies(prices, returns, volumes)
        factor_returns = pd.DataFrame(
            index=returns.index,
            data={
                name: (returns * descriptors[name].reindex(returns.columns)).mean(axis=1)
                for name in descriptors.columns
            },
        )
        super().fit(returns, factor_data=factor_returns)
        self._exposures = descriptors
        return self


def make_factor_model(method: str, *, n_factors: int | None = None) -> BaseFactorModel:
    normalized = method.strip().lower()
    if normalized == "pca":
        return PCAFactorModel(n_factors=n_factors)
    if normalized == "macro":
        return MacroFactorModel()
    if normalized == "style":
        return StyleFactorModel()
    raise ValueError(f"Unsupported factor model: {method}")
