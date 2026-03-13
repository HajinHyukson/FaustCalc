from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


def _nearest_psd(matrix: np.ndarray, floor: float = 1e-10) -> tuple[np.ndarray, dict[str, float]]:
    symmetric = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    min_eigenvalue = float(np.min(eigenvalues))
    clipped = np.clip(eigenvalues, floor, None)
    repaired = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    repaired = (repaired + repaired.T) / 2.0
    return repaired, {
        "min_eigenvalue_before": min_eigenvalue,
        "min_eigenvalue_after": float(np.min(np.linalg.eigvalsh(repaired))),
        "eigenvalues_clipped": float(np.sum(eigenvalues < floor)),
    }


def _corr_from_cov(cov: np.ndarray) -> np.ndarray:
    vol = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    denom = np.outer(vol, vol)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _annualization_factor(periods_per_year: int, horizon: int = 1) -> float:
    return float(np.sqrt(periods_per_year * max(horizon, 1)))


def _fit_univariate_garch(series: np.ndarray) -> tuple[np.ndarray, dict[str, float | bool | str]]:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        variance = float(np.var(x, ddof=1)) if x.size >= 2 else 1e-6
        cond_vol = np.full(max(x.size, 1), np.sqrt(max(variance, 1e-8)))
        return cond_vol, {
            "converged": False,
            "fallback_used": True,
            "omega": variance * 0.05,
            "alpha": 0.05,
            "beta": 0.90,
            "message": "Insufficient observations; using variance fallback.",
        }

    variance = float(np.var(x, ddof=1))
    variance = max(variance, 1e-8)

    def objective(params: np.ndarray) -> float:
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e12
        sigma2 = np.empty_like(x)
        sigma2[0] = variance
        for idx in range(1, x.size):
            sigma2[idx] = omega + alpha * x[idx - 1] ** 2 + beta * sigma2[idx - 1]
            sigma2[idx] = max(sigma2[idx], 1e-10)
        ll = 0.5 * (np.log(2 * np.pi) + np.log(sigma2) + (x**2) / sigma2)
        return float(np.sum(ll))

    guess = np.array([variance * 0.05, 0.05, 0.90], dtype=float)
    bounds = [(1e-10, variance * 10), (1e-6, 0.35), (1e-6, 0.999)]
    constraints = [{"type": "ineq", "fun": lambda p: 0.999 - p[1] - p[2]}]
    result = minimize(objective, guess, method="SLSQP", bounds=bounds, constraints=constraints)
    fallback_used = not bool(result.success)
    params = result.x if result.success else guess
    omega, alpha, beta = params
    sigma2 = np.empty_like(x)
    sigma2[0] = variance
    for idx in range(1, x.size):
        sigma2[idx] = omega + alpha * x[idx - 1] ** 2 + beta * sigma2[idx - 1]
        sigma2[idx] = max(sigma2[idx], 1e-10)
    return np.sqrt(sigma2), {
        "converged": bool(result.success),
        "fallback_used": fallback_used,
        "omega": float(omega),
        "alpha": float(alpha),
        "beta": float(beta),
        "message": "ok" if result.success else "Optimization failed; fallback parameters applied.",
    }


def _make_corr_estimator(returns: np.ndarray) -> np.ndarray:
    lw = LedoitWolf()
    lw.fit(returns)
    return _corr_from_cov(lw.covariance_)


@dataclass
class CovarianceDiagnostics:
    model: str
    observations: int
    missing_rows_dropped: int
    psd_repaired: bool
    min_eigenvalue_before: float
    min_eigenvalue_after: float
    eigenvalues_clipped: float
    mode: str
    decay: float | None = None
    convergence_warnings: int = 0
    fallback_count: int = 0


class BaseCovarianceModel(ABC):
    def __init__(self, *, periods_per_year: int = 252):
        self.periods_per_year = periods_per_year
        self.asset_names: list[str] = []
        self._cov: np.ndarray | None = None
        self._history: list[np.ndarray] = []
        self._diagnostics: CovarianceDiagnostics | None = None

    @abstractmethod
    def fit(self, returns: pd.DataFrame) -> "BaseCovarianceModel":
        raise NotImplementedError

    def get_cov(self) -> np.ndarray:
        if self._cov is None:
            raise ValueError("Covariance model has not been fit.")
        return self._cov.copy()

    def get_corr(self) -> np.ndarray:
        return _corr_from_cov(self.get_cov())

    def get_vol(self, *, annualized: bool = False) -> np.ndarray:
        vol = np.sqrt(np.clip(np.diag(self.get_cov()), 0.0, None))
        if annualized:
            vol = vol * _annualization_factor(self.periods_per_year)
        return vol

    def forecast(self, horizon: int = 1) -> np.ndarray:
        return self.get_cov() * max(horizon, 1)

    def diagnostics(self) -> dict[str, float | int | str | None]:
        if self._diagnostics is None:
            raise ValueError("Covariance model has not been fit.")
        return self._diagnostics.__dict__.copy()

    def covariance_history(self) -> list[np.ndarray]:
        return [cov.copy() for cov in self._history]


class LedoitWolfCovarianceModel(BaseCovarianceModel):
    def fit(self, returns: pd.DataFrame) -> "LedoitWolfCovarianceModel":
        clean = returns.dropna(how="any")
        if clean.empty:
            raise ValueError("Returns matrix is empty after dropping missing values.")

        lw = LedoitWolf()
        lw.fit(clean.values)
        cov, repair = _nearest_psd(lw.covariance_)
        self.asset_names = list(clean.columns)
        self._cov = cov
        self._history = [cov.copy()]
        self._diagnostics = CovarianceDiagnostics(
            model="ledoit_wolf",
            observations=int(clean.shape[0]),
            missing_rows_dropped=int(returns.shape[0] - clean.shape[0]),
            psd_repaired=bool(repair["eigenvalues_clipped"] > 0),
            min_eigenvalue_before=repair["min_eigenvalue_before"],
            min_eigenvalue_after=repair["min_eigenvalue_after"],
            eigenvalues_clipped=repair["eigenvalues_clipped"],
            mode="point_in_time",
            decay=None,
        )
        return self


class EWMACovarianceModel(BaseCovarianceModel):
    def __init__(
        self,
        *,
        decay: float = 0.94,
        mode: str = "expanding",
        periods_per_year: int = 252,
    ):
        super().__init__(periods_per_year=periods_per_year)
        if not 0.0 < decay < 1.0:
            raise ValueError("EWMA decay must be in (0, 1).")
        if mode not in {"expanding", "rolling"}:
            raise ValueError("EWMA mode must be 'expanding' or 'rolling'.")
        self.decay = decay
        self.mode = mode

    def fit(self, returns: pd.DataFrame) -> "EWMACovarianceModel":
        clean = returns.dropna(how="any")
        if clean.empty:
            raise ValueError("Returns matrix is empty after dropping missing values.")
        if clean.shape[0] < 2:
            raise ValueError("EWMA requires at least two observations.")

        values = clean.values
        n_obs, n_assets = values.shape
        history: list[np.ndarray] = []
        cov = np.cov(values[:2].T, ddof=1)
        if np.isscalar(cov):
            cov = np.array([[float(cov)]], dtype=float)

        for idx in range(2, n_obs + 1):
            r_t = values[idx - 1].reshape(n_assets, 1)
            cov = self.decay * cov + (1.0 - self.decay) * (r_t @ r_t.T)
            cov, _ = _nearest_psd(cov)
            history.append(cov.copy())

        final_cov, repair = _nearest_psd(history[-1])
        self.asset_names = list(clean.columns)
        self._cov = final_cov
        self._history = history
        self._diagnostics = CovarianceDiagnostics(
            model="ewma",
            observations=int(clean.shape[0]),
            missing_rows_dropped=int(returns.shape[0] - clean.shape[0]),
            psd_repaired=bool(repair["eigenvalues_clipped"] > 0),
            min_eigenvalue_before=repair["min_eigenvalue_before"],
            min_eigenvalue_after=repair["min_eigenvalue_after"],
            eigenvalues_clipped=repair["eigenvalues_clipped"],
            mode=self.mode,
            decay=self.decay,
        )
        return self


class GARCHCovarianceModel(BaseCovarianceModel):
    def __init__(self, *, periods_per_year: int = 252, correlation_method: str = "ledoit_wolf"):
        super().__init__(periods_per_year=periods_per_year)
        self.correlation_method = correlation_method

    def fit(self, returns: pd.DataFrame) -> "GARCHCovarianceModel":
        clean = returns.dropna(how="any")
        if clean.empty:
            raise ValueError("Returns matrix is empty after dropping missing values.")
        if clean.shape[0] < 10:
            raise ValueError("GARCH covariance requires at least 10 observations.")

        values = clean.values
        n_obs, n_assets = values.shape
        cond_vols = np.zeros((n_obs, n_assets), dtype=float)
        warnings = 0
        fallbacks = 0
        for idx in range(n_assets):
            vol_series, info = _fit_univariate_garch(values[:, idx])
            cond_vols[:, idx] = vol_series
            warnings += int(not bool(info["converged"]))
            fallbacks += int(bool(info["fallback_used"]))

        standardized = np.divide(
            values,
            cond_vols,
            out=np.zeros_like(values),
            where=cond_vols > 0,
        )
        corr = _make_corr_estimator(standardized if self.correlation_method == "ledoit_wolf" else values)

        history: list[np.ndarray] = []
        for idx in range(n_obs):
            d = np.diag(cond_vols[idx])
            cov = d @ corr @ d
            cov, _ = _nearest_psd(cov)
            history.append(cov.copy())

        final_cov, repair = _nearest_psd(history[-1])
        self.asset_names = list(clean.columns)
        self._cov = final_cov
        self._history = history
        self._diagnostics = CovarianceDiagnostics(
            model="garch",
            observations=int(clean.shape[0]),
            missing_rows_dropped=int(returns.shape[0] - clean.shape[0]),
            psd_repaired=bool(repair["eigenvalues_clipped"] > 0),
            min_eigenvalue_before=repair["min_eigenvalue_before"],
            min_eigenvalue_after=repair["min_eigenvalue_after"],
            eigenvalues_clipped=repair["eigenvalues_clipped"],
            mode="time_varying",
            decay=None,
            convergence_warnings=warnings,
            fallback_count=fallbacks,
        )
        return self


class DCCGARCHCovarianceModel(BaseCovarianceModel):
    def __init__(
        self,
        *,
        periods_per_year: int = 252,
        dcc_alpha: float = 0.03,
        dcc_beta: float = 0.95,
    ):
        super().__init__(periods_per_year=periods_per_year)
        if dcc_alpha < 0 or dcc_beta < 0 or dcc_alpha + dcc_beta >= 0.999:
            raise ValueError("DCC parameters must be non-negative and sum to less than 1.")
        self.dcc_alpha = dcc_alpha
        self.dcc_beta = dcc_beta

    def fit(self, returns: pd.DataFrame) -> "DCCGARCHCovarianceModel":
        clean = returns.dropna(how="any")
        if clean.empty:
            raise ValueError("Returns matrix is empty after dropping missing values.")
        if clean.shape[0] < 20:
            raise ValueError("DCC-GARCH requires at least 20 observations.")

        values = clean.values
        n_obs, n_assets = values.shape
        cond_vols = np.zeros((n_obs, n_assets), dtype=float)
        warnings = 0
        fallbacks = 0
        for idx in range(n_assets):
            vol_series, info = _fit_univariate_garch(values[:, idx])
            cond_vols[:, idx] = vol_series
            warnings += int(not bool(info["converged"]))
            fallbacks += int(bool(info["fallback_used"]))

        standardized = np.divide(values, cond_vols, out=np.zeros_like(values), where=cond_vols > 0)
        q_bar = np.cov(standardized.T, ddof=1)
        if np.isscalar(q_bar):
            q_bar = np.array([[float(q_bar)]], dtype=float)
        q_t = q_bar.copy()
        history: list[np.ndarray] = []
        for idx in range(n_obs):
            if idx > 0:
                resid = standardized[idx - 1].reshape(-1, 1)
                q_t = (
                    (1.0 - self.dcc_alpha - self.dcc_beta) * q_bar
                    + self.dcc_alpha * (resid @ resid.T)
                    + self.dcc_beta * q_t
                )
            q_t, _ = _nearest_psd(q_t)
            q_diag = np.sqrt(np.clip(np.diag(q_t), 1e-12, None))
            inv_diag = np.diag(1.0 / q_diag)
            r_t = inv_diag @ q_t @ inv_diag
            r_t, _ = _nearest_psd(r_t)
            d_t = np.diag(cond_vols[idx])
            cov = d_t @ r_t @ d_t
            cov, _ = _nearest_psd(cov)
            history.append(cov.copy())

        final_cov, repair = _nearest_psd(history[-1])
        self.asset_names = list(clean.columns)
        self._cov = final_cov
        self._history = history
        self._diagnostics = CovarianceDiagnostics(
            model="dcc_garch",
            observations=int(clean.shape[0]),
            missing_rows_dropped=int(returns.shape[0] - clean.shape[0]),
            psd_repaired=bool(repair["eigenvalues_clipped"] > 0),
            min_eigenvalue_before=repair["min_eigenvalue_before"],
            min_eigenvalue_after=repair["min_eigenvalue_after"],
            eigenvalues_clipped=repair["eigenvalues_clipped"],
            mode="time_varying",
            decay=None,
            convergence_warnings=warnings,
            fallback_count=fallbacks,
        )
        return self


def make_covariance_model(
    method: str,
    *,
    periods_per_year: int,
    ewma_decay: float = 0.94,
    ewma_mode: str = "expanding",
) -> BaseCovarianceModel:
    normalized = method.strip().lower()
    if normalized == "ledoit_wolf":
        return LedoitWolfCovarianceModel(periods_per_year=periods_per_year)
    if normalized == "ewma":
        return EWMACovarianceModel(
            decay=ewma_decay,
            mode=ewma_mode,
            periods_per_year=periods_per_year,
        )
    if normalized == "garch":
        return GARCHCovarianceModel(periods_per_year=periods_per_year)
    if normalized == "dcc_garch":
        return DCCGARCHCovarianceModel(periods_per_year=periods_per_year)
    raise ValueError(f"Unsupported covariance method: {method}")


def rolling_covariance_forecasts(
    returns: pd.DataFrame,
    *,
    method: str,
    periods_per_year: int,
    window: int,
    horizon: int = 1,
    min_periods: int | None = None,
    ewma_decay: float = 0.94,
    ewma_mode: str = "expanding",
) -> pd.Series:
    if window < 2:
        raise ValueError("Rolling covariance forecasts require window >= 2.")

    min_obs = max(2, min_periods or window)
    forecasts: dict[pd.Timestamp, np.ndarray] = {}
    for end_idx in range(min_obs, len(returns) + 1):
        if method == "ewma" and ewma_mode == "expanding":
            sample = returns.iloc[:end_idx]
        else:
            sample = returns.iloc[max(0, end_idx - window):end_idx]
        model = make_covariance_model(
            method,
            periods_per_year=periods_per_year,
            ewma_decay=ewma_decay,
            ewma_mode=ewma_mode,
        ).fit(sample)
        forecasts[returns.index[end_idx - 1]] = model.forecast(horizon=horizon)
    return pd.Series(forecasts, dtype=object)


def annualize_covariance(cov: np.ndarray, periods_per_year: int) -> np.ndarray:
    return np.asarray(cov, dtype=float) * float(periods_per_year)


def covariance_forecast_error(
    realized_returns: pd.DataFrame,
    forecasts: pd.Series,
) -> dict[str, float]:
    if forecasts.empty:
        return {"forecast_rmse": 0.0, "forecast_mae": 0.0}

    errors: list[float] = []
    abs_errors: list[float] = []
    for idx, cov in forecasts.items():
        if idx not in realized_returns.index:
            continue
        realized = realized_returns.loc[idx].values.reshape(-1, 1)
        realized_cov = realized @ realized.T
        error = float(np.sqrt(np.mean((np.asarray(cov) - realized_cov) ** 2)))
        errors.append(error**2)
        abs_errors.append(float(np.mean(np.abs(np.asarray(cov) - realized_cov))))
    if not errors:
        return {"forecast_rmse": 0.0, "forecast_mae": 0.0}
    return {
        "forecast_rmse": float(np.sqrt(np.mean(errors))),
        "forecast_mae": float(np.mean(abs_errors)),
    }
