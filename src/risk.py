import numpy as np

from .covariance import make_covariance_model


def covariance_matrix(returns, method="ledoit_wolf", *, periods_per_year=252, **kwargs):
    """Estimate covariance using the configured covariance model."""
    model = make_covariance_model(
        method,
        periods_per_year=periods_per_year,
        ewma_decay=float(kwargs.get("ewma_decay", 0.94)),
        ewma_mode=str(kwargs.get("ewma_mode", "expanding")),
    ).fit(returns)
    return model.get_cov()


def portfolio_vol(weights, cov, periods_per_year=52):
    """Annualized portfolio volatility from covariance matrix."""
    variance = float(weights.T @ cov @ weights)
    return float(np.sqrt(max(variance, 0.0) * periods_per_year))


def risk_contributions(weights, cov):
    """Return fractional contribution of each asset to total portfolio variance."""
    w = np.asarray(weights, dtype=float)
    marginal = cov @ w
    total_variance = float(w @ marginal)
    if total_variance <= 0:
        return np.zeros_like(w)
    contrib = w * marginal
    return contrib / total_variance


def concentration_hhi(weights):
    """Herfindahl-Hirschman Index of portfolio concentration."""
    w = np.asarray(weights, dtype=float)
    return float(np.sum(np.square(w)))
