import numpy as np
from scipy.optimize import minimize


def max_weight(n):
    """Dynamic cap used by default when cap is enabled."""
    if n <= 0:
        raise ValueError("Number of assets must be positive.")
    return min(0.30, 2 / n)


def _normalize(weights):
    clipped = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    total = clipped.sum()
    if total <= 0:
        raise ValueError("Optimization produced non-positive weights.")
    return clipped / total


def _validate_cap(n, cap):
    if cap is None:
        return
    if cap <= 0 or cap > 1:
        raise ValueError("Cap must be in (0, 1].")
    if cap * n < 1.0:
        raise ValueError(
            f"Infeasible cap: cap={cap:.4f} with {n} assets cannot sum to 1."
        )


def minimum_variance(cov, cap=None):
    """Long-only minimum-variance optimization."""
    n = cov.shape[0]
    _validate_cap(n, cap)
    bounds = [(0.0, cap if cap is not None else 1.0) for _ in range(n)]
    x0 = equal_weight(n)

    def objective(w):
        w = _normalize(w)
        return float(w @ cov @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        raise ValueError(f"Minimum variance optimization failed: {result.message}")
    return _normalize(result.x)


def minimum_variance_with_turnover(
    cov,
    previous_weights,
    *,
    cap=None,
    turnover_penalty=0.0,
    no_trade_buffer=0.0,
):
    """Long-only minimum-variance optimization with turnover controls."""
    n = cov.shape[0]
    _validate_cap(n, cap)
    prev = _normalize(previous_weights)
    upper_cap = cap if cap is not None else 1.0
    bounds = [(0.0, upper_cap) for _ in range(n)]
    if no_trade_buffer > 0:
        bounds = [
            (
                max(0.0, float(prev_i) - float(no_trade_buffer)),
                min(upper_cap, float(prev_i) + float(no_trade_buffer)),
            )
            for prev_i in prev
        ]

    x0 = np.clip(prev, [b[0] for b in bounds], [b[1] for b in bounds])
    if x0.sum() <= 0:
        x0 = equal_weight(n)
    x0 = x0 / x0.sum()

    def objective(w):
        w = _normalize(w)
        base = float(w @ cov @ w)
        if turnover_penalty > 0:
            base += float(turnover_penalty) * float(np.abs(w - prev).sum())
        return base

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 3000, "ftol": 1e-12},
    )
    if not result.success:
        raise ValueError(f"Turnover-aware minimum variance optimization failed: {result.message}")
    return _normalize(result.x)


def max_diversification(cov, cap=None):
    """Maximum diversification proxy under long-only constraints."""
    n = cov.shape[0]
    _validate_cap(n, cap)
    sigma = np.sqrt(np.diag(cov))
    bounds = [(0.0, cap if cap is not None else 1.0) for _ in range(n)]
    x0 = equal_weight(n)

    def objective(w):
        w = _normalize(w)
        denom = float(np.sqrt(max(w @ cov @ w, 1e-16)))
        return float(-(sigma @ w) / denom)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        raise ValueError(f"Maximum diversification optimization failed: {result.message}")
    return _normalize(result.x)


def equal_weight(n):
    """Equal-weight baseline portfolio."""
    return np.full(n, 1.0 / n)


def erc_risk_parity(cov, cap=None, max_iter=1000):
    """Equal Risk Contribution (risk parity) with optional long-only cap."""
    n = cov.shape[0]
    _validate_cap(n, cap)

    bounds = [(0.0, cap if cap is not None else 1.0) for _ in range(n)]
    x0 = equal_weight(n)

    def objective(w):
        w = _normalize(w)
        sigma_w = cov @ w
        total_var = float(w @ sigma_w)
        if total_var <= 0:
            return 1e6
        rc_frac = (w * sigma_w) / total_var
        target = 1.0 / n
        return float(np.sum((rc_frac - target) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": max_iter, "ftol": 1e-12},
    )

    if not result.success:
        raise ValueError(f"Risk parity optimization failed: {result.message}")
    return _normalize(result.x)


def efficient_frontier(cov, exp_returns, points=31, cap=None):
    """Compute long-only efficient frontier points under optional cap."""
    n = cov.shape[0]
    _validate_cap(n, cap)
    mu = np.asarray(exp_returns, dtype=float)
    if mu.shape[0] != n:
        raise ValueError("Expected returns vector shape does not match covariance size.")
    if points < 3:
        raise ValueError("Frontier points must be at least 3.")

    bounds = [(0.0, cap if cap is not None else 1.0) for _ in range(n)]
    x0 = equal_weight(n)
    sum_constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    min_ret_result = minimize(
        lambda w: float(mu @ _normalize(w)),
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=[sum_constraint],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not min_ret_result.success:
        raise ValueError("Efficient frontier failed: unable to find minimum-return portfolio.")
    ret_min = float(mu @ _normalize(min_ret_result.x))

    max_ret_result = minimize(
        lambda w: float(-(mu @ _normalize(w))),
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=[sum_constraint],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not max_ret_result.success:
        raise ValueError("Efficient frontier failed: unable to find maximum-return portfolio.")
    ret_max = float(mu @ _normalize(max_ret_result.x))

    targets = np.linspace(ret_min, ret_max, points)
    frontier_weights = []
    frontier_returns = []

    for target in targets:
        target_constraints = [
            sum_constraint,
            {"type": "ineq", "fun": lambda w, t=float(target): float(mu @ w) - t},
        ]
        result = minimize(
            lambda w: float(_normalize(w) @ cov @ _normalize(w)),
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=target_constraints,
            options={"maxiter": 3000, "ftol": 1e-12},
        )
        if not result.success:
            continue
        weights = _normalize(result.x)
        frontier_weights.append(weights)
        frontier_returns.append(float(mu @ weights))

    if len(frontier_weights) < 2:
        raise ValueError("Efficient frontier optimization failed for most target points.")

    return frontier_weights, frontier_returns
