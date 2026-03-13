import cvxpy as cp
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

    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cov))

    constraints = [
        w >= 0,
        cp.sum(w) == 1,
    ]
    if cap is not None:
        constraints.append(w <= cap)

    problem = cp.Problem(objective, constraints)
    problem.solve(warm_start=True)

    if w.value is None:
        raise ValueError("Minimum variance optimization failed.")
    return _normalize(w.value)


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
    w = cp.Variable(n)
    objective = cp.quad_form(w, cov)
    if turnover_penalty > 0:
        objective += float(turnover_penalty) * cp.norm1(w - prev)
    constraints = [w >= 0, cp.sum(w) == 1]
    if cap is not None:
        constraints.append(w <= cap)
    if no_trade_buffer > 0:
        constraints.append(cp.abs(w - prev) <= no_trade_buffer)

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(warm_start=True)
    if w.value is None:
        raise ValueError("Turnover-aware minimum variance optimization failed.")
    return _normalize(w.value)


def max_diversification(cov, cap=None):
    """Maximum diversification proxy under long-only constraints."""

    n = cov.shape[0]
    _validate_cap(n, cap)

    sigma = np.sqrt(np.diag(cov))
    w = cp.Variable(n)
    objective = cp.Maximize(sigma @ w)

    constraints = [
        w >= 0,
        cp.quad_form(w, cov) <= 1
    ]
    if cap is not None:
        constraints.append(w <= cap)

    problem = cp.Problem(objective, constraints)
    problem.solve(warm_start=True)

    if w.value is None:
        raise ValueError("Maximum diversification optimization failed.")
    return _normalize(w.value)


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

    w = cp.Variable(n)
    constraints = [w >= 0, cp.sum(w) == 1]
    if cap is not None:
        constraints.append(w <= cap)

    min_ret_prob = cp.Problem(cp.Minimize(mu @ w), constraints)
    min_ret_prob.solve(warm_start=True)
    if w.value is None:
        raise ValueError("Efficient frontier failed: unable to find minimum-return portfolio.")
    ret_min = float(mu @ w.value)

    max_ret_prob = cp.Problem(cp.Maximize(mu @ w), constraints)
    max_ret_prob.solve(warm_start=True)
    if w.value is None:
        raise ValueError("Efficient frontier failed: unable to find maximum-return portfolio.")
    ret_max = float(mu @ w.value)

    targets = np.linspace(ret_min, ret_max, points)
    frontier_weights = []
    frontier_returns = []

    for target in targets:
        wf = cp.Variable(n)
        cons = [wf >= 0, cp.sum(wf) == 1, mu @ wf >= target]
        if cap is not None:
            cons.append(wf <= cap)
        prob = cp.Problem(cp.Minimize(cp.quad_form(wf, cov)), cons)
        prob.solve(warm_start=True)
        if wf.value is None:
            continue
        weights = _normalize(wf.value)
        frontier_weights.append(weights)
        frontier_returns.append(float(mu @ weights))

    if len(frontier_weights) < 2:
        raise ValueError("Efficient frontier optimization failed for most target points.")

    return frontier_weights, frontier_returns
