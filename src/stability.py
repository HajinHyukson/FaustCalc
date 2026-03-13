from __future__ import annotations

import numpy as np
import pandas as pd


def one_period_turnover(previous_weights: np.ndarray, current_weights: np.ndarray) -> float:
    prev = np.asarray(previous_weights, dtype=float)
    curr = np.asarray(current_weights, dtype=float)
    if prev.shape != curr.shape:
        raise ValueError("Weight vectors must share the same shape.")
    return float(0.5 * np.abs(curr - prev).sum())


def annualized_turnover(weight_history: pd.DataFrame, periods_per_year: int) -> float:
    if weight_history.shape[0] < 2:
        return 0.0
    turns = [
        one_period_turnover(weight_history.iloc[idx - 1].values, weight_history.iloc[idx].values)
        for idx in range(1, len(weight_history))
    ]
    return float(np.mean(turns) * periods_per_year)


def average_absolute_weight_change(weight_history: pd.DataFrame) -> float:
    if weight_history.shape[0] < 2:
        return 0.0
    changes = weight_history.diff().abs().dropna(how="any")
    if changes.empty:
        return 0.0
    return float(changes.mean(axis=1).mean())


def turnover_summary(weight_history: pd.DataFrame, periods_per_year: int) -> dict[str, float]:
    if weight_history.empty:
        raise ValueError("Weight history is empty.")
    latest_turnover = (
        one_period_turnover(weight_history.iloc[-2].values, weight_history.iloc[-1].values)
        if weight_history.shape[0] >= 2
        else 0.0
    )
    return {
        "one_period_turnover": latest_turnover,
        "annualized_turnover": annualized_turnover(weight_history, periods_per_year),
        "average_absolute_weight_change": average_absolute_weight_change(weight_history),
    }
