from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass
class LiquidityMetric:
    ticker: str
    adv_shares: float
    adv_dollars: float
    participation_rate: float
    days_to_liquidate: float
    liquidation_value: float
    stale: bool


@dataclass
class ImpactEstimate:
    ticker: str
    trade_value: float
    participation_rate: float
    temporary_cost: float
    permanent_cost: float
    total_cost: float


def average_daily_volume(prices: pd.DataFrame, volumes: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    if prices.empty or volumes.empty:
        raise ValueError("Prices and volumes are required for liquidity metrics.")
    aligned_prices, aligned_volumes = prices.align(volumes, join="inner", axis=0)
    dollar_volume = aligned_prices * aligned_volumes
    return pd.DataFrame(
        {
            "adv_shares": aligned_volumes.tail(lookback).mean(axis=0),
            "adv_dollars": dollar_volume.tail(lookback).mean(axis=0),
            "stale_days": aligned_volumes.tail(lookback).isna().sum(axis=0),
        }
    )


def liquidity_summary(
    weights: np.ndarray,
    tickers: list[str],
    latest_prices: Mapping[str, float],
    adv_frame: pd.DataFrame,
    *,
    cash_amount: float,
    participation_rate: float = 0.1,
) -> list[LiquidityMetric]:
    if not 0.0 < participation_rate <= 1.0:
        raise ValueError("Participation rate must be in (0, 1].")

    metrics: list[LiquidityMetric] = []
    for ticker, weight in zip(tickers, np.asarray(weights, dtype=float)):
        price = float(latest_prices[ticker])
        position_value = float(cash_amount) * float(weight)
        adv_shares = float(adv_frame.loc[ticker, "adv_shares"])
        adv_dollars = float(adv_frame.loc[ticker, "adv_dollars"])
        daily_capacity = max(adv_dollars * participation_rate, 1e-12)
        days_to_liquidate = position_value / daily_capacity
        metrics.append(
            LiquidityMetric(
                ticker=ticker,
                adv_shares=adv_shares,
                adv_dollars=adv_dollars,
                participation_rate=participation_rate,
                days_to_liquidate=float(days_to_liquidate),
                liquidation_value=position_value,
                stale=bool(adv_frame.loc[ticker, "stale_days"] > 0 or not np.isfinite(price)),
            )
        )
    return sorted(metrics, key=lambda item: (-item.days_to_liquidate, item.ticker))


def estimate_market_impact(
    weights: np.ndarray,
    tickers: list[str],
    adv_frame: pd.DataFrame,
    *,
    cash_amount: float,
    participation_rate: float = 0.1,
    temporary_coefficient: float = 0.10,
    permanent_coefficient: float = 0.05,
) -> list[ImpactEstimate]:
    estimates: list[ImpactEstimate] = []
    for ticker, weight in zip(tickers, np.asarray(weights, dtype=float)):
        trade_value = float(cash_amount) * abs(float(weight))
        adv_dollars = max(float(adv_frame.loc[ticker, "adv_dollars"]), 1e-12)
        participation = max(trade_value / adv_dollars, participation_rate)
        temporary_cost = trade_value * temporary_coefficient * np.sqrt(participation)
        permanent_cost = trade_value * permanent_coefficient * participation
        estimates.append(
            ImpactEstimate(
                ticker=ticker,
                trade_value=trade_value,
                participation_rate=float(participation),
                temporary_cost=float(temporary_cost),
                permanent_cost=float(permanent_cost),
                total_cost=float(temporary_cost + permanent_cost),
            )
        )
    return sorted(estimates, key=lambda item: (-item.total_cost, item.ticker))


def estimate_capacity(
    weights: np.ndarray,
    tickers: list[str],
    adv_frame: pd.DataFrame,
    *,
    max_participation_rate: float = 0.1,
    liquidation_days: float = 5.0,
) -> dict[str, float | str]:
    capacities: list[tuple[str, float]] = []
    for ticker, weight in zip(tickers, np.asarray(weights, dtype=float)):
        if weight <= 0:
            continue
        adv_dollars = max(float(adv_frame.loc[ticker, "adv_dollars"]), 1e-12)
        asset_capacity = (adv_dollars * max_participation_rate * liquidation_days) / float(weight)
        capacities.append((ticker, float(asset_capacity)))
    if not capacities:
        return {"portfolio_capacity": 0.0, "bottleneck_ticker": "N/A"}
    bottleneck = min(capacities, key=lambda item: item[1])
    return {
        "portfolio_capacity": float(bottleneck[1]),
        "bottleneck_ticker": bottleneck[0],
    }


def liquidity_adjusted_var(
    base_var: float,
    impact_estimates: list[ImpactEstimate],
    *,
    mode: str = "detailed",
) -> float:
    if mode not in {"screening", "detailed"}:
        raise ValueError("Liquidity-adjusted VaR mode must be 'screening' or 'detailed'.")
    if not impact_estimates:
        return float(base_var)
    if mode == "screening":
        liquidity_cost = float(sum(item.temporary_cost for item in impact_estimates))
    else:
        liquidity_cost = float(sum(item.total_cost for item in impact_estimates))
    return float(base_var + liquidity_cost)
