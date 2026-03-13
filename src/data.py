from collections.abc import Iterable

import pandas as pd

from .config import Settings
from .errors import DataError, NotFoundError
from .fmp_client import FMPClient


def normalize_tickers(raw: str | Iterable[str]) -> list[str]:
    """Normalize and de-duplicate tickers while preserving order."""
    if isinstance(raw, str):
        candidates = raw.split(",")
    else:
        candidates = list(raw)

    seen: set[str] = set()
    tickers: list[str] = []
    for token in candidates:
        ticker = str(token).strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        tickers.append(ticker)
    return tickers


def parse_tickers(raw: str | Iterable[str], settings: Settings) -> list[str]:
    """Parse, normalize, de-duplicate tickers while preserving order."""
    tickers = normalize_tickers(raw)
    n_assets = len(tickers)
    if not (settings.min_assets <= n_assets <= settings.max_assets):
        raise DataError(
            f"Provide between {settings.min_assets} and {settings.max_assets} unique tickers."
        )
    return tickers


def build_price_matrix(
    tickers: list[str],
    client: FMPClient,
    years: int,
    *,
    missing_policy: str = "dropna",
) -> pd.DataFrame:
    """Download and align close price histories across tickers."""
    prices, _ = build_price_volume_matrices(
        tickers,
        client,
        years,
        missing_policy=missing_policy,
    )
    return prices


def build_price_volume_matrices(
    tickers: list[str],
    client: FMPClient,
    years: int,
    *,
    missing_policy: str = "dropna",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download and align close price and volume histories across tickers."""
    if missing_policy != "dropna":
        raise DataError("Unsupported missing policy. Use '--dropna'.")

    price_series: list[pd.Series] = []
    volume_series: list[pd.Series] = []
    for ticker in tickers:
        try:
            history = client.get_eod_history(ticker, years=years)
        except NotFoundError as exc:
            raise NotFoundError(
                f"No historical data returned for {ticker} (check symbol/plan)."
            ) from exc

        price_col = "adjClose" if "adjClose" in history.columns else "close"
        prices = pd.to_numeric(history[price_col], errors="coerce").dropna()
        if prices.shape[0] < 2:
            raise DataError(
                f"Ticker {ticker} has too little history for return estimation. "
                "Try a longer window or a different ticker."
            )
        prices.name = ticker
        price_series.append(prices)

        if "volume" in history.columns:
            volumes = pd.to_numeric(history["volume"], errors="coerce")
        else:
            volumes = pd.Series(index=history.index, data=float("nan"))
        volumes.name = ticker
        volume_series.append(volumes)

    prices = pd.concat(price_series, axis=1).sort_index()
    volumes = pd.concat(volume_series, axis=1).sort_index()
    if missing_policy == "dropna":
        prices = prices.dropna(how="any")
        volumes = volumes.reindex(prices.index)

    if prices.empty:
        raise DataError(
            "Not enough overlapping history after alignment; try fewer tickers or more years."
        )
    return prices, volumes


def returns_from_prices(
    prices: pd.DataFrame,
    freq: str,
    settings: Settings,
) -> pd.DataFrame:
    """Compute returns after optional resampling and strict quality gates."""
    if prices.empty:
        raise DataError("Price matrix is empty.")

    freq_normalized = freq.strip().lower()
    if freq_normalized == "weekly":
        sampled = prices.resample("W-FRI").last()
        min_obs = settings.min_weeks
    elif freq_normalized == "daily":
        sampled = prices.copy()
        min_obs = settings.min_days
    else:
        raise DataError("Unsupported frequency. Use 'weekly' or 'daily'.")

    sampled = sampled.dropna(how="any")
    returns = sampled.pct_change().dropna(how="any")

    if returns.shape[0] < min_obs:
        counts = sampled.count().sort_values()
        shortest = counts.index[0] if not counts.empty else "unknown"
        raise DataError(
            "Not enough overlapping history after alignment; try fewer tickers or more years. "
            f"Shortest series ticker: {shortest}. Required observations: {min_obs}, "
            f"available: {returns.shape[0]}."
        )
    return returns
