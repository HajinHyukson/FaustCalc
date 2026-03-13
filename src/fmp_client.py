import logging
import time
from datetime import date, timedelta
from typing import Any, Optional

import pandas as pd
import requests

from .cache import FileCache
from .config import Settings
from .errors import APIError, NotFoundError, RateLimitError, UnauthorizedError


class FMPClient:
    """FMP Stable API client with retries and optional on-disk caching."""

    def __init__(
        self,
        settings: Settings,
        *,
        cache_enabled: bool = True,
        cache_ttl_hours: Optional[float] = None,
        session: Optional[requests.Session] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.settings = settings
        self.session = session or requests.Session()
        self.cache_enabled = cache_enabled
        self.cache_ttl_hours = (
            float(cache_ttl_hours)
            if cache_ttl_hours is not None
            else float(settings.cache_ttl_hours)
        )
        self.cache = FileCache(settings.cache_dir) if cache_enabled else None
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def normalize_ticker(ticker: str) -> str:
        return ticker.strip().upper()

    def get_price_series(self, ticker: str, years: int) -> pd.Series:
        history = self.get_eod_history(ticker, years=years)
        price_col = "adjClose" if "adjClose" in history.columns else "close"
        series = pd.to_numeric(history[price_col], errors="coerce").dropna()
        if series.empty:
            raise NotFoundError(
                f"No historical data returned for {ticker} (check symbol/plan)."
            )
        series.name = self.normalize_ticker(ticker)
        return series

    def get_eod_history(self, ticker: str, years: int) -> pd.DataFrame:
        symbol = self.normalize_ticker(ticker)
        if not symbol:
            raise NotFoundError("Ticker symbol is empty after normalization.")
        if years < 1:
            raise APIError("Years must be at least 1.")

        from_date = (date.today() - timedelta(days=365 * years)).isoformat()
        endpoint = "/historical-price-eod/full"
        params = {"symbol": symbol, "from": from_date}
        data = self._request_json(endpoint, params)

        if not isinstance(data, list) or not data:
            raise NotFoundError(
                f"No historical data returned for {symbol} (check symbol/plan)."
            )

        df = pd.DataFrame(data)
        if "date" not in df.columns:
            raise APIError(f"Malformed response for {symbol}: missing 'date' field.")
        price_col = "adjClose" if "adjClose" in df.columns else "close"
        if price_col not in df.columns:
            raise APIError(
                f"Malformed response for {symbol}: missing 'adjClose'/'close' field."
            )

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        cleaned = df.dropna(subset=[price_col])
        if cleaned.empty:
            raise NotFoundError(
                f"No historical data returned for {symbol} (check symbol/plan)."
            )
        return cleaned

    def _cache_key(self, endpoint: str, params: dict[str, Any]) -> str:
        ordered = "&".join(f"{k}={params[k]}" for k in sorted(params))
        return f"{endpoint}?{ordered}"

    def _request_json(self, endpoint: str, params: dict[str, Any]) -> Any:
        url = f"{self.settings.fmp_base_url}{endpoint}"
        params_with_key = {**params, "apikey": self.settings.fmp_api_key}
        cache_key = self._cache_key(endpoint, params)

        if self.cache is not None:
            cached = self.cache.get(cache_key, ttl_hours=self.cache_ttl_hours)
            if cached is not None:
                self.logger.debug("Cache hit for %s", cache_key)
                return cached

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = self.session.get(
                    url, params=params_with_key, timeout=self.settings.request_timeout_seconds
                )
            except requests.RequestException as exc:
                if attempt == max_attempts - 1:
                    raise APIError(f"Network error while requesting {url}: {exc}") from exc
                time.sleep(2**attempt)
                continue

            if response.status_code in (401, 403):
                raise UnauthorizedError(
                    "Unauthorized: API key missing/invalid/inactive or endpoint not in plan."
                )
            if response.status_code == 404:
                raise NotFoundError("No historical data returned for request (check symbol/plan).")
            if response.status_code == 429:
                if attempt == max_attempts - 1:
                    raise RateLimitError(
                        "Rate limit reached (429). Wait and retry, or reduce request frequency."
                    )
                time.sleep(2**attempt)
                continue
            if response.status_code >= 500:
                if attempt == max_attempts - 1:
                    raise APIError(
                        f"FMP server error ({response.status_code}). Please retry shortly."
                    )
                time.sleep(2**attempt)
                continue
            if response.status_code >= 400:
                raise APIError(
                    f"FMP request failed with status {response.status_code}: {response.text[:200]}"
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise APIError("FMP returned non-JSON response.") from exc

            if self.cache is not None:
                self.cache.set(cache_key, payload)
            return payload

        raise APIError("Unexpected request failure state.")
