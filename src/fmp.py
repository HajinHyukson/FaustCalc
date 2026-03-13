"""Backward-compatible wrappers around the new settings/client modules."""

from .config import get_settings
from .errors import APIError as FMPError
from .errors import ConfigError as FMPConfigError
from .errors import DataError as FMPDataError
from .errors import UnauthorizedError as FMPAuthError
from .fmp_client import FMPClient


def get_api_key(required: bool = True) -> str | None:
    if not required:
        try:
            return get_settings().fmp_api_key
        except FMPConfigError:
            return None
    return get_settings().fmp_api_key


def fetch_adj_close(ticker: str, years: int = 3):
    settings = get_settings()
    client = FMPClient(settings)
    return client.get_price_series(ticker, years=years)
