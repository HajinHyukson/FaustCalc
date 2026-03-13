import os
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigError

_DOTENV_LOADED = False


@dataclass(frozen=True)
class Settings:
    fmp_api_key: str
    fmp_base_url: str = "https://financialmodelingprep.com/stable"
    request_timeout_seconds: int = 30
    default_years: int = 3
    default_frequency: str = "weekly"
    cache_dir: Path = Path(".cache")
    cache_ttl_hours: int = 24
    min_assets: int = 2
    max_assets: int = 1000
    min_weeks: int = 78
    min_days: int = 252
    default_covariance_method: str = "ledoit_wolf"
    default_ewma_decay: float = 0.94
    default_ewma_mode: str = "expanding"
    default_tail_risk_method: str = "historical"
    default_tail_confidence: float = 0.95
    default_tail_lookback: int = 60
    default_mc_simulations: int = 10000
    default_mc_seed: int = 42
    default_liquidity_lookback: int = 20
    default_participation_rate: float = 0.10
    default_market_impact_temporary: float = 0.10
    default_market_impact_permanent: float = 0.05
    default_capacity_days: float = 5.0
    default_factor_model: str = "pca"
    default_pca_factors: int = 3
    default_liquidity_var_mode: str = "detailed"
    default_turnover_penalty: float = 0.0
    default_no_trade_buffer: float = 0.0


def _load_dotenv_if_available() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return

    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        _DOTENV_LOADED = True
        return

    load_dotenv()
    _DOTENV_LOADED = True


def get_settings() -> Settings:
    _load_dotenv_if_available()
    api_key = (os.getenv("FMP_API_KEY") or "").strip()
    if not api_key:
        raise ConfigError(
            "Missing FMP_API_KEY. Set it in your environment or add "
            "`FMP_API_KEY=your_key_here` to a `.env` file."
        )
    return Settings(fmp_api_key=api_key)
