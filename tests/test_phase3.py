import numpy as np
import pandas as pd

from src.config import Settings
from src.covariance import (
    annualize_covariance,
    covariance_forecast_error,
    make_covariance_model,
    rolling_covariance_forecasts,
)
from src.data import returns_from_prices
from src.factor import (
    build_macro_factor_proxies,
    build_style_descriptor_proxies,
    make_factor_model,
)
from src.liquidity import (
    average_daily_volume,
    estimate_capacity,
    estimate_market_impact,
    liquidity_adjusted_var,
    liquidity_summary,
)
from src.optimize import max_weight, minimum_variance, minimum_variance_with_turnover
from src.stability import turnover_summary
from src.tail_risk import HistoricalTailRiskModel, MonteCarloTailRiskModel, rolling_tail_forecasts


def _sample_returns(rows: int = 120, cols: int = 4, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.001, 0.02, size=(rows, 1))
    noise = rng.normal(0.0, 0.01, size=(rows, cols))
    values = base + noise
    return pd.DataFrame(
        values,
        index=pd.date_range("2024-01-01", periods=rows, freq="B"),
        columns=[chr(65 + idx) for idx in range(cols)],
    )


def _sample_prices_volumes(rows: int = 120, cols: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2024-01-01", periods=rows, freq="B")
    prices = pd.DataFrame(
        {
            chr(65 + col): np.linspace(100 + 10 * col, 120 + 10 * col, rows)
            for col in range(cols)
        },
        index=idx,
    )
    volumes = pd.DataFrame(
        {
            chr(65 + col): np.full(rows, 500_000.0 + 100_000.0 * col)
            for col in range(cols)
        },
        index=idx,
    )
    return prices, volumes


def test_cap_rule():
    assert max_weight(4) == 0.30
    assert max_weight(10) == 0.20
    assert max_weight(20) == 0.10


def test_weights_sum_to_one_and_cap():
    cov = np.array([[0.04, 0.01, 0.00], [0.01, 0.03, 0.00], [0.00, 0.00, 0.02]])
    weights = minimum_variance(cov, cap=0.50)
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    assert np.all(weights >= -1e-9)


def test_turnover_aware_optimizer_respects_band():
    cov = np.array([[0.03, 0.01, 0.0], [0.01, 0.04, 0.0], [0.0, 0.0, 0.02]])
    previous = np.array([0.5, 0.3, 0.2])
    weights = minimum_variance_with_turnover(
        cov,
        previous,
        turnover_penalty=1.0,
        no_trade_buffer=0.10,
    )
    assert np.all(np.abs(weights - previous) <= 0.1001)


def test_returns_no_nan_after_preprocessing():
    idx = pd.date_range("2024-01-01", periods=400, freq="D")
    prices = pd.DataFrame(
        {"A": np.linspace(100, 140, num=400), "B": np.linspace(80, 120, num=400), "C": np.linspace(60, 100, num=400)},
        index=idx,
    )
    settings = Settings(fmp_api_key="test-key")
    rets = returns_from_prices(prices, freq="daily", settings=settings)
    assert not rets.isna().any().any()


def test_covariance_models_phase3():
    returns = _sample_returns(rows=120, cols=4)
    for method in ["ledoit_wolf", "ewma", "garch", "dcc_garch"]:
        model = make_covariance_model(method, periods_per_year=252).fit(returns)
        cov = model.get_cov()
        assert cov.shape == (4, 4)
        assert np.all(np.linalg.eigvalsh(cov) >= -1e-7)


def test_rolling_covariance_forecasts_and_error():
    returns = _sample_returns(rows=40, cols=3)
    forecasts = rolling_covariance_forecasts(
        returns,
        method="dcc_garch",
        periods_per_year=252,
        window=20,
    )
    error = covariance_forecast_error(returns, forecasts)
    assert len(forecasts) == 21
    assert forecasts.iloc[-1].shape == (3, 3)
    assert error["forecast_rmse"] >= 0
    assert annualize_covariance(forecasts.iloc[-1], 252).shape == (3, 3)


def test_historical_and_monte_carlo_var():
    returns = _sample_returns(rows=100, cols=3)
    weights = np.array([0.5, 0.3, 0.2])
    hist = HistoricalTailRiskModel(lookback=60).fit(returns, weights)
    mc = MonteCarloTailRiskModel(simulation_count=2000, random_seed=7).fit(returns, weights)
    assert hist.es(0.95) >= hist.var(0.95)
    assert mc.es(0.95) >= mc.var(0.95)


def test_rolling_tail_backtest_has_exception_flag():
    returns = _sample_returns(rows=80, cols=3)
    weights = np.array([0.4, 0.4, 0.2])
    backtest = rolling_tail_forecasts(
        returns,
        weights=weights,
        method="monte_carlo",
        confidence=0.95,
        window=30,
        simulation_count=1000,
        random_seed=11,
    )
    assert {"var", "es", "realized_loss", "exception"} <= set(backtest.columns)
    assert backtest["exception"].isin([0.0, 1.0]).all()


def test_liquidity_metrics_impact_capacity_and_lvar():
    prices, volumes = _sample_prices_volumes(rows=30, cols=2)
    adv = average_daily_volume(prices, volumes, lookback=20)
    weights = np.array([0.6, 0.4])
    metrics = liquidity_summary(
        weights,
        ["A", "B"],
        latest_prices=prices.iloc[-1].to_dict(),
        adv_frame=adv,
        cash_amount=1_000_000,
        participation_rate=0.1,
    )
    impact = estimate_market_impact(weights, ["A", "B"], adv, cash_amount=1_000_000, participation_rate=0.1)
    capacity = estimate_capacity(weights, ["A", "B"], adv, max_participation_rate=0.1, liquidation_days=5)
    assert len(metrics) == 2
    assert impact[0].total_cost >= 0
    assert capacity["portfolio_capacity"] > 0
    assert liquidity_adjusted_var(0.02, impact, mode="screening") >= 0.02


def test_turnover_summary():
    weight_history = pd.DataFrame(
        [[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.45, 0.35, 0.2]],
        index=pd.date_range("2024-01-01", periods=3, freq="W"),
        columns=["A", "B", "C"],
    )
    summary = turnover_summary(weight_history, periods_per_year=52)
    assert summary["annualized_turnover"] > 0


def test_pca_macro_and_style_factor_models():
    returns = _sample_returns(rows=100, cols=4)
    prices, volumes = _sample_prices_volumes(rows=100, cols=4)

    pca = make_factor_model("pca", n_factors=2)
    pca.fit(returns)
    assert pca.exposures().shape == (4, 2)

    macro_factors = build_macro_factor_proxies(returns, volumes)
    macro = make_factor_model("macro")
    macro.fit(returns, factor_data=macro_factors, volumes=volumes)
    assert macro.exposures().shape[0] == 4

    style_desc = build_style_descriptor_proxies(prices, returns, volumes)
    style = make_factor_model("style")
    style.fit(returns, prices=prices, volumes=volumes)
    assert style.exposures().shape[1] == style_desc.shape[1]
    decomp = style.factor_risk_decomposition(np.full(4, 0.25))
    assert np.isclose(decomp["factor_variance_share"] + decomp["specific_variance_share"], 1.0)
