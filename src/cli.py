import logging
import traceback
from typing import Optional

import numpy as np
import pandas as pd
import typer

from .charting import save_efficient_frontier_chart
from .covariance import covariance_forecast_error, make_covariance_model, rolling_covariance_forecasts
from .config import get_settings
from .data import build_price_volume_matrices, normalize_tickers, returns_from_prices
from .errors import (
    APIError,
    ConfigError,
    DataError,
    NotFoundError,
    RateLimitError,
    UnauthorizedError,
)
from .factor import build_macro_factor_proxies, make_factor_model
from .fmp_client import FMPClient
from .liquidity import (
    average_daily_volume,
    estimate_capacity,
    estimate_market_impact,
    liquidity_adjusted_var,
    liquidity_summary,
)
from .optimize import (
    efficient_frontier,
    erc_risk_parity,
    equal_weight,
    minimum_variance,
    minimum_variance_with_turnover,
)
from .reporting import (
    format_comparison_table,
    format_factor_exposures,
    format_impact_table,
    format_key_value_block,
    format_liquidity_table,
    format_matrix,
    format_portfolio_allocation_table,
    format_run_header,
    format_tail_backtest_table,
)
from .risk import portfolio_vol, risk_contributions
from .stability import turnover_summary
from .tail_risk import make_tail_risk_model, rolling_tail_forecasts


def _configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise typer.BadParameter("`--log-level` must be a valid logging level (e.g., INFO, DEBUG).")
    logging.basicConfig(level=numeric_level, format="%(asctime)s %(levelname)s %(message)s")


def _disregard_inferior_optional_tickers(
    returns: pd.DataFrame,
    periods: int,
    required_tickers: list[str],
) -> tuple[list[str], list[str]]:
    """Disregard optional assets dominated on return/risk dimensions."""
    annual_ret = returns.mean() * periods
    annual_vol = returns.std(ddof=1) * np.sqrt(periods)
    assets = list(returns.columns)
    required_set = set(required_tickers)

    kept: list[str] = []
    disregarded: list[str] = []
    for asset_i in assets:
        if asset_i in required_set:
            kept.append(asset_i)
            continue
        dominated = False
        for asset_j in assets:
            if asset_i == asset_j:
                continue
            better_or_equal_ret = annual_ret[asset_j] >= annual_ret[asset_i]
            lower_or_equal_risk = annual_vol[asset_j] <= annual_vol[asset_i]
            strictly_better_one_side = (
                annual_ret[asset_j] > annual_ret[asset_i]
                or annual_vol[asset_j] < annual_vol[asset_i]
            )
            if better_or_equal_ret and lower_or_equal_risk and strictly_better_one_side:
                dominated = True
                break
        if dominated:
            disregarded.append(asset_i)
        else:
            kept.append(asset_i)

    if not kept:
        return assets, []
    return kept, sorted(disregarded)


def _rolling_min_var_weights(
    returns: pd.DataFrame,
    *,
    method: str,
    periods: int,
    window: int,
    ewma_decay: float,
    ewma_mode: str,
) -> pd.DataFrame:
    records: list[dict[str, float | pd.Timestamp]] = []
    for end_idx in range(window, len(returns) + 1):
        sample = returns.iloc[end_idx - window:end_idx]
        cov = make_covariance_model(
            method,
            periods_per_year=periods,
            ewma_decay=ewma_decay,
            ewma_mode=ewma_mode,
        ).fit(sample).get_cov()
        weights = minimum_variance(cov)
        record = {"date": returns.index[end_idx - 1]}
        record.update({ticker: float(weight) for ticker, weight in zip(returns.columns, weights)})
        records.append(record)
    return pd.DataFrame.from_records(records).set_index("date")


def main(
    required_tickers: str = typer.Option(
        "",
        "--required-tickers",
        help="Comma-separated tickers that must always remain in the portfolio.",
    ),
    optional_tickers: str = typer.Option(
        "",
        "--optional-tickers",
        help="Comma-separated tickers that can be disregarded if inferior.",
    ),
    years: int = typer.Option(3, "--years", "-y", help="Years of history to pull"),
    freq: str = typer.Option("weekly", "--freq", help="Return frequency: weekly or daily"),
    cash: float = typer.Option(
        ...,
        "--cash",
        help="Total portfolio cash amount in USD.",
    ),
    cache: bool = typer.Option(True, "--cache", help="Enable on-disk response cache."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable on-disk response cache."),
    cache_ttl_hours: Optional[float] = typer.Option(
        None, "--cache-ttl-hours", help="Override cache TTL in hours."
    ),
    plot_frontier: bool = typer.Option(False, "--plot-frontier", help="Save efficient frontier chart (PNG)."),
    no_plot_frontier: bool = typer.Option(False, "--no-plot-frontier", help="Disable frontier chart output."),
    plot_path: str = typer.Option(
        "outputs/efficient_frontier.png",
        "--plot-path",
        help="Output path for efficient frontier chart PNG.",
    ),
    frontier_points: int = typer.Option(
        31,
        "--frontier-points",
        help="Number of target-return points for frontier chart.",
    ),
    covariance_method: str = typer.Option(
        None,
        "--covariance-method",
        help="Covariance model: ledoit_wolf, ewma, or garch.",
    ),
    ewma_decay: float = typer.Option(
        None,
        "--ewma-decay",
        help="EWMA decay parameter, typically 0.94 for daily data.",
    ),
    ewma_mode: str = typer.Option(
        None,
        "--ewma-mode",
        help="EWMA estimation mode: expanding or rolling.",
    ),
    tail_risk_method: str = typer.Option(
        None,
        "--tail-risk-method",
        help="Tail-risk method: historical or monte_carlo.",
    ),
    tail_confidence: float = typer.Option(
        None,
        "--tail-confidence",
        help="Tail-risk confidence level, e.g. 0.95.",
    ),
    tail_lookback: int = typer.Option(
        None,
        "--tail-lookback",
        help="Historical tail-risk lookback window in return periods.",
    ),
    participation_rate: float = typer.Option(
        None,
        "--participation-rate",
        help="Participation rate for ADV-based liquidity metrics.",
    ),
    liquidity_lookback: int = typer.Option(
        None,
        "--liquidity-lookback",
        help="Lookback window for ADV estimation.",
    ),
    mc_simulations: int = typer.Option(
        None,
        "--mc-simulations",
        help="Monte Carlo simulation count for tail-risk forecasts.",
    ),
    mc_seed: int = typer.Option(
        None,
        "--mc-seed",
        help="Random seed for Monte Carlo tail-risk forecasts.",
    ),
    factor_model: str = typer.Option(
        None,
        "--factor-model",
        help="Factor model: pca, macro, or style.",
    ),
    pca_factors: int = typer.Option(
        None,
        "--pca-factors",
        help="Number of PCA factors.",
    ),
    impact_temporary: float = typer.Option(
        None,
        "--impact-temporary",
        help="Temporary impact coefficient.",
    ),
    impact_permanent: float = typer.Option(
        None,
        "--impact-permanent",
        help="Permanent impact coefficient.",
    ),
    capacity_days: float = typer.Option(
        None,
        "--capacity-days",
        help="Liquidation window in days for capacity estimates.",
    ),
    liquidity_var_mode: str = typer.Option(
        None,
        "--liquidity-var-mode",
        help="Liquidity-adjusted VaR mode: screening or detailed.",
    ),
    turnover_penalty: float = typer.Option(
        None,
        "--turnover-penalty",
        help="L1 turnover penalty for minimum-variance optimization.",
    ),
    no_trade_buffer: float = typer.Option(
        None,
        "--no-trade-buffer",
        help="Absolute no-trade band around previous weights.",
    ),
    dropna: bool = typer.Option(True, "--dropna", help="Drop rows with any missing values."),
    no_dropna: bool = typer.Option(False, "--no-dropna", help="Use pairwise alignment for missing values."),
    show_risk_contrib: bool = typer.Option(
        False, "--show-risk-contrib", help="Print risk contribution by asset."
    ),
    debug: bool = typer.Option(False, "--debug", help="Show stack traces and debug details."),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level: INFO or DEBUG."),
):
    """Risk-first portfolio construction from user-selected tickers."""
    try:
        if no_cache:
            cache = False
        if no_plot_frontier:
            plot_frontier = False
        if no_dropna:
            dropna = False
        _configure_logging(log_level)
        settings = get_settings()

        if years < 1:
            raise typer.BadParameter("`--years` must be at least 1.")
        freq_normalized = freq.strip().lower()
        if freq_normalized not in {"weekly", "daily"}:
            raise typer.BadParameter("`--freq` must be either 'weekly' or 'daily'.")
        if cash <= 0:
            raise typer.BadParameter("`--cash` must be greater than 0.")
        if frontier_points < 3:
            raise typer.BadParameter("`--frontier-points` must be at least 3.")
        covariance_method = covariance_method or settings.default_covariance_method
        ewma_decay = float(settings.default_ewma_decay if ewma_decay is None else ewma_decay)
        ewma_mode = ewma_mode or settings.default_ewma_mode
        tail_risk_method = tail_risk_method or settings.default_tail_risk_method
        tail_confidence = float(
            settings.default_tail_confidence if tail_confidence is None else tail_confidence
        )
        tail_lookback = int(settings.default_tail_lookback if tail_lookback is None else tail_lookback)
        mc_simulations = int(settings.default_mc_simulations if mc_simulations is None else mc_simulations)
        mc_seed = int(settings.default_mc_seed if mc_seed is None else mc_seed)
        participation_rate = float(
            settings.default_participation_rate if participation_rate is None else participation_rate
        )
        liquidity_lookback = int(
            settings.default_liquidity_lookback if liquidity_lookback is None else liquidity_lookback
        )
        factor_model = factor_model or settings.default_factor_model
        pca_factors = int(settings.default_pca_factors if pca_factors is None else pca_factors)
        impact_temporary = float(
            settings.default_market_impact_temporary if impact_temporary is None else impact_temporary
        )
        impact_permanent = float(
            settings.default_market_impact_permanent if impact_permanent is None else impact_permanent
        )
        capacity_days = float(settings.default_capacity_days if capacity_days is None else capacity_days)
        liquidity_var_mode = liquidity_var_mode or settings.default_liquidity_var_mode
        turnover_penalty = float(
            settings.default_turnover_penalty if turnover_penalty is None else turnover_penalty
        )
        no_trade_buffer = float(settings.default_no_trade_buffer if no_trade_buffer is None else no_trade_buffer)
        if covariance_method not in {"ledoit_wolf", "ewma", "garch", "dcc_garch"}:
            raise typer.BadParameter(
                "`--covariance-method` must be 'ledoit_wolf', 'ewma', 'garch', or 'dcc_garch'."
            )
        if ewma_mode not in {"expanding", "rolling"}:
            raise typer.BadParameter("`--ewma-mode` must be 'expanding' or 'rolling'.")
        if tail_risk_method not in {"historical", "monte_carlo"}:
            raise typer.BadParameter("`--tail-risk-method` must be 'historical' or 'monte_carlo'.")
        if not 0.0 < tail_confidence < 1.0:
            raise typer.BadParameter("`--tail-confidence` must be in (0, 1).")
        if tail_lookback < 10:
            raise typer.BadParameter("`--tail-lookback` must be at least 10.")
        if mc_simulations < 100:
            raise typer.BadParameter("`--mc-simulations` must be at least 100.")
        if not 0.0 < participation_rate <= 1.0:
            raise typer.BadParameter("`--participation-rate` must be in (0, 1].")
        if liquidity_lookback < 2:
            raise typer.BadParameter("`--liquidity-lookback` must be at least 2.")
        if factor_model not in {"pca", "macro", "style"}:
            raise typer.BadParameter("`--factor-model` must be 'pca', 'macro', or 'style'.")
        if pca_factors < 1:
            raise typer.BadParameter("`--pca-factors` must be at least 1.")
        if impact_temporary < 0 or impact_permanent < 0:
            raise typer.BadParameter("Impact coefficients must be non-negative.")
        if capacity_days <= 0:
            raise typer.BadParameter("`--capacity-days` must be positive.")
        if liquidity_var_mode not in {"screening", "detailed"}:
            raise typer.BadParameter("`--liquidity-var-mode` must be 'screening' or 'detailed'.")
        if turnover_penalty < 0 or no_trade_buffer < 0:
            raise typer.BadParameter("Turnover controls must be non-negative.")

        required_list = normalize_tickers(required_tickers)
        optional_list = [t for t in normalize_tickers(optional_tickers) if t not in set(required_list)]
        ticker_list = [*required_list, *optional_list]

        if not (settings.min_assets <= len(ticker_list) <= settings.max_assets):
            raise typer.BadParameter(
                f"Provide between {settings.min_assets} and {settings.max_assets} unique tickers "
                "across required and optional groups."
            )

        effective_cap = None
        cap_policy = "disabled (no max per-asset cap)"

        client = FMPClient(
            settings,
            cache_enabled=cache,
            cache_ttl_hours=cache_ttl_hours,
            logger=logging.getLogger(__name__),
        )

        typer.echo("Downloading and preparing price data...")
        missing_policy = "dropna" if dropna else "pairwise"
        prices, volumes = build_price_volume_matrices(
            ticker_list,
            client,
            years=years,
            missing_policy=missing_policy,
        )
        returns = returns_from_prices(prices, freq=freq_normalized, settings=settings)

        periods = 52 if freq_normalized == "weekly" else 252
        kept_tickers, disregarded_tickers = _disregard_inferior_optional_tickers(
            returns,
            periods,
            required_list,
        )
        filtered_prices = prices[kept_tickers]
        filtered_volumes = volumes[kept_tickers].reindex(filtered_prices.index)
        filtered_returns = returns[kept_tickers]

        covariance_model = make_covariance_model(
            covariance_method,
            periods_per_year=periods,
            ewma_decay=ewma_decay,
            ewma_mode=ewma_mode,
        ).fit(filtered_returns)
        cov = covariance_model.get_cov()
        latest_prices = filtered_prices.iloc[-1].to_dict()
        expected_returns = filtered_returns.mean().values * periods
        baseline_weights = equal_weight(len(kept_tickers))
        portfolios = {
            "Minimum Variance": minimum_variance(cov, cap=effective_cap),
            "Risk Parity (ERC)": erc_risk_parity(cov, cap=effective_cap),
        }
        portfolios["Min Variance (Turnover-Aware)"] = minimum_variance_with_turnover(
            cov,
            baseline_weights,
            cap=effective_cap,
            turnover_penalty=turnover_penalty,
            no_trade_buffer=no_trade_buffer,
        )
        tail_model = make_tail_risk_model(
            tail_risk_method,
            lookback=tail_lookback,
            include_mean=False,
            covariance_method=covariance_method,
            simulation_count=mc_simulations,
            random_seed=mc_seed,
            ewma_decay=ewma_decay,
            ewma_mode=ewma_mode,
        )
        tail_model.fit(filtered_returns, portfolios["Minimum Variance"])
        tail_backtest = rolling_tail_forecasts(
            filtered_returns,
            weights=portfolios["Minimum Variance"],
            method=tail_risk_method,
            confidence=tail_confidence,
            window=min(tail_lookback, len(filtered_returns) - 1),
            covariance_method=covariance_method,
            simulation_count=mc_simulations,
            random_seed=mc_seed,
            ewma_decay=ewma_decay,
            ewma_mode=ewma_mode,
        )
        adv_frame = average_daily_volume(
            filtered_prices,
            filtered_volumes,
            lookback=min(liquidity_lookback, len(filtered_prices)),
        )
        adv_frame.index.name = "ticker"
        liquidity_metrics = liquidity_summary(
            portfolios["Minimum Variance"],
            kept_tickers,
            latest_prices,
            adv_frame.reindex(kept_tickers),
            cash_amount=cash,
            participation_rate=participation_rate,
        )
        impact_metrics = estimate_market_impact(
            portfolios["Minimum Variance"],
            kept_tickers,
            adv_frame.reindex(kept_tickers),
            cash_amount=cash,
            participation_rate=participation_rate,
            temporary_coefficient=impact_temporary,
            permanent_coefficient=impact_permanent,
        )
        capacity_metrics = estimate_capacity(
            portfolios["Minimum Variance"],
            kept_tickers,
            adv_frame.reindex(kept_tickers),
            max_participation_rate=participation_rate,
            liquidation_days=capacity_days,
        )
        lvar_value = liquidity_adjusted_var(
            tail_model.var(tail_confidence),
            impact_metrics,
            mode=liquidity_var_mode,
        )
        rolling_weights = _rolling_min_var_weights(
            filtered_returns,
            method=covariance_method,
            periods=periods,
            window=min(max(10, tail_lookback), len(filtered_returns)),
            ewma_decay=ewma_decay,
            ewma_mode=ewma_mode,
        )
        turnover_metrics = turnover_summary(rolling_weights, periods)
        covariance_forecasts = rolling_covariance_forecasts(
            filtered_returns,
            method=covariance_method,
            periods_per_year=periods,
            window=min(max(10, tail_lookback), len(filtered_returns)),
            ewma_decay=ewma_decay,
            ewma_mode=ewma_mode,
        )
        latest_forecast_cov = covariance_forecasts.iloc[-1] if not covariance_forecasts.empty else cov
        forecast_error = covariance_forecast_error(filtered_returns, covariance_forecasts)
        factor_model_instance = make_factor_model(
            factor_model,
            n_factors=min(pca_factors, len(kept_tickers)),
        )
        if factor_model == "pca":
            factor_model_instance.fit(filtered_returns)
        elif factor_model == "macro":
            factor_model_instance.fit(
                filtered_returns,
                factor_data=build_macro_factor_proxies(filtered_returns, filtered_volumes),
                volumes=filtered_volumes,
            )
        else:
            factor_model_instance.fit(
                filtered_returns,
                prices=filtered_prices,
                volumes=filtered_volumes,
            )
        factor_decomp = factor_model_instance.factor_risk_decomposition(portfolios["Minimum Variance"])

        typer.echo()
        typer.echo(
            format_run_header(
                tickers=ticker_list,
                freq=freq_normalized,
                years=years,
                observations=int(filtered_returns.shape[0]),
                cap_policy=cap_policy,
                cache_enabled=cache,
            )
        )
        typer.echo(f"Covariance method: {covariance_method}")
        typer.echo(f"Tail-risk method: {tail_risk_method}")
        typer.echo(f"Factor model: {factor_model}")
        typer.echo(f"Liquidity VaR mode: {liquidity_var_mode}")
        typer.echo(f"Cash Amount (USD): {cash:.2f}")
        typer.echo(f"Required tickers: {len(required_list)}")
        typer.echo(f"Optional tickers: {len(optional_list)}")
        typer.echo(f"Assets after dominance filter: {len(kept_tickers)} / {len(ticker_list)}")

        typer.echo("\nMinimum Variance Portfolio")
        typer.echo(
            format_portfolio_allocation_table(
                tickers=kept_tickers,
                weights=portfolios["Minimum Variance"],
                latest_prices=latest_prices,
                cash_amount=cash,
            )
        )

        typer.echo("\nRisk Parity (ERC) Portfolio")
        typer.echo(
            format_portfolio_allocation_table(
                tickers=kept_tickers,
                weights=portfolios["Risk Parity (ERC)"],
                latest_prices=latest_prices,
                cash_amount=cash,
            )
        )
        typer.echo("\nTurnover-Aware Minimum Variance Portfolio")
        typer.echo(
            format_portfolio_allocation_table(
                tickers=kept_tickers,
                weights=portfolios["Min Variance (Turnover-Aware)"],
                latest_prices=latest_prices,
                cash_amount=cash,
            )
        )

        typer.echo("\nPredicted Annual Volatility and Expected Return")
        for name, weights in portfolios.items():
            annual_vol = portfolio_vol(weights, cov, periods_per_year=periods) * 100
            annual_ret = float(expected_returns @ weights) * 100
            typer.echo(f"{name}: vol={annual_vol:.2f}%, expected_return={annual_ret:.2f}%")
            if show_risk_contrib:
                contrib = risk_contributions(weights, cov)
                ordered = sorted(
                    zip(kept_tickers, contrib), key=lambda x: (-float(x[1]), x[0])
                )
                typer.echo("  Risk contribution (% total variance):")
                for ticker, rc in ordered:
                    typer.echo(f"    {ticker}: {rc * 100:.2f}%")

        typer.echo()
        typer.echo(
            format_key_value_block(
                "Covariance Diagnostics",
                covariance_model.diagnostics(),
            )
        )
        typer.echo()
        typer.echo(
            format_matrix(
                "Current Correlation",
                kept_tickers,
                covariance_model.get_corr(),
                decimals=3,
            )
        )
        typer.echo()
        typer.echo(
            format_key_value_block(
                "Current Volatility",
                {
                    ticker: f"{vol * 100:.2f}%"
                    for ticker, vol in zip(kept_tickers, covariance_model.get_vol(annualized=True))
                },
            )
        )
        typer.echo()
        typer.echo(
            format_key_value_block(
                "Forecast Diagnostics",
                {
                    "forecast_window": min(max(10, tail_lookback), len(filtered_returns)),
                    "forecast_horizon": 1,
                    "latest_forecast_trace": f"{float(np.trace(latest_forecast_cov)):.8f}",
                    "realized_vs_forecast_vol_gap": (
                        f"{abs(np.sqrt(np.trace(cov)) - np.sqrt(np.trace(latest_forecast_cov))):.8f}"
                    ),
                    "forecast_rmse": f"{forecast_error['forecast_rmse']:.8f}",
                    "forecast_mae": f"{forecast_error['forecast_mae']:.8f}",
                },
            )
        )
        typer.echo()
        typer.echo(
            format_key_value_block(
                "Historical VaR / ES",
                {
                    "confidence": tail_confidence,
                    "var_1d": f"{tail_model.var(tail_confidence):.6f}",
                    "es_1d": f"{tail_model.es(tail_confidence):.6f}",
                    "liquidity_adjusted_var_1d": f"{lvar_value:.6f}",
                    "exceptions": int(tail_backtest["exception"].sum()) if not tail_backtest.empty else 0,
                },
            )
        )
        if not tail_backtest.empty:
            typer.echo()
            typer.echo(format_tail_backtest_table(tail_backtest))

        typer.echo()
        typer.echo(
            format_key_value_block(
                "Turnover Metrics",
                {
                    **{k: f"{v:.6f}" for k, v in turnover_metrics.items()},
                    "baseline_turnover_from_equal_weight": (
                        f"{0.5 * np.abs(portfolios['Minimum Variance'] - baseline_weights).sum():.6f}"
                    ),
                    "turnover_aware_shift": (
                        f"{0.5 * np.abs(portfolios['Min Variance (Turnover-Aware)'] - baseline_weights).sum():.6f}"
                    ),
                    "turnover_penalty": f"{turnover_penalty:.6f}",
                    "no_trade_buffer": f"{no_trade_buffer:.6f}",
                },
            )
        )
        typer.echo()
        typer.echo("Liquidity Metrics")
        typer.echo(format_liquidity_table(liquidity_metrics))
        typer.echo()
        typer.echo("Market Impact")
        typer.echo(format_impact_table(impact_metrics))
        typer.echo()
        typer.echo(
            format_key_value_block(
                "Capacity Estimate",
                {
                    "portfolio_capacity": f"{float(capacity_metrics['portfolio_capacity']):.2f}",
                    "bottleneck_ticker": capacity_metrics["bottleneck_ticker"],
                    "liquidation_days": capacity_days,
                },
            )
        )
        typer.echo()
        typer.echo(
            format_key_value_block(
                "Factor Diagnostics",
                factor_model_instance.diagnostics(),
            )
        )
        typer.echo()
        typer.echo("Factor Exposures")
        typer.echo(format_factor_exposures(factor_model_instance.exposures()))
        typer.echo()
        typer.echo(
            format_key_value_block(
                "Factor Risk Decomposition",
                {k: f"{v:.6f}" for k, v in factor_decomp.items()},
            )
        )
        typer.echo()
        typer.echo(
            "Model Comparison"
        )
        typer.echo(
            format_comparison_table(
                [
                    {
                        "model": covariance_method,
                        "forecast_rmse": f"{forecast_error['forecast_rmse']:.6f}",
                        "forecast_mae": f"{forecast_error['forecast_mae']:.6f}",
                        "vol_trace": f"{float(np.trace(cov)):.6f}",
                    },
                    {
                        "model": tail_risk_method,
                        "forecast_rmse": "N/A",
                        "forecast_mae": "N/A",
                        "vol_trace": f"{tail_model.var(tail_confidence):.6f}",
                    },
                    {
                        "model": factor_model,
                        "forecast_rmse": "N/A",
                        "forecast_mae": "N/A",
                        "vol_trace": f"{factor_decomp['factor_variance_share']:.6f}",
                    },
                ]
            )
        )

        if plot_frontier:
            w_mv = portfolios["Minimum Variance"]
            w_erc = portfolios["Risk Parity (ERC)"]
            fw, fr = efficient_frontier(
                cov,
                expected_returns,
                points=frontier_points,
                cap=effective_cap,
            )
            frontier_df = pd.DataFrame(
                [
                    {
                        "annual_vol": float(portfolio_vol(w, cov, periods_per_year=periods)),
                        "annual_return": float(r),
                    }
                    for w, r in zip(fw, fr)
                ]
            ).sort_values("annual_vol")

            chart_path = save_efficient_frontier_chart(
                frontier=frontier_df,
                min_var_vol=float(portfolio_vol(w_mv, cov, periods_per_year=periods)),
                min_var_ret=float(expected_returns @ w_mv),
                erc_vol=float(portfolio_vol(w_erc, cov, periods_per_year=periods)),
                erc_ret=float(expected_returns @ w_erc),
                output_path=plot_path,
            )
            typer.echo(f"\nEfficient frontier chart saved to: {chart_path}")

        typer.echo("\nDisregarded Optional Stocks")
        if disregarded_tickers:
            typer.echo(", ".join(disregarded_tickers))
        else:
            typer.echo("None")

    except typer.BadParameter as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    except (ConfigError, UnauthorizedError, RateLimitError, NotFoundError, DataError, APIError, ValueError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        if debug:
            typer.echo(traceback.format_exc(), err=True)
        raise typer.Exit(code=1)
    except Exception as exc:  # pragma: no cover
        if debug:
            typer.echo(traceback.format_exc(), err=True)
        else:
            typer.secho(f"Error: Unexpected failure: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
