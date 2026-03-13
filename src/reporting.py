from datetime import datetime
from typing import Mapping

import numpy as np
import pandas as pd


def format_run_header(
    *,
    tickers: list[str],
    freq: str,
    years: int,
    observations: int,
    cap_policy: str,
    cache_enabled: bool,
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "Run Summary",
        "-" * 40,
        f"Timestamp: {timestamp}",
        f"Tickers ({len(tickers)}): {', '.join(tickers)}",
        f"Frequency/Years: {freq} / {years}",
        f"Observations after alignment: {observations}",
        f"Cap policy: {cap_policy}",
        f"Cache: {'on' if cache_enabled else 'off'}",
    ]
    return "\n".join(lines)


def _format_row(cells: list[str], widths: list[int]) -> str:
    return " | ".join(cell.ljust(width) for cell, width in zip(cells, widths))


def format_weights(tickers: list[str], weights: np.ndarray) -> str:
    ordered = sorted(
        zip(tickers, np.asarray(weights, dtype=float)),
        key=lambda x: (-x[1], x[0]),
    )
    headers = ["Asset", "Weight"]
    rows = [[asset, f"{weight:.6f}"] for asset, weight in ordered]
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]
    body = [_format_row(headers, widths), _format_row(["-" * widths[0], "-" * widths[1]], widths)]
    body.extend(_format_row(row, widths) for row in rows)
    return "\n".join(body)


def format_varcov(tickers: list[str], cov: np.ndarray) -> str:
    cov_cells = [[f"{float(value):.8f}" for value in row] for row in cov]
    first_col_width = max(len("Asset"), *(len(t) for t in tickers))
    col_widths = [
        max(len(tickers[col_idx]), *(len(cov_cells[row_idx][col_idx]) for row_idx in range(len(tickers))))
        for col_idx in range(len(tickers))
    ]
    widths = [first_col_width, *col_widths]

    lines = [_format_row(["Asset", *tickers], widths)]
    for ticker, row in zip(tickers, cov_cells):
        lines.append(_format_row([ticker, *row], widths))
    return "\n".join(lines)


def format_summary(metrics: Mapping[str, float]) -> str:
    ordered_items = sorted(metrics.items(), key=lambda kv: kv[0])
    return "\n".join(f"{name}: {value:.6f}" for name, value in ordered_items)


def format_key_value_block(title: str, metrics: Mapping[str, object]) -> str:
    lines = [title, "-" * len(title)]
    for key, value in metrics.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def format_matrix(name: str, labels: list[str], matrix: np.ndarray, decimals: int = 4) -> str:
    cells = [[f"{float(value):.{decimals}f}" for value in row] for row in matrix]
    widths = [
        max(len("Asset"), *(len(label) for label in labels)),
        *[
            max(len(labels[col]), *(len(cells[row][col]) for row in range(len(labels))))
            for col in range(len(labels))
        ],
    ]
    lines = [name, "-" * len(name), _format_row(["Asset", *labels], widths)]
    for label, row in zip(labels, cells):
        lines.append(_format_row([label, *row], widths))
    return "\n".join(lines)


def format_liquidity_table(metrics: list[object]) -> str:
    headers = ["Asset", "ADV $", "ADV Shares", "Days To Liquidate", "Stale"]
    rows = [
        [
            getattr(metric, "ticker"),
            f"{getattr(metric, 'adv_dollars'):.2f}",
            f"{getattr(metric, 'adv_shares'):.0f}",
            f"{getattr(metric, 'days_to_liquidate'):.2f}",
            "yes" if getattr(metric, "stale") else "no",
        ]
        for metric in metrics
    ]
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]
    body = [
        _format_row(headers, widths),
        _format_row(["-" * w for w in widths], widths),
    ]
    body.extend(_format_row(row, widths) for row in rows)
    return "\n".join(body)


def format_impact_table(metrics: list[object]) -> str:
    headers = ["Asset", "Trade Value", "Participation", "Temp Cost", "Perm Cost", "Total Cost"]
    rows = [
        [
            getattr(metric, "ticker"),
            f"{getattr(metric, 'trade_value'):.2f}",
            f"{getattr(metric, 'participation_rate'):.4f}",
            f"{getattr(metric, 'temporary_cost'):.2f}",
            f"{getattr(metric, 'permanent_cost'):.2f}",
            f"{getattr(metric, 'total_cost'):.2f}",
        ]
        for metric in metrics
    ]
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]
    body = [_format_row(headers, widths), _format_row(["-" * w for w in widths], widths)]
    body.extend(_format_row(row, widths) for row in rows)
    return "\n".join(body)


def format_factor_exposures(exposures: pd.DataFrame) -> str:
    rounded = exposures.round(4)
    headers = ["Asset", *list(rounded.columns)]
    rows = [[idx, *[f"{float(v):.4f}" for v in row]] for idx, row in rounded.iterrows()]
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]
    body = [_format_row(headers, widths), _format_row(["-" * w for w in widths], widths)]
    body.extend(_format_row(row, widths) for row in rows)
    return "\n".join(body)


def format_comparison_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "None"
    headers = list(rows[0].keys())
    string_rows = [[str(row[key]) for key in headers] for row in rows]
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in string_rows))
        for col in range(len(headers))
    ]
    body = [_format_row(headers, widths), _format_row(["-" * w for w in widths], widths)]
    body.extend(_format_row(row, widths) for row in string_rows)
    return "\n".join(body)


def format_tail_backtest_table(backtest: pd.DataFrame) -> str:
    headers = ["Date", "VaR", "ES", "Realized Loss", "Exception"]
    rows = [
        [
            idx.strftime("%Y-%m-%d"),
            f"{row['var']:.6f}",
            f"{row['es']:.6f}",
            f"{row['realized_loss']:.6f}",
            "yes" if bool(row["exception"]) else "no",
        ]
        for idx, row in backtest.tail(10).iterrows()
    ]
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]
    body = [
        _format_row(headers, widths),
        _format_row(["-" * w for w in widths], widths),
    ]
    body.extend(_format_row(row, widths) for row in rows)
    return "\n".join(body)


def format_allocation_table(
    *,
    portfolio_weights: Mapping[str, np.ndarray],
    tickers: list[str],
    latest_prices: Mapping[str, float],
    cash_amount: float,
) -> str:
    """Format one table with weights and dollar/share allocations for each portfolio."""
    headers = ["Portfolio", "Asset", "Weight %", "Price", "Shares", "Dollar Amount"]
    rows: list[list[str]] = []

    for portfolio_name, weights in portfolio_weights.items():
        ordered = sorted(
            zip(tickers, np.asarray(weights, dtype=float)),
            key=lambda x: (-x[1], x[0]),
        )
        for asset, weight in ordered:
            price = float(latest_prices[asset])
            target_dollars = float(cash_amount) * float(weight)
            shares = int(np.floor(target_dollars / price)) if price > 0 else 0
            allocated_dollars = shares * price
            rows.append(
                [
                    portfolio_name,
                    asset,
                    f"{weight * 100:.2f}%",
                    f"{price:.2f}",
                    str(shares),
                    f"{allocated_dollars:.2f}",
                ]
            )

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]
    body = [
        _format_row(headers, widths),
        _format_row(["-" * w for w in widths], widths),
    ]
    body.extend(_format_row(row, widths) for row in rows)
    return "\n".join(body)


def format_portfolio_allocation_table(
    *,
    tickers: list[str],
    weights: np.ndarray,
    latest_prices: Mapping[str, float],
    cash_amount: float,
) -> str:
    """Format allocation table for a single portfolio."""
    headers = ["Asset", "Weight %", "Price", "Shares", "Dollar Amount"]
    ordered = sorted(
        zip(tickers, np.asarray(weights, dtype=float)),
        key=lambda x: (-x[1], x[0]),
    )

    rows: list[list[str]] = []
    for asset, weight in ordered:
        price = float(latest_prices[asset])
        target_dollars = float(cash_amount) * float(weight)
        shares = int(np.floor(target_dollars / price)) if price > 0 else 0
        allocated_dollars = shares * price
        rows.append(
            [
                asset,
                f"{weight * 100:.2f}%",
                f"{price:.2f}",
                str(shares),
                f"{allocated_dollars:.2f}",
            ]
        )

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]
    body = [
        _format_row(headers, widths),
        _format_row(["-" * w for w in widths], widths),
    ]
    body.extend(_format_row(row, widths) for row in rows)
    return "\n".join(body)
