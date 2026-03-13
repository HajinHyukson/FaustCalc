from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_efficient_frontier_chart(
    *,
    frontier: pd.DataFrame,
    min_var_vol: float,
    min_var_ret: float,
    erc_vol: float,
    erc_ret: float,
    output_path: str,
) -> Path:
    """Save efficient frontier chart (expected return vs volatility)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        frontier["annual_vol"],
        frontier["annual_return"],
        color="steelblue",
        linewidth=2.0,
        label="Efficient Frontier",
    )
    ax.scatter([min_var_vol], [min_var_ret], color="darkgreen", s=70, label="Minimum Variance")
    ax.scatter([erc_vol], [erc_ret], color="firebrick", s=70, label="Risk Parity (ERC)")

    ax.set_title("Efficient Frontier")
    ax.set_xlabel("Predicted Annual Volatility")
    ax.set_ylabel("Expected Annual Return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
