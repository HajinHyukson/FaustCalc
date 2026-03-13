# Risk-First Portfolio CLI

Risk-first portfolio analytics CLI for long-only portfolios using Financial Modeling Prep (FMP) Stable EOD prices.

## Setup

1. Create and activate virtual environment (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

3. Set API key:
- Environment variable (current shell):
```powershell
$env:FMP_API_KEY="your_key_here"
```
- Or project `.env` file:
```text
FMP_API_KEY=your_key_here
```

`python-dotenv` is optional at runtime. If installed, `.env` is auto-loaded.

## Run

Canonical command:
```powershell
python -m src.cli --required-tickers "SPY,QQQ" --optional-tickers "AAPL,MSFT,NVDA,AMZN" --years 3 --freq weekly --cash 100000
```

CLI options:
- `--required-tickers` (always included in portfolio)
- `--optional-tickers` (can be disregarded if inferior)
- `--years`, `-y` (default `3`)
- `--freq` (`weekly|daily`, default `weekly`)
- `--cash` (required positive float, total portfolio amount in USD)
- `--covariance-method` (`ledoit_wolf|ewma|garch|dcc_garch`, default `ledoit_wolf`)
- `--ewma-decay` (default from config: `0.94`)
- `--ewma-mode` (`expanding|rolling`, default `expanding`)
- `--tail-risk-method` (`historical|monte_carlo`)
- `--tail-confidence` (default `0.95`)
- `--tail-lookback` (default `60`)
- `--mc-simulations` (default `10000`)
- `--mc-seed` (default `42`)
- `--participation-rate` (default `0.10`)
- `--liquidity-lookback` (default `20`)
- `--impact-temporary`, `--impact-permanent`
- `--capacity-days`
- `--factor-model` (`pca|macro|style`)
- `--pca-factors`
- `--liquidity-var-mode` (`screening|detailed`)
- `--turnover-penalty`
- `--no-trade-buffer`
- `--cache/--no-cache` (default cache on)
- `--cache-ttl-hours` (override default 24h TTL)
- `--plot-frontier/--no-plot-frontier` (save efficient frontier chart as PNG)
- `--plot-path` (default `outputs/efficient_frontier.png`)
- `--frontier-points` (default `31`)
- `--dropna/--no-dropna` (default `--dropna`)
- `--show-risk-contrib`
- `--debug`
- `--log-level INFO|DEBUG`

## Output

Each run prints:
- Run metadata: timestamp, tickers, freq/years, aligned observations, cap policy, cache setting
- Covariance model used plus covariance diagnostics
- Current correlation matrix and annualized asset volatility snapshot
- Rolling covariance forecast error metrics
- DCC-GARCH covariance support and time-varying correlation estimates
- One allocation table for:
  - Minimum Variance
  - Risk Parity (ERC)
- Historical VaR / ES summary and recent rolling backtest rows
- Monte Carlo VaR / ES when selected
- Turnover metrics from rolling minimum-variance rebalances
- ADV-based liquidity metrics and days-to-liquidate estimates
- Market-impact estimates and portfolio capacity bottleneck
- Liquidity-adjusted VaR output
- PCA factor diagnostics, exposures, and factor/specific risk split
- Macro and style proxy factor models with exposures and decomposition
- Turnover-aware minimum-variance portfolio output
- Table columns:
  - Weight %
  - Latest price
  - Shares to buy (whole shares)
  - Dollar amount allocated
- Predicted annualized volatility and expected annual return for both portfolios
- Disregarded optional stocks list (optional assets dominated on return/risk dimensions)
- Optional efficient frontier chart (expected annual return vs predicted annual volatility) when `--plot-frontier` is enabled

Input constraints:
- Minimum total unique tickers across both groups: `2`
- Maximum total unique tickers across both groups: `1000`

## Troubleshooting

- Unauthorized:
  - `Unauthorized: API key missing/invalid/inactive or endpoint not in plan.`
  - Verify `FMP_API_KEY`, plan status, and that Stable endpoint access is enabled.

- No ticker history:
  - `No historical data returned for TICKER (check symbol/plan).`
  - Check symbol spelling and plan coverage.

- Not enough overlap:
  - `Not enough overlapping history after alignment; try fewer tickers or more years.`
  - Reduce ticker count or increase `--years`.

- Cache:
  - Cache is stored in `.cache/` (TTL default 24h).
  - Use `--no-cache` to bypass cache for fresh pulls.

## Tests

Run local smoke tests (no network):
```powershell
pytest
```

## Vercel Deployment (Frontend + API Route)

This repo now includes a Next.js app in `frontend/` with a server API route:
- `POST /api/portfolio` -> runs `python -m src.cli` and returns the report text.

### 1) Vercel project settings

- Import the repo in Vercel.
- Set **Root Directory** to `frontend`.
- Framework Preset: `Next.js`.
- Build command: `npm run build`.
- Install command: `npm install`.

### 2) Environment variables in Vercel

Add the following in **Project Settings -> Environment Variables**:

- `FMP_API_KEY` = your Financial Modeling Prep key (required)
- `PYTHON_PATH` = `python3` (recommended for Vercel Linux runtime)

### 3) What the deployment runs

- UI page: `/`
- API endpoint: `/api/portfolio`
- API payload example:

```json
{
  "requiredTickers": "SPY,QQQ",
  "optionalTickers": "AAPL,MSFT,NVDA,AMZN",
  "years": 3,
  "freq": "weekly",
  "cash": 100000,
  "cache": true,
  "logLevel": "INFO"
}
```

### 4) Notes

- The API route executes from Node runtime and shells out to Python, so the Python source under `src/` must be present in the deployment bundle.
- If Python command resolution fails, set `PYTHON_PATH` explicitly.
