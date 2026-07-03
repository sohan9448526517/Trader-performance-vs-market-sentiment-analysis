# Trader Performance vs Market Sentiment

Analysis of how Bitcoin Fear/Greed sentiment relates to trader behavior and performance on
Hyperliquid, built for the Primetrade.ai Data Science Intern assignment.

## Repo structure

```
.
├── data/
│   ├── historical_data__2_.csv        # raw Hyperliquid trade data (211,224 rows, 32 accounts)
│   └── fear_greed_index__2_.csv       # raw Bitcoin Fear/Greed index (2018-2025, daily)
├── scripts/
│   ├── 01_prepare.py                  # Part A: load, clean, align, build metrics
│   ├── 02_analysis.py                 # Part B: sentiment analysis, segmentation, charts
│   ├── 03_bonus.py                    # Bonus: predictive model + clustering
│   └── build_notebook.py              # assembles the .ipynb from the scripts above
├── notebooks/
│   └── trader_sentiment_analysis.ipynb   # single notebook containing everything, with charts
├── outputs/                           # all generated CSVs and PNG charts
├── requirements.txt
├── WRITEUP.md                         # 1-page methodology / insights / strategy summary
└── README.md
```

## How to run

```bash
python -m venv venv && source venv/bin/activate     # optional
pip install -r requirements.txt

python scripts/01_prepare.py    # Part A -> outputs/*.csv
python scripts/02_analysis.py   # Part B -> outputs/chart1-6*.png, more csvs
python scripts/03_bonus.py      # Bonus  -> outputs/chart7-8*.png, model + cluster csvs

# Optional: regenerate the notebook from the scripts
python scripts/build_notebook.py
```

Or just open `notebooks/trader_sentiment_analysis.ipynb` and run all cells top to bottom
(it embeds the exact same code as the scripts, plus narrative and chart references).

## Data summary

| Dataset | Rows | Cols | Missing | Duplicates | Date range |
|---|---|---|---|---|---|
| historical_data (trades) | 211,224 | 16 | 0 | 0 | 2023-05-01 → 2025-05-01 |
| fear_greed_index | 2,644 | 4 | 0 | 0 | 2018-02-01 → 2025-05-02 |

32 unique accounts, 246 unique traded symbols. Only 6 trades (0.003%) fell outside the sentiment
index's date coverage and were dropped during the merge.

## Key metrics built (Part A)

At the **account-day** and **pooled market-day** level:
- `total_pnl` — sum of Closed PnL
- `win_rate` — share of closing trades with positive PnL
- `trades_count`, `avg_trade_size_usd`, `total_volume_usd`
- `long_short_ratio` — long trades / short trades
- `drawdown` — running cumulative-PnL drawdown proxy (pooled)

At the **account** level (for segmentation): `avg_trades_per_day`, `avg_daily_pnl`, `pnl_std`,
`avg_win_rate`, `win_rate_std`.

**Leverage caveat:** the raw extract has no margin/collateral/leverage field, so trade **notional
size (USD)** is used throughout as the proxy for risk-taking intensity instead of true leverage.

## Key findings (Part B) — see `WRITEUP.md` for full detail

1. Fear-day account-day PnL is higher on average than Greed-day PnL (~$5.2k vs ~$4.1k), but the
   gap is not statistically significant at p<0.05 (Mann-Whitney U, p≈0.06); win rates are flat
   across regimes (~83-86%), so the edge is driven by sizing/volatility, not accuracy.
2. Traders scale up on Fear days: +37% more trades/account-day and ~57% larger average trade size
   vs Greed days, with a mild long-bias tilt (54% long on Fear vs 47% long on Greed).
3. The Fear-day PnL edge is concentrated in traders who are already high-size / high-frequency —
   low-size/infrequent traders see almost no sentiment-driven edge.
4. Consistent-winner accounts hold ~90%+ win rate in every sentiment regime; sentiment does not
   close the gap with inconsistent traders (~75-78% regardless of regime).

## Bonus results

- Random Forest predicting next-day account profitability: **ROC AUC ≈ 0.65** using behavior +
  sentiment features (trade size, trade count, PnL, sentiment index value are the top drivers).
- KMeans (k=3) recovers three trader archetypes: steady/consistent, volatile mid-size, and
  high-volume power traders.

## Notes on reproducibility

All randomness is seeded (`random_state=42`). Charts are saved as static PNGs in `outputs/` so
they render even without re-running the notebook. The scripts are idempotent — safe to re-run.
