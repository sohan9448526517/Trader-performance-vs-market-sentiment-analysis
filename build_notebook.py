import json

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}

cells = []

cells.append(md("""# Trader Performance vs Market Sentiment
### Hyperliquid Historical Trades x Bitcoin Fear/Greed Index

**Author:** Data Science Intern Assignment — Primetrade.ai
**Data:** 211,224 trades across 32 accounts (May 2023 – May 2025) and the daily Bitcoin Fear/Greed Index.

This notebook covers:
- **Part A** — Data loading, cleaning, alignment, and metric construction
- **Part B** — Sentiment vs performance/behavior analysis, and trader segmentation
- **Part C** — Actionable strategy recommendations
- **Bonus** — A simple next-day profitability classifier + KMeans trader archetypes

Run `pip install -r requirements.txt` then run cells top to bottom, or run the standalone scripts in `scripts/` in order (`01_prepare.py`, `02_analysis.py`, `03_bonus.py`).
"""))

# ---------------- PART A ----------------
cells.append(md("## Part A — Data Preparation"))
cells.append(code(open('scripts/01_prepare.py').read()))

cells.append(md("""### Part A notes

- **historical_data.csv**: 211,224 rows x 16 columns. Zero missing values, zero exact duplicate rows.
- **fear_greed_index.csv**: 2,644 rows x 4 columns. Zero missing values, zero duplicates.
- Trades span **2023-05-01 to 2025-05-01**; sentiment data covers 2018-2025, so full overlap exists.
- Only **6 of 211,224 trades** (0.003%) had no matching sentiment date and were dropped — negligible.
- The 5-way sentiment classification (`Extreme Fear`, `Fear`, `Neutral`, `Greed`, `Extreme Greed`) was
  collapsed into a 3-way bucket (`Fear`, `Neutral`, `Greed`) for cleaner comparison, since the extreme
  categories are sparse on a daily-account basis.
- **Leverage caveat**: the raw extract does not include an explicit margin/leverage field (no collateral
  or account-equity column), so "leverage distribution" from the brief is approximated using **trade
  notional size (Size USD)** as a proxy for risk-taking intensity. This is flagged throughout the analysis.
"""))

# ---------------- PART B ----------------
cells.append(md("## Part B — Analysis"))
cells.append(code(open('scripts/02_analysis.py').read()))

cells.append(md("""### Chart 1 — PnL distribution by sentiment
![chart1](../outputs/chart1_pnl_by_sentiment.png)

### Chart 2 — Win rate by sentiment
![chart2](../outputs/chart2_winrate_by_sentiment.png)

### Chart 3 — Trade frequency & size by sentiment
![chart3](../outputs/chart3_frequency_size_by_sentiment.png)

### Chart 4 — Long/short bias by sentiment
![chart4](../outputs/chart4_long_short_by_sentiment.png)

### Chart 5 — PnL: high vs low trade-size traders, by sentiment
![chart5](../outputs/chart5_pnl_size_segment_sentiment.png)

### Chart 6 — Trade frequency: frequent vs infrequent traders, by sentiment
![chart6](../outputs/chart6_freq_segment_sentiment.png)
"""))

cells.append(md("""### Part B findings — see the full write-up in `WRITEUP.md` for detailed interpretation.

Quick summary of what the numbers show:

1. **Fear days have the highest average PnL per account-day (~$5,185) vs Greed (~$4,144) and Neutral (~$3,439)**,
   but the difference is *not* statistically significant at the 5% level (Mann-Whitney U, p≈0.06 for PnL,
   p≈0.26 for win rate). Win rates are similar across regimes (~83-86%), so the PnL edge on Fear days is
   driven by trade sizing and volatility, not accuracy.
2. **Traders scale up on Fear days**: trades/account-day rises (105 vs 77 on Greed) and average trade size
   nearly doubles (~$7,182 vs ~$4,574). They also lean more long-biased (54% long) on Fear days than on
   Greed days (47% long, i.e. tilted short) — a mild contrarian pattern (buying fear, some profit-taking/short
   bias on greed).
3. **Segment interaction matters**: High trade-size traders earn far more on Fear days (~$9,540 avg) than
   Low trade-size traders (~$2,576) — the sentiment effect is concentrated in traders who already size up.
   Frequent traders show the same pattern (~$7,955 on Fear vs ~$2,525 for infrequent).
4. **Consistent-winner accounts hold a stable ~90%+ win rate across all sentiment regimes**, while
   inconsistent accounts sit around 75-78% regardless of sentiment — sentiment does not close the
   skill/consistency gap between trader types.
"""))

# ---------------- PART C ----------------
cells.append(md("""## Part C — Actionable Output

**Strategy idea 1 — Sentiment-scaled risk budget for high-conviction (large-size) traders.**
High trade-size accounts show a 3.7x PnL uplift on Fear days vs Low trade-size accounts on the same days,
while win rate stays flat across regimes. This suggests the edge on Fear days is a *sizing/volatility*
effect, not a *skill* effect. Rule of thumb: for traders already in the "high size" segment, allow a
modest (e.g. +20-30%) increase in position size specifically on Fear/Extreme-Fear days, capped by a hard
daily-loss stop, since realized variance (`pnl_std`) is also higher in Fear regimes (the tail risk cuts
both ways).

**Strategy idea 2 — Frequency throttle for low-size/infrequent traders during Greed.**
Infrequent and low-size traders do *not* show a Fear-day PnL edge (their averages are close to flat or
even lower on Fear days), meaning they are not capturing the same opportunity as the top segment. Rather
than increasing risk on Fear days for this group, the safer rule of thumb is: keep frequency and size
unchanged into Fear days for infrequent/low-size traders, and instead focus effort on trade *selection*
(higher win-rate entries) during Greed days, when the pooled long/short bias flips short and average
trade size for this segment stays roughly flat — i.e. this group's edge does not come from riding
sentiment momentum.

**Caveat:** none of the Fear vs Greed PnL differences clear a strict p<0.05 significance bar on this
sample (32 accounts, ~2 years), so these are *directional* rules of thumb to test further with more
data / a live paper-trading period, not settled statistical facts.
"""))

# ---------------- BONUS ----------------
cells.append(md("## Bonus — Predictive Model & Trader Clustering"))
cells.append(code(open('scripts/03_bonus.py').read()))

cells.append(md("""### Chart 7 — Feature importance (next-day profitability model)
![chart7](../outputs/chart7_feature_importance.png)

### Chart 8 — Trader archetypes (KMeans, k=3)
![chart8](../outputs/chart8_trader_clusters.png)

### Bonus notes

- A Random Forest predicting **next-day account profitability** (profit vs loss) from today's behavior +
  sentiment reaches **ROC AUC ≈ 0.65** — meaningfully better than a coin flip but far from a tradeable
  signal on its own. The most important features are trade size, trade count, today's PnL, and the
  sentiment index value itself — confirming sentiment carries *some* predictive signal on top of pure
  trading behavior.
- **KMeans (k=3)** on account-level features (trade frequency, size, avg daily PnL, win rate, win-rate
  volatility) recovers three intuitive archetypes:
  - **Cluster 0 — "Volatile mid-size traders"**: moderate frequency/size, lower and noisier win rate.
  - **Cluster 1 — "Steady/consistent traders"**: lower frequency, high and stable win rate (~96%).
  - **Cluster 2 — "High-volume power traders"**: very high frequency and large trade size, strong win rate.
"""))

cells.append(md("""## Reproducibility

```bash
pip install -r requirements.txt
python scripts/01_prepare.py   # Part A: cleaning + metrics -> outputs/*.csv
python scripts/02_analysis.py  # Part B: sentiment analysis + segmentation -> outputs/chart1-6, csvs
python scripts/03_bonus.py     # Bonus: model + clustering -> outputs/chart7-8, csvs
```

All intermediate and final tables are saved under `outputs/`. See `README.md` for the full repo structure
and `WRITEUP.md` for the one-page methodology / insights / strategy summary.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open('notebooks/trader_sentiment_analysis.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook written.")
