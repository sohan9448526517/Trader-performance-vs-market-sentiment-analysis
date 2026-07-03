# Write-up: Trader Performance vs Market Sentiment

## Methodology

Two datasets were used: 211,224 Hyperliquid trades from 32 accounts (May 2023 – May 2025), and the
daily Bitcoin Fear/Greed index (2018–2025, fully covering the trade window). Both were clean —
zero missing values and zero duplicate rows in either file. Trades were timestamped and rounded to
a daily date, then left-joined to the sentiment index on that date; only 6 of 211,224 trades
(0.003%) had no match and were dropped. The 5-way sentiment label was collapsed into a 3-way
bucket (Fear = Fear + Extreme Fear, Greed = Greed + Extreme Greed, Neutral) to keep group sizes
usable at the daily level.

From the row-level trades, two aggregate tables were built: **account-day** (32 accounts × active
days) and **pooled market-day**, each carrying total PnL, win rate (share of PnL-realizing trades
that were positive), trade count, average/total notional size, and long/short ratio. An
account-level summary (avg trades/day, avg daily PnL, PnL volatility, win rate and its volatility)
was used for segmentation. Note: the raw data has no margin/collateral/leverage field, so **trade
notional size (USD)** stands in as the risk-intensity proxy throughout — this is a real limitation
of this data extract, not a design choice.

Group comparisons (Fear vs Greed) used the Mann-Whitney U test rather than a t-test, since PnL is
heavily right-skewed with extreme outliers. Segments were split on the median of the relevant
account-level feature (trade size, trade frequency, or win-rate level/volatility for the
consistency segment).

## Insights

1. **Fear days produce higher average PnL, but the gap isn't statistically significant, and win
   rate barely moves.** Average account-day PnL is ~$5,185 on Fear days vs ~$4,144 on Greed days
   (Mann-Whitney p≈0.06), while win rate stays flat at 83–86% across all regimes (p≈0.26). The
   apparent Fear-day edge comes from bigger, more volatile bets, not better trade selection —
   PnL variance is also markedly higher on Fear days.

2. **Traders behaviorally lean in during Fear, not away from it.** Trades/account-day rises from
   ~77 (Greed) to ~105 (Fear), average trade notional nearly doubles (~$4,574 → ~$7,182), and the
   long/short mix flips from net-short-leaning on Greed days (47% long) to net-long on Fear days
   (54% long) — consistent with a "buy the fear" tendency across this trader population, together
   with more short positioning as sentiment turns euphoric (profit-taking or fading the rally).

3. **The sentiment edge is concentrated in already-aggressive traders, not evenly distributed.**
   Splitting accounts into High vs Low trade-size segments, the Fear-day PnL advantage is ~$9,540
   avg for High-size traders vs only ~$2,576 for Low-size traders — a >3x gap. The same pattern
   holds for Frequent vs Infrequent traders (~$7,955 vs ~$2,525 on Fear days). Meanwhile,
   consistent-winner accounts hold ~90%+ win rates in every regime while inconsistent accounts sit
   around 75–78% regardless of sentiment — sentiment shifts the payoff, not the skill gap.

## Strategy recommendations

**1. Sentiment-scaled sizing, gated by segment.** For traders already in the High-size /
Frequent segment (the ones actually capturing the Fear-day PnL edge), a modest, capped increase
in position size (e.g. +20–30%) on Fear/Extreme-Fear days is defensible, paired with a hard daily
stop-loss, since PnL variance is also elevated in that regime — the same conditions that produce
upside tails also produce downside tails.

**2. No blanket sentiment-chasing for low-size/infrequent traders.** This segment shows little to
no Fear-day PnL uplift, so the better rule of thumb is to hold their size and frequency roughly
constant through sentiment swings, and instead focus on trade-selection quality (entries/exits)
during Greed regimes, when the pooled book tilts short — riding sentiment momentum isn't where
this segment's edge lives.

**Caveat:** with 32 accounts and ~2 years of data, the core Fear-vs-Greed PnL gap does not clear a
strict 5% significance threshold — these are directional rules of thumb worth validating with a
longer sample or a live paper-trading window, not settled conclusions.
