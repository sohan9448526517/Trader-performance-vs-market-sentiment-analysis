"""
Part B - Analysis
Compares performance & behavior across Fear / Greed / Neutral days,
builds 3 trader segments, and runs simple significance checks.
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 110

daily_acct = pd.read_csv('outputs/daily_account_metrics.csv', parse_dates=['date'])
daily_mkt = pd.read_csv('outputs/daily_market_metrics.csv', parse_dates=['date'])
acct_summary = pd.read_csv('outputs/account_summary.csv')
merged = pd.read_csv('outputs/merged_trades.csv', parse_dates=['date'])

order = ['Fear', 'Neutral', 'Greed']
palette = {'Fear': '#c0392b', 'Neutral': '#7f8c8d', 'Greed': '#27ae60'}

# =================================================================
# B1. Performance by sentiment (pooled market, and account-day level)
# =================================================================
print("### Daily pooled PnL by sentiment ###")
g = daily_mkt.groupby('sentiment_bucket')['total_pnl'].agg(['count', 'mean', 'median', 'std'])
print(g.reindex(order))

print("\n### Account-day win rate by sentiment ###")
g2 = daily_acct.groupby('sentiment_bucket')['win_rate'].agg(['count', 'mean', 'median'])
print(g2.reindex(order))

print("\n### Account-day PnL by sentiment ###")
g3 = daily_acct.groupby('sentiment_bucket')['total_pnl'].agg(['count', 'mean', 'median', 'std'])
print(g3.reindex(order))

# Mann-Whitney U test: Fear vs Greed, account-day PnL (non-normal, robust to outliers)
fear_pnl = daily_acct.loc[daily_acct.sentiment_bucket == 'Fear', 'total_pnl'].dropna()
greed_pnl = daily_acct.loc[daily_acct.sentiment_bucket == 'Greed', 'total_pnl'].dropna()
u_stat, p_val = stats.mannwhitneyu(fear_pnl, greed_pnl, alternative='two-sided')
print(f"\nMann-Whitney U (account-day PnL, Fear vs Greed): U={u_stat:.0f}, p={p_val:.4f}")

fear_wr = daily_acct.loc[daily_acct.sentiment_bucket == 'Fear', 'win_rate'].dropna()
greed_wr = daily_acct.loc[daily_acct.sentiment_bucket == 'Greed', 'win_rate'].dropna()
u_stat2, p_val2 = stats.mannwhitneyu(fear_wr, greed_wr, alternative='two-sided')
print(f"Mann-Whitney U (account-day win rate, Fear vs Greed): U={u_stat2:.0f}, p={p_val2:.4f}")

# Drawdown proxy: avg daily drawdown by sentiment (pooled)
print("\n### Drawdown proxy (pooled cumulative PnL) by sentiment ###")
print(daily_mkt.groupby('sentiment_bucket')['drawdown'].agg(['mean', 'min']).reindex(order))

# -----------------------------------------------------------------
# Chart 1: Distribution of account-day PnL by sentiment (boxplot, clipped)
# -----------------------------------------------------------------
plot_df = daily_acct[daily_acct['total_pnl'].abs() < daily_acct['total_pnl'].abs().quantile(0.95)]
fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=plot_df, x='sentiment_bucket', y='total_pnl', order=order, palette=palette, ax=ax, showfliers=False)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('Account-Day Closed PnL by Market Sentiment\n(outliers >95th pct trimmed for readability)')
ax.set_xlabel('Sentiment')
ax.set_ylabel('Closed PnL (USD)')
plt.tight_layout()
plt.savefig('outputs/chart1_pnl_by_sentiment.png')
plt.close()

# -----------------------------------------------------------------
# Chart 2: Win rate by sentiment
# -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
wr_summary = daily_acct.groupby('sentiment_bucket')['win_rate'].mean().reindex(order)
bars = ax.bar(wr_summary.index, wr_summary.values, color=[palette[k] for k in wr_summary.index])
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005, f"{b.get_height():.1%}", ha='center')
ax.set_title('Average Account-Day Win Rate by Market Sentiment')
ax.set_ylabel('Win Rate')
ax.set_ylim(0, max(wr_summary.values) * 1.25)
plt.tight_layout()
plt.savefig('outputs/chart2_winrate_by_sentiment.png')
plt.close()

# =================================================================
# B2. Behavior by sentiment: trade frequency, size, long/short bias
# =================================================================
print("\n### Behavior by sentiment ###")
beh = daily_acct.groupby('sentiment_bucket').agg(
    avg_trades_per_acct_day=('trades_count', 'mean'),
    avg_trade_size_usd=('avg_trade_size_usd', 'mean'),
    avg_long_short_ratio=('long_short_ratio', 'mean'),
).reindex(order)
print(beh)

# long/short bias directly from row-level data
ls = merged.groupby('sentiment_bucket').apply(
    lambda d: pd.Series({
        'pct_long': d['is_long'].sum() / (d['is_long'].sum() + d['is_short'].sum()),
        'pct_short': d['is_short'].sum() / (d['is_long'].sum() + d['is_short'].sum()),
        'avg_size_usd': d['Size USD'].mean(),
        'trades': len(d)
    })
).reindex(order)
print("\n### Long/Short bias (row-level) ###")
print(ls)

# -----------------------------------------------------------------
# Chart 3: Trade frequency & avg trade size by sentiment (dual panel)
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
beh['avg_trades_per_acct_day'].plot(kind='bar', ax=axes[0], color=[palette[k] for k in beh.index])
axes[0].set_title('Avg Trades per Account-Day')
axes[0].set_xlabel('')
axes[0].tick_params(axis='x', rotation=0)

beh['avg_trade_size_usd'].plot(kind='bar', ax=axes[1], color=[palette[k] for k in beh.index])
axes[1].set_title('Avg Trade Size (USD)')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig('outputs/chart3_frequency_size_by_sentiment.png')
plt.close()

# -----------------------------------------------------------------
# Chart 4: Long vs Short share by sentiment (stacked bar)
# -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ls[['pct_long', 'pct_short']].plot(kind='bar', stacked=True, ax=ax,
                                    color=['#2980b9', '#e67e22'])
ax.set_title('Long vs Short Trade Share by Sentiment')
ax.set_ylabel('Share of Trades')
ax.tick_params(axis='x', rotation=0)
ax.legend(['Long', 'Short'])
plt.tight_layout()
plt.savefig('outputs/chart4_long_short_by_sentiment.png')
plt.close()

# =================================================================
# B3. Segmentation
# =================================================================
# Segment 1: High vs Low trade-size traders
size_median = acct_summary['avg_trade_size_usd'].median()
acct_summary['size_segment'] = np.where(acct_summary['avg_trade_size_usd'] >= size_median, 'High Size', 'Low Size')

# Segment 2: Frequent vs Infrequent traders
freq_median = acct_summary['avg_trades_per_day'].median()
acct_summary['freq_segment'] = np.where(acct_summary['avg_trades_per_day'] >= freq_median, 'Frequent', 'Infrequent')

# Segment 3: Consistent winners vs inconsistent (based on win_rate mean & std)
wr_median = acct_summary['avg_win_rate'].median()
consistency_median = acct_summary['win_rate_std'].median()
def consistency_label(row):
    if row['avg_win_rate'] >= wr_median and row['win_rate_std'] <= consistency_median:
        return 'Consistent Winner'
    elif row['avg_win_rate'] < wr_median and row['win_rate_std'] > consistency_median:
        return 'Inconsistent'
    else:
        return 'Mixed'
acct_summary['consistency_segment'] = acct_summary.apply(consistency_label, axis=1)

acct_summary.to_csv('outputs/account_summary_segmented.csv', index=False)
print("\n### Segment sizes ###")
print(acct_summary['size_segment'].value_counts())
print(acct_summary['freq_segment'].value_counts())
print(acct_summary['consistency_segment'].value_counts())

# Merge segments back onto daily_acct for sentiment x segment analysis
seg_map = acct_summary.set_index('Account')[['size_segment', 'freq_segment', 'consistency_segment']]
daily_acct_seg = daily_acct.merge(seg_map, on='Account', how='left')
daily_acct_seg.to_csv('outputs/daily_account_segmented.csv', index=False)

print("\n### PnL by Size Segment x Sentiment ###")
print(daily_acct_seg.groupby(['size_segment', 'sentiment_bucket'])['total_pnl'].mean().unstack().reindex(columns=order))

print("\n### PnL by Freq Segment x Sentiment ###")
print(daily_acct_seg.groupby(['freq_segment', 'sentiment_bucket'])['total_pnl'].mean().unstack().reindex(columns=order))

print("\n### Win rate by Consistency Segment x Sentiment ###")
print(daily_acct_seg.groupby(['consistency_segment', 'sentiment_bucket'])['win_rate'].mean().unstack().reindex(columns=order))

# -----------------------------------------------------------------
# Chart 5: PnL by size segment x sentiment (grouped bar)
# -----------------------------------------------------------------
pivot = daily_acct_seg.groupby(['sentiment_bucket', 'size_segment'])['total_pnl'].mean().unstack()
pivot = pivot.reindex(order)
fig, ax = plt.subplots(figsize=(8, 5))
pivot.plot(kind='bar', ax=ax, color=['#8e44ad', '#f1c40f'])
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title('Avg Account-Day PnL: High vs Low Trade-Size Traders, by Sentiment')
ax.set_ylabel('Avg PnL (USD)')
ax.tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig('outputs/chart5_pnl_size_segment_sentiment.png')
plt.close()

# -----------------------------------------------------------------
# Chart 6: Trade frequency by freq segment x sentiment
# -----------------------------------------------------------------
pivot2 = daily_acct_seg.groupby(['sentiment_bucket', 'freq_segment'])['trades_count'].mean().unstack().reindex(order)
fig, ax = plt.subplots(figsize=(8, 5))
pivot2.plot(kind='bar', ax=ax, color=['#16a085', '#d35400'])
ax.set_title('Avg Trades per Account-Day: Frequent vs Infrequent Traders, by Sentiment')
ax.set_ylabel('Avg Trades / Day')
ax.tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig('outputs/chart6_freq_segment_sentiment.png')
plt.close()

print("\nDONE - Part B complete. Charts saved to outputs/.")
