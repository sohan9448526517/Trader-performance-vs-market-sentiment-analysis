"""
Part A - Data preparation
Loads trader data + fear/greed index, cleans, aligns by date,
and builds the core daily/account-level metrics used in the analysis.
"""
import pandas as pd
import numpy as np

pd.set_option('display.width', 140)

# ---------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------
trades = pd.read_csv('data/historical_data__2_.csv')
fg = pd.read_csv('data/fear_greed_index__2_.csv')

print("=== RAW SHAPES ===")
print("trades:", trades.shape)
print("fear_greed:", fg.shape)

print("\n=== MISSING VALUES ===")
print("trades:\n", trades.isna().sum()[trades.isna().sum() > 0])
print("fear_greed:\n", fg.isna().sum()[fg.isna().sum() > 0])

print("\n=== DUPLICATES ===")
print("trades duplicated rows:", trades.duplicated().sum())
print("fear_greed duplicated rows:", fg.duplicated().sum())

# ---------------------------------------------------------------
# 2. Clean + parse dates
# ---------------------------------------------------------------
trades['datetime'] = pd.to_datetime(trades['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
trades['date'] = trades['datetime'].dt.date
trades['date'] = pd.to_datetime(trades['date'])

fg['date'] = pd.to_datetime(fg['date'])
fg = fg[['date', 'value', 'classification']].drop_duplicates(subset='date').sort_values('date')

# Collapse 5-way classification into a simple Fear/Greed/Neutral bucket
def bucket(c):
    if c in ('Fear', 'Extreme Fear'):
        return 'Fear'
    if c in ('Greed', 'Extreme Greed'):
        return 'Greed'
    return 'Neutral'

fg['sentiment_bucket'] = fg['classification'].apply(bucket)

print("\n=== TRADE DATE RANGE ===")
print(trades['date'].min(), '->', trades['date'].max())
print("=== SENTIMENT DATE RANGE ===")
print(fg['date'].min(), '->', fg['date'].max())
print("unique accounts:", trades['Account'].nunique())
print("unique coins:", trades['Coin'].nunique())

# ---------------------------------------------------------------
# 3. Row-level flags
# ---------------------------------------------------------------
# A "closing" trade is one that realizes PnL (Closed PnL != 0 OR direction contains Close)
trades['is_close'] = trades['Closed PnL'] != 0
trades['is_win'] = trades['Closed PnL'] > 0
trades['is_long'] = trades['Direction'].str.contains('Long', case=False, na=False) | (trades['Side'] == 'BUY')
trades['is_short'] = trades['Direction'].str.contains('Short', case=False, na=False) | (trades['Side'] == 'SELL')

# ---------------------------------------------------------------
# 4. Merge trades with sentiment on date
# ---------------------------------------------------------------
merged = trades.merge(fg[['date', 'value', 'classification', 'sentiment_bucket']], on='date', how='left')
print("\n=== TRADES WITHOUT SENTIMENT MATCH ===")
print(merged['sentiment_bucket'].isna().sum(), "/", len(merged))

merged = merged.dropna(subset=['sentiment_bucket'])

merged.to_csv('outputs/merged_trades.csv', index=False)
print("\nSaved outputs/merged_trades.csv", merged.shape)

# ---------------------------------------------------------------
# 5. Daily x Account metrics  (the core analysis table)
# ---------------------------------------------------------------
daily_acct = merged.groupby(['date', 'Account']).agg(
    trades_count=('Trade ID', 'count'),
    total_pnl=('Closed PnL', 'sum'),
    closing_trades=('is_close', 'sum'),
    wins=('is_win', 'sum'),
    avg_trade_size_usd=('Size USD', 'mean'),
    total_volume_usd=('Size USD', 'sum'),
    long_trades=('is_long', 'sum'),
    short_trades=('is_short', 'sum'),
    total_fees=('Fee', 'sum'),
).reset_index()

daily_acct['win_rate'] = np.where(daily_acct['closing_trades'] > 0,
                                   daily_acct['wins'] / daily_acct['closing_trades'], np.nan)
daily_acct['long_short_ratio'] = np.where(daily_acct['short_trades'] > 0,
                                           daily_acct['long_trades'] / daily_acct['short_trades'],
                                           np.nan)

daily_acct = daily_acct.merge(fg[['date', 'value', 'classification', 'sentiment_bucket']], on='date', how='left')
daily_acct.to_csv('outputs/daily_account_metrics.csv', index=False)
print("Saved outputs/daily_account_metrics.csv", daily_acct.shape)

# ---------------------------------------------------------------
# 6. Daily x Market (all accounts pooled) metrics
# ---------------------------------------------------------------
daily_mkt = merged.groupby('date').agg(
    trades_count=('Trade ID', 'count'),
    total_pnl=('Closed PnL', 'sum'),
    closing_trades=('is_close', 'sum'),
    wins=('is_win', 'sum'),
    avg_trade_size_usd=('Size USD', 'mean'),
    total_volume_usd=('Size USD', 'sum'),
    long_trades=('is_long', 'sum'),
    short_trades=('is_short', 'sum'),
    active_accounts=('Account', 'nunique'),
).reset_index()
daily_mkt['win_rate'] = daily_mkt['wins'] / daily_mkt['closing_trades']
daily_mkt['long_short_ratio'] = daily_mkt['long_trades'] / daily_mkt['short_trades']
daily_mkt = daily_mkt.merge(fg[['date', 'value', 'classification', 'sentiment_bucket']], on='date', how='left')

# simple drawdown proxy on cumulative pooled PnL
daily_mkt = daily_mkt.sort_values('date')
daily_mkt['cum_pnl'] = daily_mkt['total_pnl'].cumsum()
daily_mkt['running_max'] = daily_mkt['cum_pnl'].cummax()
daily_mkt['drawdown'] = daily_mkt['cum_pnl'] - daily_mkt['running_max']

daily_mkt.to_csv('outputs/daily_market_metrics.csv', index=False)
print("Saved outputs/daily_market_metrics.csv", daily_mkt.shape)

# ---------------------------------------------------------------
# 7. Account-level summary (for segmentation)
# ---------------------------------------------------------------
acct_summary = daily_acct.groupby('Account').agg(
    active_days=('date', 'nunique'),
    total_trades=('trades_count', 'sum'),
    avg_trades_per_day=('trades_count', 'mean'),
    total_pnl=('total_pnl', 'sum'),
    avg_daily_pnl=('total_pnl', 'mean'),
    pnl_std=('total_pnl', 'std'),
    avg_trade_size_usd=('avg_trade_size_usd', 'mean'),
    avg_win_rate=('win_rate', 'mean'),
    win_rate_std=('win_rate', 'std'),
).reset_index()
acct_summary['pnl_consistency'] = acct_summary['avg_daily_pnl'] / acct_summary['pnl_std'].replace(0, np.nan)
acct_summary.to_csv('outputs/account_summary.csv', index=False)
print("Saved outputs/account_summary.csv", acct_summary.shape)

print("\nDONE - Part A complete")
