"""
Bonus:
1. Simple predictive model - predict next-day account profitability bucket
   (profit / loss) using today's sentiment + behavior features.
2. Cluster traders into behavioral archetypes (KMeans on account-level features).
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 110

daily_acct = pd.read_csv('outputs/daily_account_segmented.csv', parse_dates=['date'])
acct_summary = pd.read_csv('outputs/account_summary_segmented.csv')

# =================================================================
# 1. Predictive model: next-day profitable (1) vs not (0), per account
# =================================================================
df = daily_acct.sort_values(['Account', 'date']).copy()
df['is_profitable_day'] = (df['total_pnl'] > 0).astype(int)

# next day's outcome, within same account
df['next_day_profitable'] = df.groupby('Account')['is_profitable_day'].shift(-1)
df['sentiment_num'] = df['value']  # fear/greed index 0-100 for today

feature_cols = ['trades_count', 'total_pnl', 'win_rate', 'avg_trade_size_usd',
                 'total_volume_usd', 'long_short_ratio', 'sentiment_num']
model_df = df.dropna(subset=feature_cols + ['next_day_profitable']).copy()
model_df['long_short_ratio'] = model_df['long_short_ratio'].replace([np.inf, -np.inf], np.nan)
model_df = model_df.dropna(subset=['long_short_ratio'])

print("Model dataset shape:", model_df.shape)
print("Target balance:\n", model_df['next_day_profitable'].value_counts(normalize=True))

X = model_df[feature_cols]
y = model_df['next_day_profitable'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("\n=== Classification report (next-day profitable) ===")
print(classification_report(y_test, y_pred))
try:
    auc = roc_auc_score(y_test, y_proba)
    print("ROC AUC:", round(auc, 3))
except Exception as e:
    print("AUC error:", e)

importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importances:\n", importances)

fig, ax = plt.subplots(figsize=(7, 5))
importances.plot(kind='barh', ax=ax, color='#2c3e50')
ax.invert_yaxis()
ax.set_title('Feature Importance: Predicting Next-Day Profitability')
plt.tight_layout()
plt.savefig('outputs/chart7_feature_importance.png')
plt.close()

# =================================================================
# 2. Clustering traders into behavioral archetypes
# =================================================================
cluster_features = ['avg_trades_per_day', 'avg_trade_size_usd', 'avg_daily_pnl',
                     'avg_win_rate', 'win_rate_std']
cdf = acct_summary.dropna(subset=cluster_features).copy()

scaler = StandardScaler()
Xc = scaler.fit_transform(cdf[cluster_features])

k = 3
km = KMeans(n_clusters=k, random_state=42, n_init=10)
cdf['cluster'] = km.fit_predict(Xc)

print("\n=== Cluster profiles (mean of raw features) ===")
profile = cdf.groupby('cluster')[cluster_features].mean()
print(profile)
print("\nCluster sizes:\n", cdf['cluster'].value_counts())

cdf.to_csv('outputs/account_clusters.csv', index=False)

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(cdf['avg_trades_per_day'], cdf['avg_daily_pnl'],
                      c=cdf['cluster'], cmap='Set2', s=100, edgecolor='black')
ax.set_xlabel('Avg Trades / Day')
ax.set_ylabel('Avg Daily PnL (USD)')
ax.set_title('Trader Archetypes (KMeans, k=3)')
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.savefig('outputs/chart8_trader_clusters.png')
plt.close()

print("\nDONE - Bonus complete.")
