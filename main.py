import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("🚀 Market Regime Detection")

data = yf.download("SPY", start="2000-01-01", progress=False)
df = pd.DataFrame()
df['price'] = data['Close']
df['ret'] = df['price'].pct_change()
df = df.dropna()

df['vol_20'] = df['ret'].rolling(20).std()
df['ret_20'] = df['price'].pct_change(20)
df['ma_dist'] = df['price'] / df['price'].rolling(20).mean() - 1
df['ret_5'] = df['price'].pct_change(5)
df = df.dropna()

features = df[['ret_20', 'vol_20', 'ma_dist', 'ret_5']]
X = StandardScaler().fit_transform(features)
df['regime'] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)

order = df.groupby('regime')['ret_20'].mean().sort_values().index
df['regime'] = df['regime'].map({old: new for new, old in enumerate(order)})
df['regime_name'] = df['regime'].map({0:'Bear', 1:'Sideways', 2:'Bull'})

plt.figure(figsize=(15,8))
plt.plot(df.index, df['price'], 'k-', linewidth=1.5, label='SPY')
colors = ['#ff6666', '#cccccc', '#66ff66']
for i in [0,1,2]:
    mask = df['regime']==i
    plt.fill_between(df.index, df['price'].min(), df['price'].max(), 
                    where=mask, color=colors[i], alpha=0.3, 
                    label=df.loc[mask,'regime_name'].iloc[0])
plt.title('SPY Market Regimes (KMeans)', fontsize=16, fontweight='bold')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('SPY_regimes.png', dpi=300)
plt.show()

print("🎉 SPY_regimes.png created!")
