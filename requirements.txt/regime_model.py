import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class RegimeModel:
    def __init__(self, ticker="SPY", start="2000-01-01"):
        self.ticker = ticker
        self.start = start
        self.data = None
        self.features = None
        self.scaler = None
        self.kmeans = None

    def download_data(self):
        print(f"Downloading {self.ticker} data...")
        df = yf.download(self.ticker, start=self.start)
        df = df[["Adj Close"]].rename(columns={"Adj Close": "price"})
        df["ret"] = df["price"].pct_change()
        self.data = df.dropna()
        print(f"Downloaded {len(self.data)} days of data")
        return self.data

    def engineer_features(self):
        df = self.data.copy()
        # Key features for regime detection
        df["vol_20"] = df["ret"].rolling(20).std()  # 20-day volatility
        df["ret_20"] = df["price"].pct_change(20)   # 20-day return
        df["ma_20"] = df["price"].rolling(20).mean()
        df["ma_dist"] = (df["price"] - df["ma_20"]) / df["ma_20"]
        df["ret_5"] = df["price"].pct_change(5)     # 5-day momentum
        
        self.features = df[["ret_20", "vol_20", "ma_dist", "ret_5"]].dropna()
        self.data = df.loc[self.features.index]
        print(f"Created features for {len(self.features)} days")
        return self.features

    def detect_regimes(self):
        X = self.features.values
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(X_scaled)
        
        # Map clusters to regimes (Bear=0, Sideways=1, Bull=2)
        regime_stats = pd.Series(labels).map(
            lambda x: self.data.loc[self.features.index]["ret_20"][labels == x].mean()
        ).sort_values()
        mapping = {cluster: i for i, cluster in enumerate(regime_stats.index)}
        
        self.data["regime"] = [mapping[label] for label in labels]
        self.data["regime_name"] = self.data["regime"].map({0: "Bear", 1: "Sideways", 2: "Bull"})
        
        regime_counts = self.data["regime_name"].value_counts()
        print("Regime detection complete:")
        print(regime_counts)
        return self.data
