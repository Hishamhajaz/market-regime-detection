import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from regime_model import RegimeModel

def create_folders():
    os.makedirs("plots", exist_ok=True)

def plot_regimes(df):
    create_folders()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot price
    ax.plot(df.index, df["price"], color="black", linewidth=1.5, label="SPY Price")
    
    # Background colors for regimes
    colors = {0: "#ff9999", 1: "#c3c3c3", 2: "#99ff99"}  # Red, Gray, Green
    for regime in [0, 1, 2]:
        mask = df["regime"] == regime
        ax.fill_between(df.index, df["price"].min(), df["price"].max(),
                       where=mask, color=colors[regime], alpha=0.3, label=df.loc[mask, "regime_name"].iloc[0])
    
    ax.set_title("SPY Market Regimes (KMeans Clustering)", fontsize=16, fontweight='bold')
    ax.set_ylabel("Price ($)", fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("plots/SPY_regimes.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Chart saved: plots/SPY_regimes.png")

def main():
    print("🚀 Starting Market Regime Detection...")
    
    # Create SPY model
    model = RegimeModel(ticker="SPY", start="2000-01-01")
    
    # Step 1: Download data
    model.download_data()
    
    # Step 2: Create features
    model.engineer_features()
    
    # Step 3: Detect regimes
    model.detect_regimes()
    
    # Step 4: Plot
    plot_regimes(model.data)
    
    print("\n🎉 SUCCESS! Check plots/SPY_regimes.png")

if __name__ == "__main__":
    main()
