"""
TASK 2: TECHNICAL INDICATORS - SIMPLIFIED WORKING VERSION
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TASK 2: TECHNICAL INDICATORS ANALYSIS")
print("=" * 60)

# Define indicators
def calc_sma(data, window):
    return data.rolling(window=window).mean()

def calc_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Stock to analyze
ticker = "AAPL"
print(f"\nAnalyzing {ticker}...")

# Download data
stock = yf.Ticker(ticker)
df = stock.history(period="6mo")
print(f"Downloaded {len(df)} days of data")

# Calculate indicators
df['SMA20'] = calc_sma(df['Close'], 20)
df['SMA50'] = calc_sma(df['Close'], 50)
df['RSI'] = calc_rsi(df['Close'], 14)

# Get latest values
latest_price = df['Close'].iloc[-1]
latest_sma20 = df['SMA20'].iloc[-1]
latest_sma50 = df['SMA50'].iloc[-1]
latest_rsi = df['RSI'].iloc[-1]

print(f"\nCurrent Price: ${latest_price:.2f}")
print(f"SMA 20: ${latest_sma20:.2f}")
print(f"SMA 50: ${latest_sma50:.2f}")
print(f"RSI: {latest_rsi:.1f}")

# Generate signals
print("\n" + "-" * 40)
print("TRADING SIGNALS:")
print("-" * 40)

if latest_rsi < 30:
    print("🔴 RSI: OVERSOLD -> BUY SIGNAL")
elif latest_rsi > 70:
    print("🟢 RSI: OVERBOUGHT -> SELL SIGNAL")
else:
    print("⚪ RSI: NEUTRAL -> NO SIGNAL")

if latest_price > latest_sma50:
    print("📈 Price ABOVE SMA50 -> UPTREND")
else:
    print("📉 Price BELOW SMA50 -> DOWNTREND")

# Create visualization
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Plot 1: Price with SMAs
axes[0].plot(df.index, df['Close'], label='Close Price', color='black', linewidth=1.5)
axes[0].plot(df.index, df['SMA20'], label='SMA 20', color='blue', linewidth=1)
axes[0].plot(df.index, df['SMA50'], label='SMA 50', color='red', linewidth=1)
axes[0].set_title(f'{ticker} - Price with Moving Averages', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Price ($)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: RSI
axes[1].plot(df.index, df['RSI'], color='purple', linewidth=1.5)
axes[1].axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
axes[1].axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
axes[1].fill_between(df.index, 30, 70, alpha=0.1, color='gray')
axes[1].set_title('RSI (14-day)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('RSI')
axes[1].set_ylim(0, 100)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Volume
axes[2].bar(df.index, df['Volume'], color='steelblue', alpha=0.7)
axes[2].set_title('Trading Volume', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Volume')
axes[2].set_xlabel('Date')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{ticker}_technical_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Chart saved as '{ticker}_technical_analysis.png'")

plt.show()

print("\n" + "=" * 60)
print("TASK 2 COMPLETED SUCCESSFULLY!")
print("=" * 60)