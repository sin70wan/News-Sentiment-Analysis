"""
TASK 2: MULTIPLE STOCKS TECHNICAL ANALYSIS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("MULTI-STOCK TECHNICAL ANALYSIS")
print("=" * 60)

def calc_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Stocks to analyze
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

results = []

print("\n📈 Analyzing stocks...")
print("-" * 50)

for ticker in stocks:
    print(f"\n{ticker}:")
    
    # Download data
    stock = yf.Ticker(ticker)
    df = stock.history(period="3mo")
    
    if len(df) < 20:
        print(f"  ⚠️ Not enough data")
        continue
    
    # Calculate indicators
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['RSI'] = calc_rsi(df['Close'], 14)
    
    # Latest values
    price = df['Close'].iloc[-1]
    sma20 = df['SMA20'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    
    # Generate signal
    if rsi < 30:
        signal = "BUY 🔴"
    elif rsi > 70:
        signal = "SELL 🟢"
    else:
        signal = "HOLD ⚪"
    
    results.append({
        'Stock': ticker,
        'Price': price,
        'SMA20': sma20,
        'RSI': rsi,
        'Signal': signal
    })
    
    print(f"  Price: ${price:.2f}")
    print(f"  RSI: {rsi:.1f}")
    print(f"  Signal: {signal}")

# Create summary table
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)

print(f"\n{'Stock':<8} {'Price':<12} {'SMA20':<12} {'RSI':<10} {'Signal':<10}")
print("-" * 55)

for r in results:
    print(f"{r['Stock']:<8} ${r['Price']:<11.2f} ${r['SMA20']:<11.2f} {r['RSI']:<9.1f} {r['Signal']:<10}")

# Create comparison chart
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

stocks_list = [r['Stock'] for r in results]
rsi_values = [r['RSI'] for r in results]
price_values = [r['Price'] for r in results]

# RSI Comparison
colors_rsi = ['red' if rsi > 70 else ('green' if rsi < 30 else 'gray') for rsi in rsi_values]
axes[0].bar(stocks_list, rsi_values, color=colors_rsi, edgecolor='black')
axes[0].axhline(y=70, color='red', linestyle='--', linewidth=2, label='Overbought (70)')
axes[0].axhline(y=30, color='green', linestyle='--', linewidth=2, label='Oversold (30)')
axes[0].set_ylim(0, 100)
axes[0].set_title('RSI Comparison Across Stocks', fontsize=14, fontweight='bold')
axes[0].set_ylabel('RSI Value')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Add RSI values on bars
for i, (bar, rsi) in enumerate(zip(axes[0].patches, rsi_values)):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                 f'{rsi:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Price Comparison
axes[1].bar(stocks_list, price_values, color='steelblue', edgecolor='black')
axes[1].set_title('Current Price Comparison', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Price ($)')
axes[1].grid(True, alpha=0.3)

# Add price values on bars
for i, (bar, price) in enumerate(zip(axes[1].patches, price_values)):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                 f'${price:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('multi_stock_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Chart saved as 'multi_stock_comparison.png'")

plt.show()

print("\n" + "=" * 60)
print("✅ TASK 2 - MULTI-STOCK ANALYSIS COMPLETED!")
print("=" * 60)