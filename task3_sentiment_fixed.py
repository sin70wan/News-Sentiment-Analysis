

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("📊 TASK 3: SENTIMENT ANALYSIS & CORRELATION")
print("=" * 80)

# ============================================
# PART 1: CREATE SAMPLE NEWS DATA
# ============================================
print("\n📁 PART 1: Creating news dataset...")

np.random.seed(42)

end_date = datetime.now().date()
start_date = end_date - timedelta(days=180)
all_dates = [start_date + timedelta(days=x) for x in range(181)]

stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

positive_headlines = [
    "Stock hits all-time high after strong earnings",
    "Analyst upgrades to Buy, raises price target by 20%",
    "Company beats revenue estimates by wide margin",
    "New product launch exceeds expectations",
    "Profit margins expand significantly",
    "Board approves $5 billion share buyback",
    "Dividend increased by 15%",
    "Partnership announced with industry leader"
]

negative_headlines = [
    "Stock plunges on disappointing earnings",
    "Analyst downgrades to Sell",
    "Company misses revenue estimates",
    "CEO resigns effective immediately",
    "Regulatory investigation launched",
    "Profit warning issued for next quarter",
    "Lawsuit filed against company",
    "Supply chain disruptions impact production"
]

neutral_headlines = [
    "Company announces routine board meeting",
    "Executive to speak at conference",
    "Quarterly report scheduled for next week",
    "Hiring new CFO",
    "Office expansion plans revealed",
    "Patent application filed"
]

all_headlines = positive_headlines + negative_headlines + neutral_headlines

news_data = []
for _ in range(1000):
    date = np.random.choice(all_dates)
    stock = np.random.choice(stock_symbols)
    headline = np.random.choice(all_headlines)
    
    news_data.append({
        'date': date,
        'headline': headline,
        'stock': stock
    })

news_df = pd.DataFrame(news_data)
news_df['date'] = pd.to_datetime(news_df['date'])

print(f"✅ Created {len(news_df)} news articles")
print(f"📅 Date range: {news_df['date'].min().date()} to {news_df['date'].max().date()}")

# ============================================
# PART 2: SENTIMENT ANALYSIS WITH VADER
# ============================================
print("\n" + "=" * 80)
print("📝 PART 2: Sentiment Analysis with VADER")
print("=" * 80)

analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    try:
        return analyzer.polarity_scores(str(text))['compound']
    except:
        return 0.0

def classify_sentiment(score, threshold=0.05):
    if score > threshold:
        return 'Positive'
    elif score < -threshold:
        return 'Negative'
    else:
        return 'Neutral'

news_df['vader_score'] = news_df['headline'].apply(get_vader_sentiment)
news_df['sentiment_class'] = news_df['vader_score'].apply(classify_sentiment)

print("\n📰 Sample headlines:")
for i, row in news_df.head(8).iterrows():
    print(f"[{row['sentiment_class']:8}] {row['vader_score']:+.2f} | {row['headline'][:50]}")

sentiment_counts = news_df['sentiment_class'].value_counts()
print("\n📊 Sentiment Distribution:")
for sentiment, count in sentiment_counts.items():
    print(f"   {sentiment}: {count} articles ({count/len(news_df)*100:.1f}%)")

# ============================================
# PART 3: FETCH STOCK PRICE DATA
# ============================================
print("\n" + "=" * 80)
print("📈 PART 3: Fetching Stock Price Data")
print("=" * 80)

primary_stock = "AAPL"
print(f"\n📊 Analyzing {primary_stock}...")

stock = yf.Ticker(primary_stock)
stock_df = stock.history(start=start_date, end=end_date + timedelta(days=1))

if stock_df.empty:
    print("Creating simulated data...")
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    np.random.seed(42)
    price = 150
    prices = []
    for i in range(len(dates)):
        price += np.random.normal(0, 1)
        prices.append(max(price, 50))
    stock_df = pd.DataFrame({'Close': prices, 'Volume': np.random.randint(1000000, 10000000, len(dates))}, index=dates)

stock_df['daily_return'] = stock_df['Close'].pct_change() * 100
stock_df['date'] = stock_df.index.date

print(f"✅ Processed {len(stock_df)} days of data")
print(f"📊 Avg return: {stock_df['daily_return'].mean():+.3f}%")
print(f"📊 Volatility: {stock_df['daily_return'].std():.3f}%")

# ============================================
# PART 4: AGGREGATE DAILY SENTIMENT
# ============================================
print("\n" + "=" * 80)
print("📊 PART 4: Aggregating Daily Sentiment")
print("=" * 80)

stock_news = news_df[news_df['stock'] == primary_stock].copy()
if len(stock_news) == 0:
    stock_news = news_df.copy()

daily_sentiment = stock_news.groupby('date')['vader_score'].mean().reset_index()
daily_sentiment.columns = ['date', 'avg_sentiment']

stock_df['date'] = pd.to_datetime(stock_df['date'])
merged_df = pd.merge(stock_df, daily_sentiment, on='date', how='left')
merged_df['avg_sentiment'] = merged_df['avg_sentiment'].fillna(0)
merged_df['sentiment_class'] = merged_df['avg_sentiment'].apply(classify_sentiment)

print(f"✅ Merged {len(merged_df)} trading days")

# ============================================
# PART 5: CORRELATION ANALYSIS
# ============================================
print("\n" + "=" * 80)
print("📊 PART 5: Correlation Analysis")
print("=" * 80)

corr_df = merged_df.dropna(subset=['daily_return'])

if len(corr_df) > 1:
    pearson_corr, pearson_p = pearsonr(corr_df['avg_sentiment'], corr_df['daily_return'])
    spearman_corr, spearman_p = spearmanr(corr_df['avg_sentiment'], corr_df['daily_return'])

    print(f"\n📈 Pearson Correlation: {pearson_corr:.4f} (p={pearson_p:.4e})")
    print(f"📈 Spearman Correlation: {spearman_corr:.4f} (p={spearman_p:.4e})")

    if abs(pearson_corr) < 0.2:
        strength = "weak"
    elif abs(pearson_corr) < 0.4:
        strength = "weak to moderate"
    elif abs(pearson_corr) < 0.6:
        strength = "moderate"
    else:
        strength = "strong"

    direction = "positive" if pearson_corr > 0 else "negative"
    print(f"\n📝 Interpretation: {strength} {direction} correlation")
    print(f"   {'✅ Statistically significant' if pearson_p < 0.05 else '❌ Not statistically significant'}")

    sentiment_returns = merged_df.groupby('sentiment_class')['daily_return'].agg(['mean', 'count'])
    print(f"\n💰 Returns by Sentiment:")
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        if sentiment in sentiment_returns.index:
            print(f"   {sentiment}: {sentiment_returns.loc[sentiment, 'mean']:+.3f}% (n={int(sentiment_returns.loc[sentiment, 'count'])})")

# ============================================
# PART 6: LAG ANALYSIS
# ============================================
print("\n" + "=" * 80)
print("📊 PART 6: Lag Analysis")
print("=" * 80)

lag_results = []
for lag in range(1, 6):
    corr_df[f'future_lag_{lag}'] = corr_df['daily_return'].shift(-lag)
    valid = corr_df.dropna(subset=[f'future_lag_{lag}'])
    if len(valid) > 10:
        lag_corr, lag_p = pearsonr(valid['avg_sentiment'], valid[f'future_lag_{lag}'])
        lag_results.append({'lag': lag, 'correlation': lag_corr, 'p_value': lag_p})

print("\n   Lag | Correlation | p-value | Significance")
print("   " + "-" * 45)
for r in lag_results:
    sig = "✓" if r['p_value'] < 0.05 else "✗"
    print(f"   {r['lag']:3}d | {r['correlation']:+.4f}     | {r['p_value']:.4f} | {sig}")

# ============================================
# PART 7: CREATE SIMPLE VISUALIZATIONS
# ============================================
print("\n" + "=" * 80)
print("📊 PART 7: Creating Visualizations")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Sentiment Distribution
axes[0,0].bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'gray', 'red'], edgecolor='black')
axes[0,0].set_title('Sentiment Distribution', fontsize=12, fontweight='bold')
axes[0,0].set_ylabel('Number of Articles')

# Plot 2: Sentiment vs Returns Scatter
axes[0,1].scatter(corr_df['avg_sentiment'], corr_df['daily_return'], alpha=0.5, color='steelblue')
axes[0,1].axhline(y=0, color='gray', linestyle='--')
axes[0,1].axvline(x=0, color='gray', linestyle='--')
axes[0,1].set_title(f'Sentiment vs Returns (r = {pearson_corr:.3f})', fontsize=12, fontweight='bold')
axes[0,1].set_xlabel('Sentiment Score')
axes[0,1].set_ylabel('Daily Return (%)')

# Plot 3: Returns by Sentiment
sentiment_order = ['Positive', 'Neutral', 'Negative']
returns_data = [merged_df[merged_df['sentiment_class'] == s]['daily_return'].dropna() for s in sentiment_order if s in merged_df['sentiment_class'].values]
labels = [s for s in sentiment_order if s in merged_df['sentiment_class'].values]
bp = axes[1,0].boxplot(returns_data, labels=labels, patch_artist=True)
colors_box = ['green', 'gray', 'red']
for patch, color in zip(bp['boxes'], colors_box[:len(returns_data)]):
    patch.set_facecolor(color)
axes[1,0].axhline(y=0, color='black', linewidth=0.5)
axes[1,0].set_title('Returns by Sentiment Category', fontsize=12, fontweight='bold')
axes[1,0].set_ylabel('Daily Return (%)')

# Plot 4: Lag Analysis
if lag_results:
    lags = [r['lag'] for r in lag_results]
    corrs = [r['correlation'] for r in lag_results]
    colors_lag = ['green' if c > 0 else 'red' for c in corrs]
    axes[1,1].bar(lags, corrs, color=colors_lag, edgecolor='black')
    axes[1,1].axhline(y=0, color='black', linewidth=0.5)
    axes[1,1].set_title('Predictive Power by Lag', fontsize=12, fontweight='bold')
    axes[1,1].set_xlabel('Days Lag')
    axes[1,1].set_ylabel('Correlation')

plt.suptitle(f'Sentiment Analysis Report: {primary_stock}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('task3_sentiment_report.png', dpi=150, bbox_inches='tight')
print("✅ Chart saved as 'task3_sentiment_report.png'")
plt.show()

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "=" * 80)
print("📊 FINAL SUMMARY - TASK 3 COMPLETED")
print("=" * 80)

print(f"""
╔════════════════════════════════════════════════════════════════════╗
║                    SENTIMENT ANALYSIS RESULTS                      ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  📝 SENTIMENT DISTRIBUTION:                                        ║
║     • Positive: {sentiment_counts.get('Positive', 0)} ({sentiment_counts.get('Positive', 0)/len(news_df)*100:.1f}%)               ║
║     • Neutral: {sentiment_counts.get('Neutral', 0)} ({sentiment_counts.get('Neutral', 0)/len(news_df)*100:.1f}%)                ║
║     • Negative: {sentiment_counts.get('Negative', 0)} ({sentiment_counts.get('Negative', 0)/len(news_df)*100:.1f}%)               ║
║                                                                    ║
║  📈 CORRELATION ({primary_stock}):                                     ║
║     • Pearson r = {pearson_corr:.4f} ({'significant' if pearson_p < 0.05 else 'not significant'})               ║
║     • Direction: {direction} ({strength})                                 ║
║                                                                    ║
║  💰 TRADING EDGE:                                                  ║
║     • Positive sentiment days: +{sentiment_returns.loc['Positive', 'mean']:.2f}% avg return            ║
║     • Negative sentiment days: {sentiment_returns.loc['Negative', 'mean']:.2f}% avg return             ║
║     • Edge: {sentiment_returns.loc['Positive', 'mean'] - sentiment_returns.loc['Negative', 'mean']:.2f}%              ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")

print("\n✅ TASK 3 COMPLETED SUCCESSFULLY!")
print("📁 Files created: task3_sentiment_report.png")