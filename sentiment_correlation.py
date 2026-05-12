"""
TASK 3: SENTIMENT ANALYSIS & CORRELATION WITH STOCK RETURNS
Analyzes news sentiment and its relationship with stock price movements
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("📊 TASK 3: SENTIMENT ANALYSIS & CORRELATION")
print("=" * 70)

# ============================================
# PART 1: CREATE SAMPLE NEWS DATA WITH SENTIMENT
# ============================================
print("\n📁 PART 1: Creating news dataset with sentiment...")

np.random.seed(42)

# Create dates for last 3 months
end_date = datetime.now().date()
start_date = end_date - timedelta(days=90)
all_dates = [start_date + timedelta(days=x) for x in range(91)]

# Stock to analyze
stock_symbol = "AAPL"

# Headlines with sentiment labels
positive_headlines = [
    "Apple beats earnings expectations", "Apple stock hits all-time high",
    "iPhone sales surge", "Apple announces record revenue",
    "Analyst upgrades Apple to Buy", "Apple's new product launch successful",
    "Apple services revenue grows 20%", "Apple wins patent case"
]

negative_headlines = [
    "Apple faces supply chain issues", "Apple stock drops on weak guidance",
    "iPhone sales miss estimates", "Apple faces antitrust investigation",
    "Analyst downgrades Apple", "Apple loses market share",
    "Apple production cuts announced", "Legal challenges for Apple"
]

neutral_headlines = [
    "Apple announces board meeting", "Apple hires new executive",
    "Apple store opening in new location", "Apple releases software update",
    "Apple partners with supplier", "Apple research and development update"
]

# Generate random news articles
news_data = []
for _ in range(200):
    date = np.random.choice(all_dates)
    # More news on weekdays
    if date.weekday() < 5:
        num_articles = np.random.randint(1, 5)
    else:
        num_articles = np.random.randint(0, 2)
    
    for _ in range(num_articles):
        sentiment_type = np.random.choice(['positive', 'negative', 'neutral'], p=[0.4, 0.3, 0.3])
        if sentiment_type == 'positive':
            headline = np.random.choice(positive_headlines)
        elif sentiment_type == 'negative':
            headline = np.random.choice(negative_headlines)
        else:
            headline = np.random.choice(neutral_headlines)
        
        news_data.append({
            'date': date,
            'headline': headline,
            'stock': stock_symbol
        })

news_df = pd.DataFrame(news_data)
print(f"✅ Created {len(news_df)} news articles")

# ============================================
# PART 2: SENTIMENT ANALYSIS
# ============================================
print("\n" + "=" * 70)
print("📝 PART 2: Sentiment Analysis with VADER")
print("=" * 70)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    """Get sentiment score using VADER"""
    return analyzer.polarity_scores(str(text))['compound']

def classify_sentiment(score):
    """Classify sentiment into categories"""
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis
news_df['sentiment_score'] = news_df['headline'].apply(get_sentiment_score)
news_df['sentiment_class'] = news_df['sentiment_score'].apply(classify_sentiment)

# Display sample results
print("\nSample headlines with sentiment scores:")
print("-" * 60)
for i, row in news_df.head(10).iterrows():
    print(f"[{row['sentiment_class']:8}] ({row['sentiment_score']:+.2f}) {row['headline'][:50]}")

# Sentiment distribution
print("\n📊 Sentiment Distribution:")
sentiment_counts = news_df['sentiment_class'].value_counts()
for sentiment, count in sentiment_counts.items():
    pct = (count / len(news_df)) * 100
    print(f"   {sentiment}: {count} articles ({pct:.1f}%)")

# ============================================
# PART 3: FETCH STOCK PRICE DATA
# ============================================
print("\n" + "=" * 70)
print("📈 PART 3: Fetching Stock Price Data")
print("=" * 70)

# Fetch stock data for the same period
stock = yf.Ticker(stock_symbol)
stock_df = stock.history(start=start_date, end=end_date + timedelta(days=1))

if stock_df.empty:
    print(f"⚠️ No stock data found for {stock_symbol}")
    exit()

print(f"✅ Downloaded {len(stock_df)} days of stock data")
print(f"📅 Date range: {stock_df.index[0].date()} to {stock_df.index[-1].date()}")

# Calculate daily returns
stock_df['daily_return'] = stock_df['Close'].pct_change() * 100
print(f"📊 Average daily return: {stock_df['daily_return'].mean():.2f}%")
print(f"📊 Volatility (std dev): {stock_df['daily_return'].std():.2f}%")

# ============================================
# PART 4: AGGREGATE DAILY SENTIMENT
# ============================================
print("\n" + "=" * 70)
print("📊 PART 4: Aggregating Daily Sentiment")
print("=" * 70)

# Group by date and calculate average sentiment
daily_sentiment = news_df.groupby('date').agg({
    'sentiment_score': 'mean',
    'sentiment_class': lambda x: x.mode()[0] if len(x) > 0 else 'Neutral'
}).reset_index()

daily_sentiment.columns = ['date', 'avg_sentiment', 'dominant_sentiment']
print(f"✅ Aggregated {len(daily_sentiment)} days with news coverage")

# Merge with stock data
stock_df['date'] = stock_df.index.date
merged_df = pd.merge(stock_df, daily_sentiment, on='date', how='left')

# Fill missing sentiment with neutral
merged_df['avg_sentiment'] = merged_df['avg_sentiment'].fillna(0)
merged_df['dominant_sentiment'] = merged_df['dominant_sentiment'].fillna('Neutral')

print(f"✅ Merged dataset: {len(merged_df)} trading days")

# ============================================
# PART 5: CORRELATION ANALYSIS
# ============================================
print("\n" + "=" * 70)
print("📊 PART 5: Correlation Analysis")
print("=" * 70)

# Remove NaN returns for correlation
corr_df = merged_df.dropna(subset=['daily_return'])

# Calculate Pearson correlation
pearson_corr, pearson_p = pearsonr(corr_df['avg_sentiment'], corr_df['daily_return'])

# Calculate Spearman correlation (more robust for non-linear relationships)
spearman_corr, spearman_p = spearmanr(corr_df['avg_sentiment'], corr_df['daily_return'])

print(f"\n📈 Correlation Results for {stock_symbol}:")
print("-" * 40)
print(f"Pearson Correlation:  {pearson_corr:.4f} (p-value: {pearson_p:.4e})")
print(f"Spearman Correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4e})")

# Interpretation
print(f"\n📝 Interpretation:")
if abs(pearson_corr) < 0.2:
    strength = "very weak"
elif abs(pearson_corr) < 0.4:
    strength = "weak"
elif abs(pearson_corr) < 0.6:
    strength = "moderate"
elif abs(pearson_corr) < 0.8:
    strength = "strong"
else:
    strength = "very strong"

direction = "positive" if pearson_corr > 0 else "negative"
print(f"   There is a {strength} {direction} correlation between sentiment and daily returns.")

if pearson_p < 0.05:
    print(f"   The correlation is STATISTICALLY SIGNIFICANT (p < 0.05)")
else:
    print(f"   The correlation is NOT statistically significant")

# Returns by sentiment category
print(f"\n📊 Average Returns by Sentiment Category:")
sentiment_returns = merged_df.groupby('dominant_sentiment')['daily_return'].agg(['mean', 'std', 'count'])
for sentiment in ['Positive', 'Neutral', 'Negative']:
    if sentiment in sentiment_returns.index:
        mean_ret = sentiment_returns.loc[sentiment, 'mean']
        count = sentiment_returns.loc[sentiment, 'count']
        print(f"   {sentiment}: {mean_ret:+.3f}% (n={count})")

# ============================================
# PART 6: LAG ANALYSIS (Predictive Power)
# ============================================
print("\n" + "=" * 70)
print("📊 PART 6: Lag Analysis (Sentiment Predicting Future Returns)")
print("=" * 70)

lag_results = []
for lag in range(1, 6):
    # Shift returns backward to test if sentiment predicts future returns
    corr_df[f'future_return_lag{lag}'] = corr_df['daily_return'].shift(-lag)
    valid = corr_df.dropna(subset=[f'future_return_lag{lag}'])
    
    if len(valid) > 10:
        lag_corr, lag_p = pearsonr(valid['avg_sentiment'], valid[f'future_return_lag{lag}'])
        lag_results.append({'lag': lag, 'correlation': lag_corr, 'p_value': lag_p, 'n': len(valid)})

lag_df = pd.DataFrame(lag_results)
print(f"\n   Lag (days) | Correlation | p-value | Sample Size")
print(f"   {'-' * 50}")
for _, row in lag_df.iterrows():
    sig = "✓" if row['p_value'] < 0.05 else "✗"
    print(f"   {row['lag']:10} | {row['correlation']:+.4f}     | {row['p_value']:.4f} {sig} | {row['n']}")

best_lag = lag_df.loc[lag_df['correlation'].abs().idxmax()] if len(lag_df) > 0 else None
if best_lag is not None:
    print(f"\n💡 Best predictive power at {int(best_lag['lag'])}-day lag: correlation = {best_lag['correlation']:.4f}")

# ============================================
# PART 7: VISUALIZATIONS
# ============================================
print("\n" + "=" * 70)
print("📊 PART 7: Creating Visualizations")
print("=" * 70)

fig = plt.figure(figsize=(16, 14))

# Plot 1: Sentiment Distribution
ax1 = plt.subplot(3, 3, 1)
sentiment_counts.plot(kind='bar', ax=ax1, color=['green', 'gray', 'red'])
ax1.set_title('Sentiment Distribution', fontsize=12, fontweight='bold')
ax1.set_xlabel('Sentiment')
ax1.set_ylabel('Number of Articles')
ax1.tick_params(axis='x', rotation=0)

# Plot 2: Sentiment vs Returns Scatter
ax2 = plt.subplot(3, 3, 2)
ax2.scatter(corr_df['avg_sentiment'], corr_df['daily_return'], alpha=0.5, color='steelblue')
z = np.polyfit(corr_df['avg_sentiment'], corr_df['daily_return'], 1)
p = np.poly1d(z)
x_line = np.linspace(corr_df['avg_sentiment'].min(), corr_df['avg_sentiment'].max(), 100)
ax2.plot(x_line, p(x_line), color='red', linewidth=2, label=f'Trend (slope: {z[0]:.3f})')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_title(f'Sentiment vs Daily Returns (r = {pearson_corr:.3f})', fontsize=12, fontweight='bold')
ax2.set_xlabel('Sentiment Score')
ax2.set_ylabel('Daily Return (%)')
ax2.legend()

# Plot 3: Returns by Sentiment (Boxplot)
ax3 = plt.subplot(3, 3, 3)
sentiment_order = ['Positive', 'Neutral', 'Negative']
box_data = [merged_df[merged_df['dominant_sentiment'] == s]['daily_return'].dropna() for s in sentiment_order if s in merged_df['dominant_sentiment'].values]
bp = ax3.boxplot(box_data, labels=[s for s in sentiment_order if s in merged_df['dominant_sentiment'].values], patch_artist=True)
for patch, color in zip(bp['boxes'], ['green', 'gray', 'red']):
    patch.set_facecolor(color)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.set_title('Returns Distribution by Sentiment', fontsize=12, fontweight='bold')
ax3.set_ylabel('Daily Return (%)')

# Plot 4: Time Series - Price and Sentiment
ax4 = plt.subplot(3, 3, 4)
ax4_twin = ax4.twinx()
ax4.plot(merged_df['date'], merged_df['Close'], color='black', linewidth=1.5, label='Stock Price')
ax4_twin.fill_between(merged_df['date'], 0, merged_df['avg_sentiment'], alpha=0.3, color='blue', label='Sentiment')
ax4_twin.plot(merged_df['date'], merged_df['avg_sentiment'], color='blue', alpha=0.5, linewidth=0.8)
ax4.set_title('Stock Price vs Sentiment Over Time', fontsize=12, fontweight='bold')
ax4.set_xlabel('Date')
ax4.set_ylabel('Price ($)', color='black')
ax4_twin.set_ylabel('Sentiment Score', color='blue')
ax4.tick_params(axis='x', rotation=45)

# Plot 5: Sentiment Score Distribution
ax5 = plt.subplot(3, 3, 5)
ax5.hist(news_df['sentiment_score'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
ax5.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
ax5.set_title('Sentiment Score Distribution', fontsize=12, fontweight='bold')
ax5.set_xlabel('Sentiment Score')
ax5.set_ylabel('Frequency')

# Plot 6: Lag Correlation
if len(lag_df) > 0:
    ax6 = plt.subplot(3, 3, 6)
    colors = ['green' if c > 0 else 'red' for c in lag_df['correlation']]
    ax6.bar(lag_df['lag'], lag_df['correlation'], color=colors, edgecolor='black')
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax6.set_title('Sentiment Predictive Power by Lag', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Days Lag (Sentiment → Future Return)')
    ax6.set_ylabel('Correlation')
    for i, row in lag_df.iterrows():
        ax6.text(row['lag'], row['correlation'] + (0.02 if row['correlation'] >= 0 else -0.05),
                f"p={row['p_value']:.3f}", ha='center', fontsize=8)

# Plot 7: Daily Sentiment Over Time
ax7 = plt.subplot(3, 3, 7)
daily_sent_avg = news_df.groupby('date')['sentiment_score'].mean()
ax7.plot(daily_sent_avg.index, daily_sent_avg.values, color='green', linewidth=1)
ax7.fill_between(daily_sent_avg.index, -1, 1, alpha=0.1, color='gray')
ax7.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
ax7.set_title('Average Daily Sentiment', fontsize=12, fontweight='bold')
ax7.set_xlabel('Date')
ax7.set_ylabel('Average Sentiment Score')
ax7.tick_params(axis='x', rotation=45)

# Plot 8: Returns Distribution
ax8 = plt.subplot(3, 3, 8)
ax8.hist(merged_df['daily_return'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='coral')
ax8.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
ax8.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
ax8.set_xlabel('Daily Return (%)')
ax8.set_ylabel('Frequency')

# Plot 9: Summary Statistics Table
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
stats_text = f"""
SUMMARY STATISTICS - {stock_symbol}

News Data:
• Total articles: {len(news_df):,}
• Positive: {sentiment_counts.get('Positive', 0)} ({sentiment_counts.get('Positive', 0)/len(news_df)*100:.0f}%)
• Neutral: {sentiment_counts.get('Neutral', 0)} ({sentiment_counts.get('Neutral', 0)/len(news_df)*100:.0f}%)
• Negative: {sentiment_counts.get('Negative', 0)} ({sentiment_counts.get('Negative', 0)/len(news_df)*100:.0f}%)

Correlation:
• Pearson: {pearson_corr:.4f}
• Spearman: {spearman_corr:.4f}
• p-value: {pearson_p:.4e}

Returns by Sentiment:
• Positive days: {sentiment_returns.loc['Positive', 'mean']:+.3f}%
• Neutral days: {sentiment_returns.loc['Neutral', 'mean']:+.3f}%
• Negative days: {sentiment_returns.loc['Negative', 'mean']:+.3f}%
"""
ax9.text(0.1, 0.9, stats_text, transform=ax9.transAxes, fontsize=9, verticalalignment='top', fontfamily='monospace')
ax9.set_title('Key Findings', fontsize=12, fontweight='bold')

plt.suptitle(f'Sentiment Analysis Report: {stock_symbol}', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('sentiment_correlation_report.png', dpi=150, bbox_inches='tight')
print("✅ Chart saved as 'sentiment_correlation_report.png'")

plt.show()

# ============================================
# PART 8: FINAL SUMMARY
# ============================================
print("\n" + "=" * 70)
print("📊 TASK 3 - FINAL SUMMARY")
print("=" * 70)

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    SENTIMENT ANALYSIS RESULTS                    ║
╚══════════════════════════════════════════════════════════════════╝

📝 SENTIMENT INSIGHTS:
   • Analyzed {len(news_df)} news articles for {stock_symbol}
   • {sentiment_counts.get('Positive', 0)} positive ({sentiment_counts.get('Positive', 0)/len(news_df)*100:.0f}%)
   • {sentiment_counts.get('Neutral', 0)} neutral ({sentiment_counts.get('Neutral', 0)/len(news_df)*100:.0f}%)
   • {sentiment_counts.get('Negative', 0)} negative ({sentiment_counts.get('Negative', 0)/len(news_df)*100:.0f}%)

📈 CORRELATION FINDINGS:
   • Pearson correlation: {pearson_corr:.4f}
   • This is a {strength} {direction} relationship
   • {'Statistically significant' if pearson_p < 0.05 else 'Not statistically significant'}

💰 TRADING IMPLICATIONS:
   • Positive sentiment days → {sentiment_returns.loc['Positive', 'mean']:+.2f}% avg return
   • Negative sentiment days → {sentiment_returns.loc['Negative', 'mean']:+.2f}% avg return
   • Difference of {sentiment_returns.loc['Positive', 'mean'] - sentiment_returns.loc['Negative', 'mean']:.2f}%

🎯 RECOMMENDATION:
   {('Sentiment can be used as a trading signal' if abs(pearson_corr) > 0.3 else 'Sentiment alone is not a strong predictor - combine with technical analysis')}
""")

print("\n" + "=" * 70)
print("✅ TASK 3 - SENTIMENT ANALYSIS & CORRELATION COMPLETED!")
print("=" * 70)
print("\n📁 Files created:")
print("   • sentiment_correlation_report.png - Complete visualization report")