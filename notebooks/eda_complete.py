"""
COMPLETE EDA FOR FINANCIAL NEWS SENTIMENT ANALYSIS
Task 1: Exploratory Data Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('stopwords', quiet=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

print("=" * 70)
print("📊 COMPLETE EDA - FINANCIAL NEWS SENTIMENT ANALYSIS")
print("=" * 70)

# ============================================
# PART 1: CREATE SAMPLE DATA
# ============================================
print("\n📁 PART 1: Creating sample dataset...")

np.random.seed(42)

# Generate dates for 2024
all_dates = [datetime(2024, 1, 1) + timedelta(days=x) for x in range(366)]

# Stock tickers (popular stocks)
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ']

# Publishers
publishers = ['Reuters', 'Bloomberg', 'CNBC', 'WSJ', 'FT', 'Yahoo Finance', 'MarketWatch', 'Seeking Alpha']

# Headline templates with different sentiments
positive_headlines = [
    'Stock hits all-time high after strong earnings',
    'Analyst upgrades to Buy, raises price target',
    'Company beats revenue estimates by wide margin',
    'New product launch exceeds expectations',
    'Profit margins expand significantly',
    'Board approves $5 billion share buyback',
    'Dividend increased by 15%',
    'Partnership announced with industry leader',
    'Record quarterly revenue reported',
    'Stock upgraded by multiple analysts'
]

negative_headlines = [
    'Stock plunges on disappointing earnings',
    'Analyst downgrades to Sell',
    'Company misses revenue estimates',
    'CEO resigns effective immediately',
    'Regulatory investigation launched',
    'Profit warning issued for next quarter',
    'Lawsuit filed against company',
    'Supply chain disruptions impact production',
    'Credit rating downgraded',
    'Worst quarter in company history'
]

neutral_headlines = [
    'Company announces routine board meeting',
    'Executive to speak at conference',
    'Quarterly report scheduled for next week',
    'Hiring new CFO',
    'Office expansion plans revealed',
    'Patent application filed',
    'Annual shareholder meeting announced',
    'Credit rating reaffirmed',
    'Company hires new marketing director',
    'Office location announced'
]

all_headlines = positive_headlines + negative_headlines + neutral_headlines

# Generate 2000 random articles
data = []
for _ in range(2000):
    headline = np.random.choice(all_headlines)
    # Add some randomness
    if np.random.random() > 0.8:
        headline = headline + f" - {np.random.choice(['Update', 'Breaking', 'Exclusive'])}"
    
    data.append({
        'headline': headline,
        'publisher': np.random.choice(publishers),
        'date': np.random.choice(all_dates),
        'stock': np.random.choice(stocks)
    })

df = pd.DataFrame(data)

# Save to CSV
import os
os.makedirs('data/raw', exist_ok=True)
df.to_csv('data/raw/fnspid.csv', index=False)

print(f"✅ Created {len(df):,} sample articles")
print(f"📅 Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"🏢 Unique stocks: {df['stock'].nunique()}")
print(f"📰 Unique publishers: {df['publisher'].nunique()}")

# ============================================
# PART 2: DATA OVERVIEW
# ============================================
print("\n" + "=" * 70)
print("📊 PART 2: DATA OVERVIEW")
print("=" * 70)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nData types:\n{df.dtypes}")

# ============================================
# PART 3: HEADLINE LENGTH ANALYSIS
# ============================================
print("\n" + "=" * 70)
print("📝 PART 3: HEADLINE LENGTH ANALYSIS")
print("=" * 70)

df['headline_length'] = df['headline'].str.len()

print(f"Mean length: {df['headline_length'].mean():.1f} characters")
print(f"Median length: {df['headline_length'].median():.1f}")
print(f"Standard deviation: {df['headline_length'].std():.1f}")
print(f"Minimum length: {df['headline_length'].min()}")
print(f"Maximum length: {df['headline_length'].max()}")
print(f"25th percentile: {df['headline_length'].quantile(0.25):.0f}")
print(f"75th percentile: {df['headline_length'].quantile(0.75):.0f}")

# ============================================
# PART 4: PUBLISHER ANALYSIS
# ============================================
print("\n" + "=" * 70)
print("📰 PART 4: PUBLISHER ANALYSIS")
print("=" * 70)

publisher_counts = df['publisher'].value_counts()
total_articles = len(df)

print("Top 10 Most Active Publishers:")
for i, (pub, count) in enumerate(publisher_counts.head(10).items(), 1):
    pct = (count / total_articles) * 100
    print(f"{i:2}. {pub[:35]:35} {count:4,} articles ({pct:.1f}%)")

# ============================================
# PART 5: STOCK COVERAGE ANALYSIS
# ============================================
print("\n" + "=" * 70)
print("📈 PART 5: STOCK COVERAGE ANALYSIS")
print("=" * 70)

stock_counts = df['stock'].value_counts()

print("Top 10 Most Covered Stocks:")
for i, (stock, count) in enumerate(stock_counts.head(10).items(), 1):
    pct = (count / total_articles) * 100
    print(f"{i:2}. {stock:6} {count:4,} articles ({pct:.1f}%)")

# ============================================
# PART 6: TIME SERIES ANALYSIS
# ============================================
print("\n" + "=" * 70)
print("⏰ PART 6: TIME SERIES ANALYSIS")
print("=" * 70)

df['date'] = pd.to_datetime(df['date'])
df['publish_date'] = df['date'].dt.date
df['publish_year'] = df['date'].dt.year
df['publish_month'] = df['date'].dt.month
df['publish_day'] = df['date'].dt.day
df['publish_hour'] = df['date'].dt.hour
df['publish_dayofweek'] = df['date'].dt.dayofweek

daily_volume = df.groupby('publish_date').size()

print(f"Date range: {df['publish_date'].min()} to {df['publish_date'].max()}")
print(f"Total unique days: {len(daily_volume)}")
print(f"Average articles per day: {daily_volume.mean():.1f}")
print(f"Maximum articles in a day: {daily_volume.max()}")
print(f"Minimum articles in a day: {daily_volume.min()}")

# Find top news days
print(f"\nTop 5 busiest news days:")
for date, volume in daily_volume.nlargest(5).items():
    print(f"  📅 {date}: {volume} articles")

# ============================================
# PART 7: VOLUME SPIKE DETECTION
# ============================================
print("\n" + "=" * 70)
print("⚠️ PART 7: VOLUME SPIKE DETECTION")
print("=" * 70)

# Calculate rolling statistics
rolling_mean = daily_volume.rolling(window=7, min_periods=1).mean()
rolling_std = daily_volume.rolling(window=7, min_periods=1).std()

# Identify spikes (2 standard deviations above mean)
spike_threshold = rolling_mean + 2 * rolling_std
spike_days = daily_volume[daily_volume > spike_threshold]

print(f"Found {len(spike_days)} days with unusually high publication volume")

if len(spike_days) > 0:
    print("\nTop 5 volume spikes:")
    for date, volume in spike_days.nlargest(5).items():
        normal_avg = rolling_mean[date]
        pct_increase = ((volume - normal_avg) / normal_avg * 100)
        print(f"  📅 {date}: {volume} articles (↑{pct_increase:.0f}% above normal)")

# ============================================
# PART 8: KEYWORD EXTRACTION (TF-IDF)
# ============================================
print("\n" + "=" * 70)
print("🔑 PART 8: KEYWORD EXTRACTION")
print("=" * 70)

# Setup stopwords
stop_words = set(stopwords.words('english'))
financial_stops = {'said', 'says', 'will', 'can', 'may', 'would', 'could', 'also', 'one', 'two', 'three', 'get', 'make', 'company', 'announces'}
stop_words.update(financial_stops)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=20, stop_words=list(stop_words), ngram_range=(1,2))
tfidf_matrix = tfidf.fit_transform(df['headline'].fillna(''))

# Get top keywords
feature_names = tfidf.get_feature_names_out()
tfidf_scores = tfidf_matrix.sum(axis=0).A1
top_indices = tfidf_scores.argsort()[-15:][::-1]

print("\nTop 15 Most Important Keywords/Phrases:")
for i, idx in enumerate(top_indices, 1):
    print(f"{i:2}. {feature_names[idx]:30} (importance: {tfidf_scores[idx]:.4f})")

# ============================================
# PART 9: CREATE VISUALIZATIONS
# ============================================
print("\n" + "=" * 70)
print("📊 PART 9: CREATING VISUALIZATIONS")
print("=" * 70)

fig = plt.figure(figsize=(16, 14))

# Plot 1: Headline Length Distribution
ax1 = plt.subplot(3, 3, 1)
ax1.hist(df['headline_length'], bins=40, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(df['headline_length'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["headline_length"].mean():.0f}')
ax1.axvline(df['headline_length'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["headline_length"].median():.0f}')
ax1.set_title('Headline Length Distribution', fontsize=12, fontweight='bold')
ax1.set_xlabel('Number of Characters')
ax1.set_ylabel('Frequency')
ax1.legend()

# Plot 2: Boxplot of Headline Lengths
ax2 = plt.subplot(3, 3, 2)
bp = ax2.boxplot(df['headline_length'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
ax2.set_title('Boxplot of Headline Lengths', fontsize=12, fontweight='bold')
ax2.set_ylabel('Characters')
ax2.set_xticklabels(['All Headlines'])

# Plot 3: Top Publishers
ax3 = plt.subplot(3, 3, 3)
top_pubs = publisher_counts.head(8)
colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_pubs)))
ax3.barh(range(len(top_pubs)), top_pubs.values, color=colors)
ax3.set_yticks(range(len(top_pubs)))
ax3.set_yticklabels(top_pubs.index)
ax3.set_xlabel('Number of Articles')
ax3.set_title('Top Publishers', fontsize=12, fontweight='bold')
ax3.invert_yaxis()

# Plot 4: Stock Coverage
ax4 = plt.subplot(3, 3, 4)
top_stocks = stock_counts.head(10)
ax4.bar(top_stocks.index, top_stocks.values, color='coral', edgecolor='black')
ax4.set_title('Most Covered Stocks', fontsize=12, fontweight='bold')
ax4.set_xlabel('Stock Symbol')
ax4.set_ylabel('Number of Articles')
ax4.tick_params(axis='x', rotation=45)

# Plot 5: Daily Volume Over Time
ax5 = plt.subplot(3, 3, 5)
ax5.plot(daily_volume.index, daily_volume.values, color='darkgreen', alpha=0.7, linewidth=1)
ax5.fill_between(daily_volume.index, daily_volume.values, alpha=0.3, color='green')
ax5.set_title('Daily News Volume', fontsize=12, fontweight='bold')
ax5.set_xlabel('Date')
ax5.set_ylabel('Number of Articles')
ax5.tick_params(axis='x', rotation=45)

# Plot 6: Volume by Hour
ax6 = plt.subplot(3, 3, 6)
hourly_dist = df['publish_hour'].value_counts().sort_index()
ax6.bar(hourly_dist.index, hourly_dist.values, color='purple', edgecolor='black')
ax6.set_title('Articles by Hour of Day', fontsize=12, fontweight='bold')
ax6.set_xlabel('Hour (0-23)')
ax6.set_ylabel('Number of Articles')

# Plot 7: Volume by Day of Week
ax7 = plt.subplot(3, 3, 7)
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dow_dist = df['publish_dayofweek'].value_counts().sort_index()
ax7.bar(day_names, dow_dist.values, color='teal', edgecolor='black')
ax7.set_title('Articles by Day of Week', fontsize=12, fontweight='bold')
ax7.set_xlabel('Day')
ax7.set_ylabel('Number of Articles')

# Plot 8: Volume Spikes Detection
ax8 = plt.subplot(3, 3, 8)
ax8.plot(daily_volume.index, daily_volume.values, alpha=0.6, label='Daily Volume', color='blue', linewidth=0.8)
ax8.plot(rolling_mean.index, rolling_mean, color='orange', label='7-day Avg', linewidth=1.5)
if len(spike_days) > 0:
    ax8.scatter(spike_days.index, spike_days.values, color='red', s=30, zorder=5, label='Spikes')
ax8.set_title('Volume Spike Detection', fontsize=12, fontweight='bold')
ax8.set_xlabel('Date')
ax8.set_ylabel('Number of Articles')
ax8.legend(fontsize=8)
ax8.tick_params(axis='x', rotation=45)

# Plot 9: Top Keywords Bar Chart
ax9 = plt.subplot(3, 3, 9)
top_keywords = [feature_names[i] for i in top_indices[:10]]
top_scores = [tfidf_scores[i] for i in top_indices[:10]]
y_pos = range(len(top_keywords))
ax9.barh(y_pos, top_scores, color='steelblue')
ax9.set_yticks(y_pos)
ax9.set_yticklabels(top_keywords)
ax9.invert_yaxis()
ax9.set_xlabel('TF-IDF Score')
ax9.set_title('Top 10 Keywords/Phrases', fontsize=12, fontweight='bold')

plt.suptitle('Financial News EDA Report', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('eda_complete_report.png', dpi=150, bbox_inches='tight')
print("✅ Main visualization saved as 'eda_complete_report.png'")

# ============================================
# PART 10: FINAL SUMMARY
# ============================================
print("\n" + "=" * 70)
print("📊 FINAL SUMMARY REPORT")
print("=" * 70)

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    EDA COMPLETED SUCCESSFULLY                    ║
╚══════════════════════════════════════════════════════════════════╝

📁 DATASET OVERVIEW:
   • Total articles: {len(df):,}
   • Unique stocks: {df['stock'].nunique()}
   • Unique publishers: {df['publisher'].nunique()}
   • Date range: {df['publish_date'].min()} to {df['publish_date'].max()}

📝 HEADLINE ANALYSIS:
   • Average length: {df['headline_length'].mean():.1f} characters
   • Shortest: {df['headline_length'].min()} chars
   • Longest: {df['headline_length'].max()} chars

⏰ TEMPORAL PATTERNS:
   • Peak hour: {hourly_dist.idxmax()}:00 ({hourly_dist.max():,} articles)
   • Busiest day: {day_names[dow_dist.idxmax()]} ({dow_dist.max():,} articles)
   • Busiest month: Most articles in month {df['publish_month'].mode()[0]}

📰 PUBLISHER INSIGHTS:
   • Most active: {publisher_counts.index[0]} ({publisher_counts.iloc[0]:,} articles)
   • Top 5 publishers share: {(publisher_counts.head(5).sum()/len(df)*100):.1f}%

📈 STOCK COVERAGE:
   • Most covered: {stock_counts.index[0]} ({stock_counts.iloc[0]:,} articles)
   • Top 10 stocks share: {(stock_counts.head(10).sum()/len(df)*100):.1f}%

⚠️ VOLUME SPIKES:
   • Number of spike days: {len(spike_days)}
   • Largest spike: {spike_days.max() if len(spike_days) > 0 else 'None'} articles

🔑 TOP KEYWORDS:
   • Most important: {feature_names[top_indices[0]]} (score: {tfidf_scores[top_indices[0]]:.4f})
""")

print("\n" + "=" * 70)
print("✅ TASK 1 - EDA COMPLETED!")
print("=" * 70)
print("\n📁 Files created:")
print("   • data/raw/fnspid.csv - Sample dataset")
print("   • eda_complete_report.png - Complete visualization report")
print("\n➡️ Ready to proceed to Task 2: Technical Indicators")