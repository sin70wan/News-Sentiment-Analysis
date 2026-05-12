"""Sentiment analysis module for financial news headlines."""

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from typing import Dict, Tuple


class FinancialSentimentAnalyzer:
    """Analyze sentiment of financial news headlines."""
    
    def __init__(self):
        """Initialize sentiment analyzers."""
        self.vader_analyzer = SentimentIntensityAnalyzer()
    
    def get_vader_sentiment(self, text: str) -> float:
        """
        Get VADER sentiment score (-1 to 1).
        VADER is preferred for financial/social text.
        """
        if pd.isna(text) or text == "":
            return 0.0
        return self.vader_analyzer.polarity_scores(str(text))['compound']
    
    def get_textblob_sentiment(self, text: str) -> float:
        """Get TextBlob polarity score (-1 to 1)."""
        if pd.isna(text) or text == "":
            return 0.0
        return TextBlob(str(text)).sentiment.polarity
    
    def classify_sentiment(self, score: float, threshold: float = 0.05) -> str:
        """Classify sentiment as Positive, Neutral, or Negative."""
        if score > threshold:
            return "Positive"
        elif score < -threshold:
            return "Negative"
        else:
            return "Neutral"
    
    def analyze_headlines(self, headlines: pd.Series) -> pd.DataFrame:
        """
        Analyze multiple headlines.
        
        Args:
            headlines: Series of headline strings
            
        Returns:
            DataFrame with sentiment scores and classifications
        """
        results = []
        for headline in headlines:
            vader_score = self.get_vader_sentiment(headline)
            textblob_score = self.get_textblob_sentiment(headline)
            results.append({
                'headline': headline,
                'sentiment_vader': vader_score,
                'sentiment_textblob': textblob_score,
                'sentiment_class': self.classify_sentiment(vader_score)
            })
        return pd.DataFrame(results)