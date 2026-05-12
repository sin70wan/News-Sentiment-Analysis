"""Unit tests for sentiment analysis module."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.sentiment_analyzer import FinancialSentimentAnalyzer


class TestFinancialSentimentAnalyzer:
    """Test cases for sentiment analyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = FinancialSentimentAnalyzer()
    
    def test_vader_positive_sentiment(self):
       """Test VADER identifies positive sentiment."""
       score = self.analyzer.get_vader_sentiment("Excellent earnings report with massive profit")
       assert score > 0
    
    def test_vader_negative_sentiment(self):
        """Test VADER identifies negative sentiment."""
        score = self.analyzer.get_vader_sentiment("Company reports major loss")
        assert score < 0
    
    def test_empty_headline(self):
        """Test empty headline returns neutral."""
        score = self.analyzer.get_vader_sentiment("")
        assert score == 0.0
    
    def test_classification_positive(self):
        """Test sentiment classification - positive."""
        result = self.analyzer.classify_sentiment(0.1)
        assert result == "Positive"
    
    def test_classification_negative(self):
        """Test sentiment classification - negative."""
        result = self.analyzer.classify_sentiment(-0.1)
        assert result == "Negative"
    
    def test_classification_neutral(self):
        """Test sentiment classification - neutral."""
        result = self.analyzer.classify_sentiment(0.02)
        assert result == "Neutral"
    
    def test_analyze_multiple_headlines(self):
        """Test batch processing of headlines."""
        import pandas as pd
        headlines = pd.Series([
            "Great earnings report",
            "Stock crashes badly",
            "Market remains flat"
        ])
        results = self.analyzer.analyze_headlines(headlines)
        assert len(results) == 3
        assert 'sentiment_vader' in results.columns
        assert 'sentiment_class' in results.columns