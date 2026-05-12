"""Unit tests for sentiment analysis module."""

import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.sentiment_analyzer import FinancialSentimentAnalyzer


class TestFinancialSentimentAnalyzer:
    def setup_method(self):
        self.analyzer = FinancialSentimentAnalyzer()
    
    def test_vader_positive_sentiment(self):
        score = self.analyzer.get_vader_sentiment("Stock hits all time high")
        assert score > 0
    
    def test_vader_negative_sentiment(self):
        score = self.analyzer.get_vader_sentiment("Company reports major loss")
        assert score < 0
    
    def test_empty_headline(self):
        score = self.analyzer.get_vader_sentiment("")
        assert score == 0.0
    
    def test_classification_positive(self):
        result = self.analyzer.classify_sentiment(0.1)
        assert result == "Positive"
    
    def test_classification_negative(self):
        result = self.analyzer.classify_sentiment(-0.1)
        assert result == "Negative"
    
    def test_classification_neutral(self):
        result = self.analyzer.classify_sentiment(0.02)
        assert result == "Neutral"