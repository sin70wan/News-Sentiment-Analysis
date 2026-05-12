"""Technical indicators module for stock price analysis."""

import pandas as pd
import numpy as np


class TechnicalIndicators:
    """Compute technical indicators for stock price data."""
    
    @staticmethod
    def calculate_sma(data: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, window: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return data.ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(data: pd.Series):
        """Calculate MACD (12, 26, 9)."""
        ema_12 = TechnicalIndicators.calculate_ema(data, 12)
        ema_26 = TechnicalIndicators.calculate_ema(data, 26)
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        return {'macd': macd_line, 'signal': signal_line, 'histogram': histogram}
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, window: int = 20, num_std: int = 2):
        """Calculate Bollinger Bands."""
        sma = TechnicalIndicators.calculate_sma(data, window)
        std = data.rolling(window=window).std()
        return {'upper': sma + (std * num_std), 'middle': sma, 'lower': sma - (std * num_std)}
    
    @staticmethod
    def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals."""
        signals = pd.DataFrame(index=df.index)
        signals['position'] = 0
        if 'rsi' in df.columns:
            signals.loc[df['rsi'] < 30, 'position'] = 1
            signals.loc[df['rsi'] > 70, 'position'] = -1
        return signals