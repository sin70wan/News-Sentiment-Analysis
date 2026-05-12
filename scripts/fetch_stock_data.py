"""Script to fetch stock price data using yfinance."""

import yfinance as yf
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


def fetch_stock_data(tickers: list, start_date: str, end_date: str, save_csv: bool = True):
    """Fetch stock price data for given tickers."""
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            print(f"Warning: No data found for {ticker}")
            continue
        
        data.reset_index(inplace=True)
        
        if save_csv:
            Path("data/raw").mkdir(parents=True, exist_ok=True)
            filename = f"data/raw/{ticker}_{start_date}_to_{end_date}.csv"
            data.to_csv(filename, index=False)
            print(f"Saved to {filename}")
        
        print(f"Rows: {len(data)}")
    return data


if __name__ == "__main__":
    tickers_to_fetch = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    fetch_stock_data(tickers_to_fetch, "2023-01-01", "2024-12-31")