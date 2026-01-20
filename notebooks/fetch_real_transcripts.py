"""
Fetch Real Earnings Call Transcripts from SEC EDGAR

This script fetches ~100 earnings call transcripts from SEC EDGAR
and saves them for use in the pipeline.
"""

import sys
import os
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.fetch_edgar_transcripts import fetch_transcripts_for_tickers
from src.data.load_real_data import prepare_real_data_pipeline, fetch_prices_for_tickers
from src.data.load_data import align_transcripts_with_prices

print("=" * 80)
print("FETCHING REAL EARNINGS CALL TRANSCRIPTS FROM SEC EDGAR")
print("=" * 80)

print("\nSelecting tickers for ~100 transcripts...")
print("Using major companies with frequent earnings calls")

tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM',
    'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD', 'DIS', 'NFLX',
    'BAC', 'XOM', 'CVX', 'ABBV', 'PFE', 'KO', 'PEP', 'TMO',
    'COST', 'AVGO', 'MRK', 'ABT', 'ACN', 'CSCO', 'ADBE', 'NKE',
    'TXN', 'CMCSA', 'PM', 'NEE', 'LIN', 'HON', 'UPS', 'RTX'
]

print(f"Target tickers: {len(tickers)} companies")
print(f"Target: ~100 transcripts (~2-3 per company)")

print("\n" + "=" * 80)
print("STEP 1: Fetch Transcripts from SEC EDGAR")
print("=" * 80)
print("\nNote: This may take 10-20 minutes due to rate limiting")
print("SEC EDGAR requires delays between requests")

start_date = '2023-01-01'
end_date = '2024-12-31'

print("\nNote: SEC EDGAR requires an email address for API access")
print("Update the email parameter in the function call below with your email")

transcripts_df = fetch_transcripts_for_tickers(
    tickers,
    start_date=start_date,
    end_date=end_date,
    max_filings_per_ticker=3,
    email='mythvardy@gmail.com'
)

if transcripts_df.empty:
    print("\nError: No transcripts fetched. Check your internet connection and try again.")
    sys.exit(1)

print(f"\nFetched {len(transcripts_df)} transcripts")

print("\n" + "=" * 80)
print("STEP 2: Save Transcripts")
print("=" * 80)

os.makedirs('data/raw', exist_ok=True)
transcripts_path = 'data/raw/real_transcripts.csv'
transcripts_df.to_csv(transcripts_path, index=False)
print(f"Saved transcripts to {transcripts_path}")

print("\n" + "=" * 80)
print("STEP 3: Fetch Stock Prices")
print("=" * 80)

print("Fetching prices for all tickers in date range...")
unique_tickers = transcripts_df['ticker'].unique().tolist()
date_range_start = (transcripts_df['date'].min() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
date_range_end = (transcripts_df['date'].max() + pd.Timedelta(days=30)).strftime('%Y-%m-%d')

prices_df = fetch_prices_for_tickers(unique_tickers, date_range_start, date_range_end)

if prices_df.empty:
    print("\nError: No prices fetched. Check your internet connection and try again.")
    sys.exit(1)

print(f"Fetched {len(prices_df)} price records")

print("\n" + "=" * 80)
print("STEP 4: Save Prices")
print("=" * 80)

prices_path = 'data/raw/real_prices.csv'
prices_df.to_csv(prices_path, index=False)
print(f"Saved prices to {prices_path}")

print("\n" + "=" * 80)
print("STEP 5: Align Transcripts with Prices")
print("=" * 80)

aligned_df = align_transcripts_with_prices(transcripts_df, prices_df)

print(f"\nAligned {len(aligned_df)} transcripts with price data")

if len(aligned_df) < 50:
    print("\nWarning: Less than 50 aligned transcripts. You may want to fetch more.")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Transcripts fetched: {len(transcripts_df)}")
print(f"Price records: {len(prices_df)}")
print(f"Successfully aligned: {len(aligned_df)}")
print(f"\nFiles saved:")
print(f"  - {transcripts_path}")
print(f"  - {prices_path}")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("""
1. Review the fetched transcripts:
   - Check data/raw/real_transcripts.csv
   - Verify transcript quality

2. Run the complete pipeline with real data:
   - Update notebooks/run_complete_pipeline.py to use real data
   - Or create a new script that loads from real_transcripts.csv

3. Analyze results with real earnings call data!
""")
