"""
Process Downloaded SEC EDGAR Filings

This script processes the already-downloaded SEC filings and extracts transcripts.
"""

import sys
import os
import pandas as pd
import re
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.fetch_edgar_transcripts import (
    get_ticker_to_cik,
    extract_transcript_from_filing,
    clean_transcript_text
)
from src.data.load_real_data import fetch_prices_for_tickers
from src.data.load_data import align_transcripts_with_prices

def process_downloaded_filings():
    """
    Process all downloaded SEC filings and extract transcripts.
    """
    
    ticker_to_cik = get_ticker_to_cik()
    cik_to_ticker = {v: k for k, v in ticker_to_cik.items()}
    
    base_dir = "sec-edgar-filings"
    
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} directory not found")
        return pd.DataFrame(columns=['ticker', 'date', 'transcript_text'])
    
    transcripts = []
    
    for cik_dir in os.listdir(base_dir):
        cik_path = os.path.join(base_dir, cik_dir)
        if not os.path.isdir(cik_path):
            continue
        
        ticker = cik_to_ticker.get(cik_dir)
        if not ticker:
            continue
        
        filing_type_dir = os.path.join(cik_path, "8-K")
        if not os.path.exists(filing_type_dir):
            continue
        
        print(f"\nProcessing {ticker} (CIK: {cik_dir})...")
        
        for root, dirs, files in os.walk(filing_type_dir):
            for file in files:
                if file.endswith(('.txt', '.htm', '.html')):
                    file_path = os.path.join(root, file)
                    
                    transcript_text = extract_transcript_from_filing(file_path)
                    
                    if len(transcript_text) > 500:
                        date_obj = None
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                header = f.read(5000)
                            
                            date_match = re.search(r'CONFORMED PERIOD OF REPORT:\s+(\d{8})', header)
                            if date_match:
                                date_str_raw = date_match.group(1)
                                date_str = f"{date_str_raw[:4]}-{date_str_raw[4:6]}-{date_str_raw[6:]}"
                                try:
                                    date_obj = pd.to_datetime(date_str)
                                except:
                                    pass
                        except:
                            pass
                        
                        if date_obj is None:
                            path_parts = root.split(os.sep)
                            for part in path_parts:
                                accession_match = re.search(r'(\d{10})-(\d{2})-(\d{6})', part)
                                if accession_match:
                                    year_short = accession_match.group(2)
                                    year = f"20{year_short}"
                                    try:
                                        date_obj = pd.to_datetime(f"{year}-01-01")
                                    except:
                                        pass
                                    break
                        
                        if date_obj is not None:
                            transcripts.append({
                                'ticker': ticker,
                                'date': date_obj,
                                'transcript_text': transcript_text
                            })
                            print(f"  Found transcript for {date_obj.strftime('%Y-%m-%d')}")
    
    if transcripts:
        df = pd.DataFrame(transcripts)
        df = df.drop_duplicates(subset=['ticker', 'date'])
        df = df.sort_values('date')
        print(f"\nTotal transcripts extracted: {len(df)}")
        return df
    else:
        print("\nNo transcripts extracted")
        return pd.DataFrame(columns=['ticker', 'date', 'transcript_text'])

print("=" * 80)
print("PROCESSING DOWNLOADED SEC FILINGS")
print("=" * 80)

print("\nExtracting transcripts from downloaded files...")
transcripts_df = process_downloaded_filings()

if transcripts_df.empty:
    print("\nNo transcripts found. The 8-K filings may not contain full transcripts.")
    print("Consider checking the files manually or using a different data source.")
    sys.exit(1)

print(f"\nExtracted {len(transcripts_df)} transcripts")

print("\n" + "=" * 80)
print("SAVING TRANSCRIPTS")
print("=" * 80)

os.makedirs('data/raw', exist_ok=True)
transcripts_path = 'data/raw/real_transcripts.csv'
transcripts_df.to_csv(transcripts_path, index=False)
print(f"Saved transcripts to {transcripts_path}")

print("\n" + "=" * 80)
print("FETCHING STOCK PRICES")
print("=" * 80)

unique_tickers = transcripts_df['ticker'].unique().tolist()
date_range_start = (transcripts_df['date'].min() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
date_range_end = (transcripts_df['date'].max() + pd.Timedelta(days=30)).strftime('%Y-%m-%d')

print(f"Fetching prices for {len(unique_tickers)} tickers")
print(f"Date range: {date_range_start} to {date_range_end}")

prices_df = fetch_prices_for_tickers(unique_tickers, date_range_start, date_range_end)

if prices_df.empty:
    print("\nError: No prices fetched")
    sys.exit(1)

print(f"Fetched {len(prices_df)} price records")

print("\n" + "=" * 80)
print("SAVING PRICES")
print("=" * 80)

prices_path = 'data/raw/real_prices.csv'
prices_df.to_csv(prices_path, index=False)
print(f"Saved prices to {prices_path}")

print("\n" + "=" * 80)
print("ALIGNING TRANSCRIPTS WITH PRICES")
print("=" * 80)

aligned_df = align_transcripts_with_prices(transcripts_df, prices_df)

print(f"\nAligned {len(aligned_df)} transcripts with price data")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Transcripts extracted: {len(transcripts_df)}")
print(f"Price records: {len(prices_df)}")
print(f"Successfully aligned: {len(aligned_df)}")
print(f"\nFiles saved:")
print(f"  - {transcripts_path}")
print(f"  - {prices_path}")
