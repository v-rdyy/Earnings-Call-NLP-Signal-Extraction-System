"""
Sample Data Generator for Earnings Call NLP Project

This module generates sample earnings call transcripts for testing.
It's designed so we can easily swap in real PDF data later.
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import os


def generate_sample_transcripts(num_calls=20):
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

    positive_templates = [
        "We are pleased to report strong quarterly results. Revenue exceeded expectations.",
        "This quarter demonstrated exceptional performance. We achieved record revenue.",
        "We delivered outstanding results. Strong execution positions us well for growth."
    ]
    
    negative_templates = [
        "We faced headwinds this quarter that impacted our results. Revenue declined.",
        "This quarter's results were below expectations. We experienced margin pressure.",
        "We encountered challenges this quarter. Revenue growth slowed significantly."
    ]

    data = []

    start_date = datetime(2023, 1, 1)

    for i in range(num_calls):
        ticker = np.random.choice(tickers)
        call_date = start_date + timedelta(days=i*45)
        is_positive = np.random.choice([True, False])

        if is_positive:
            transcript = np.random.choice(positive_templates)
        else:
            transcript = np.random.choice(negative_templates)

        call_data = {
            'ticker': ticker,
            'date': call_date.strftime('%Y-%m-%d'),
            'transcript_text': transcript
        }

        data.append(call_data)

    df = pd.DataFrame(data)

    return df

def generate_sample_prices(tickers, start_date, end_date):
    data = []

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    base_prices = {'AAPL': 150, 'MSFT': 300, 'GOOGL': 100, 'AMZN': 120, 'TSLA': 200}

    current_date = start
    while current_date <= end:
        if current_date.weekday() < 5:
            for ticker in tickers:
                base = base_prices.get(ticker, 100)
                price = base + np.random.normal(0,5)
                price = max(price, 10)

                data.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'ticker': ticker,
                    'close_price': round(price, 2)
                })
                
        current_date += timedelta(days=1)

    df = pd.DataFrame(data)
    return df


def save_transcripts_to_csv(transcripts_df, file_path='data/raw/sample_transcripts.csv'):
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    transcripts_df.to_csv(file_path, index=False)
    print(f"Saved {len(transcripts_df)} transcripts to {file_path}")

def save_prices_to_csv(prices_df, file_path='data/raw/sample_prices.csv'):
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    prices_df.to_csv(file_path, index=False)
    print(f"Saved {len(prices_df)} price records to {file_path}")

def generate_and_save_sample_data(num_calls=20):
    transcripts_df = generate_sample_transcripts(num_calls=num_calls)
    dates = pd.to_datetime(transcripts_df['date'])
    start_date = dates.min().strftime('%Y-%m-%d')
    end_date = (dates.max() + timedelta(days=10)).strftime('%Y-%m-%d')

    tickers = transcripts_df['ticker'].unique().tolist()

    prices_df = generate_sample_prices(tickers, start_date, end_date)

    save_transcripts_to_csv(transcripts_df)
    save_prices_to_csv(prices_df)

    print("\nSample data generation complete!")
    print(f"  - Transcripts: {len(transcripts_df)} calls")
    print(f"  - Price records: {len(prices_df)} days")

    return transcripts_df, prices_df

def load_transcripts_from_pdf(pdf_path):
    raise NotImplementedError(
        "PDF loading not yet implemented. "
        "Use generate_sample_transcripts() for now, or implement PDF parsing here."
    )