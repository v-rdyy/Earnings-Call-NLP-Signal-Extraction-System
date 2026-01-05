from datetime import timedelta
import pandas as pd
import os

def load_transcripts(file_path='data/raw/sample_transcripts.csv'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Transcripts file not found: {file_path}")

    df = pd.read_csv(file_path)

    df['date'] = pd.to_datetime(df['date'])

    print(f"Loaded {len(df)} transcripts from {file_path}")
    return df

def load_prices(file_path='data/raw/sample_prices.csv'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prices file not found: {file_path}")

    df = pd.read_csv(file_path)

    df['date'] = pd.to_datetime(df['date'])

    print(f"Loaded {len(df)} price records from {file_path}")
    return df

def get_next_trading_day(date):
    next_day = date + timedelta(days=1)

    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    
    return next_day

def align_transcripts_with_prices(transcripts_df, prices_df):
    aligned_data = []

    for _, row in transcripts_df.iterrows():
        ticker = row['ticker']
        call_date = row['date']
        transcript_text = row['transcript_text']

        call_date_price = prices_df[
            (prices_df['ticker'] == ticker) &
            (prices_df['date'] == call_date)
        ]

        if len(call_date_price) == 0:
            continue

        reference_price = call_date_price['close_price'].iloc[0]

        next_trading_day = get_next_trading_day(call_date)

        next_day_price = prices_df[
            (prices_df['ticker'] == ticker) & 
            (prices_df['date'] == next_trading_day)
        ]

        if len(next_day_price) == 0:
            continue

        target_price = next_day_price['close_price'].iloc[0]

        label = 1 if target_price > reference_price else 0

        aligned_data.append({
            'ticker': ticker,
            'date': call_date,
            'transcript_text': transcript_text,
            'reference_price': reference_price,
            'target_price': target_price,
            'label': label
        })
    
    df = pd.DataFrame(aligned_data)
    return df