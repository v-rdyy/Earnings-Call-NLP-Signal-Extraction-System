from datetime import timedelta
import pandas as pd
import os

def load_transcripts(file_path='data/raw/real_transcripts.csv'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Transcripts file not found: {file_path}")

    df = pd.read_csv(file_path)

    df['date'] = pd.to_datetime(df['date'])

    print(f"Loaded {len(df)} transcripts from {file_path}")
    return df

def load_prices(file_path='data/raw/real_prices.csv'):
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

        ticker_prices = prices_df[prices_df['ticker'] == ticker].sort_values('date').reset_index(drop=True)
        call_date_idx = ticker_prices[ticker_prices['date'] == call_date].index
        
        if len(call_date_idx) == 0:
            continue
        
        call_date_pos = ticker_prices.index.get_loc(call_date_idx[0])
        
        return_1d = None
        return_3d = None
        return_5d = None
        
        if call_date_pos + 1 < len(ticker_prices):
            price_1d = ticker_prices.iloc[call_date_pos + 1]['close_price']
            return_1d = (price_1d - reference_price) / reference_price
        
        if call_date_pos + 3 < len(ticker_prices):
            price_3d = ticker_prices.iloc[call_date_pos + 3]['close_price']
            return_3d = (price_3d - reference_price) / reference_price
        
        if call_date_pos + 5 < len(ticker_prices):
            price_5d = ticker_prices.iloc[call_date_pos + 5]['close_price']
            return_5d = (price_5d - reference_price) / reference_price

        pre_volatility = None
        post_volatility = None
        volatility_change = None

        if call_date_pos >= 10:
            pre_window = ticker_prices.iloc[call_date_pos - 10 : call_date_pos + 1]
            pre_returns = pre_window['close_price'].pct_change().dropna()
            if len(pre_returns) > 0:
                pre_volatility = pre_returns.std()

        if call_date_pos + 10 < len(ticker_prices):
            post_window = ticker_prices.iloc[call_date_pos:call_date_pos + 11]
            post_returns = post_window['close_price'].pct_change().dropna()
            if len(post_returns) > 0:
                post_volatility = post_returns.std()

        if pre_volatility is not None and post_volatility is not None:
            volatility_change = post_volatility - pre_volatility
        
        aligned_data.append({
            'ticker': ticker,
            'date': call_date,
            'transcript_text': transcript_text,
            'reference_price': reference_price,
            'return_1d': return_1d,
            'return_3d': return_3d,
            'return_5d': return_5d,
            'pre_volatility': pre_volatility,
            'post_volatility': post_volatility,
            'volatility_change': volatility_change
        })
    
    df = pd.DataFrame(aligned_data)
    return df