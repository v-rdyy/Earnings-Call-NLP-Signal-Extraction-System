"""
Real Data Loading Module

This module loads real earnings call transcripts and stock price data.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os


def fetch_stock_prices(ticker, start_date, end_date):
    """
    Fetch real stock price data from Yahoo Finance.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL')
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: date, ticker, close_price
    """
    
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            print(f"Warning: No data found for {ticker} from {start_date} to {end_date}")
            return pd.DataFrame(columns=['date', 'ticker', 'close_price'])
        
        data = data.reset_index()
        data['date'] = pd.to_datetime(data['Date']).dt.date
        data['ticker'] = ticker
        data['close_price'] = data['Close']
        
        result = data[['date', 'ticker', 'close_price']].copy()
        result['date'] = pd.to_datetime(result['date'])
        
        print(f"Fetched {len(result)} price records for {ticker}")
        return result
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame(columns=['date', 'ticker', 'close_price'])


def fetch_prices_for_tickers(tickers, start_date, end_date):
    """
    Fetch stock prices for multiple tickers.
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
        
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with prices for all tickers
    """
    
    all_prices = []
    
    for ticker in tickers:
        prices = fetch_stock_prices(ticker, start_date, end_date)
        if not prices.empty:
            all_prices.append(prices)
    
    if all_prices:
        combined = pd.concat(all_prices, ignore_index=True)
        print(f"\nFetched prices for {len(tickers)} tickers: {len(combined)} total records")
        return combined
    else:
        print("Warning: No price data fetched")
        return pd.DataFrame(columns=['date', 'ticker', 'close_price'])


def load_transcripts_from_csv(file_path):
    """
    Load transcripts from a CSV file.
    
    Expected CSV format:
    - ticker: Stock ticker symbol
    - date: Earnings call date (YYYY-MM-DD)
    - transcript_text: Full transcript text
    
    Parameters:
    -----------
    file_path : str
        Path to CSV file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: ticker, date, transcript_text
    """
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Transcripts file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    required_columns = ['ticker', 'date', 'transcript_text']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Loaded {len(df)} transcripts from {file_path}")
    return df


def load_transcripts_from_pdf_directory(pdf_directory):
    """
    Load transcripts from PDF files in a directory.
    
    Each PDF should be named with ticker and date information.
    Example: AAPL_2023-01-15.pdf
    
    Parameters:
    -----------
    pdf_directory : str
        Directory containing PDF files
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: ticker, date, transcript_text
    """
    
    try:
        import PyPDF2
    except ImportError:
        raise ImportError("PyPDF2 required for PDF loading. Install with: pip install PyPDF2")
    
    if not os.path.exists(pdf_directory):
        raise FileNotFoundError(f"PDF directory not found: {pdf_directory}")
    
    transcripts = []
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        
        try:
            filename = os.path.splitext(pdf_file)[0]
            parts = filename.split('_')
            
            if len(parts) >= 2:
                ticker = parts[0]
                date_str = parts[1]
                
                try:
                    date = pd.to_datetime(date_str)
                except:
                    print(f"Warning: Could not parse date from {pdf_file}, skipping")
                    continue
                
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                
                transcripts.append({
                    'ticker': ticker,
                    'date': date,
                    'transcript_text': text
                })
                
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            continue
    
    if transcripts:
        df = pd.DataFrame(transcripts)
        print(f"Loaded {len(df)} transcripts from PDF directory")
        return df
    else:
        print("Warning: No transcripts loaded from PDF directory")
        return pd.DataFrame(columns=['ticker', 'date', 'transcript_text'])


def prepare_real_data_pipeline(transcripts_df, prices_df=None, tickers=None, 
                               start_date=None, end_date=None):
    """
    Prepare complete data pipeline with real data.
    
    If prices_df is not provided, fetches prices using yfinance.
    
    Parameters:
    -----------
    transcripts_df : pd.DataFrame
        DataFrame with transcripts (ticker, date, transcript_text)
    prices_df : pd.DataFrame, optional
        Pre-loaded prices. If None, will fetch from yfinance
    tickers : list, optional
        List of tickers to fetch prices for (if prices_df not provided)
    start_date : str, optional
        Start date for price fetching (if prices_df not provided)
    end_date : str, optional
        End date for price fetching (if prices_df not provided)
        
    Returns:
    --------
    tuple
        (transcripts_df, prices_df) ready for alignment
    """
    
    if prices_df is None:
        if tickers is None:
            tickers = transcripts_df['ticker'].unique().tolist()
        
        if start_date is None:
            start_date = transcripts_df['date'].min() - timedelta(days=30)
            start_date = start_date.strftime('%Y-%m-%d')
        
        if end_date is None:
            end_date = transcripts_df['date'].max() + timedelta(days=30)
            end_date = end_date.strftime('%Y-%m-%d')
        
        print(f"Fetching prices for tickers: {tickers}")
        print(f"Date range: {start_date} to {end_date}")
        prices_df = fetch_prices_for_tickers(tickers, start_date, end_date)
    
    return transcripts_df, prices_df
