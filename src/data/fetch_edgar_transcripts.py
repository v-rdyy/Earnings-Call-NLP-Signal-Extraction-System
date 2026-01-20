"""
SEC EDGAR Transcript Fetcher

This module fetches earnings call transcripts from SEC EDGAR using sec-edgar-downloader.
"""

import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
import re
from sec_edgar_downloader import Downloader


def get_ticker_to_cik():
    """
    Get mapping of ticker symbols to CIK numbers.
    
    Returns:
    --------
    dict
        Dictionary mapping ticker to CIK
    """
    
    ticker_to_cik = {
        'AAPL': '0000320193', 'MSFT': '0000789019', 'GOOGL': '0001652044',
        'AMZN': '0001018724', 'META': '0001326801', 'TSLA': '0001318605',
        'NVDA': '0001045810', 'JPM': '0000019617', 'V': '0001403161',
        'JNJ': '0000200406', 'WMT': '0000104169', 'PG': '0000080424',
        'MA': '0001141391', 'UNH': '0000731766', 'HD': '0000354950',
        'DIS': '0001001039', 'NFLX': '0001065280', 'BAC': '0000070858',
        'XOM': '0000034088', 'CVX': '0000093410', 'ABBV': '0001551152',
        'PFE': '0000078003', 'KO': '0000021344', 'PEP': '0000077476',
        'TMO': '0000975554', 'COST': '0000909832', 'AVGO': '0001730168',
        'MRK': '0000310158', 'ABT': '0000001800', 'ACN': '0001467373',
        'CSCO': '0000858877', 'ADBE': '0000796343', 'NKE': '0000320187',
        'TXN': '0000097476', 'CMCSA': '0001166691', 'PM': '0001413329',
        'NEE': '0000753308', 'LIN': '0001707925', 'HON': '0000773840',
        'UPS': '0001090727', 'RTX': '000101829'
    }
    
    return ticker_to_cik


def extract_transcript_section(content):
    """
    Extract meaningful content from SEC filing.
    
    Extracts text from EX-99 exhibits or any substantial narrative content.
    Since 8-K filings rarely contain full transcripts, we extract press releases
    and other narrative content that may be useful for analysis.
    """
    
    content_lower = content.lower()
    
    transcript_keywords = [
        'earnings call', 'conference call', 'q&a', 'question and answer',
        'operator', 'good morning', 'good afternoon', 'thank you for',
        'prepared remarks', 'opening remarks'
    ]
    
    has_transcript = any(keyword in content_lower for keyword in transcript_keywords)
    
    if has_transcript:
        text = content
    else:
        if '<TYPE>EX-99' in content or 'ex-99' in content_lower or 'exhibit 99' in content_lower:
            text = extract_ex99_content(content)
        else:
            text = extract_narrative_content(content)
    
    if len(text) > 5000:
        lines = text.split('\n')
        start_idx = 0
        end_idx = len(lines)
        
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in ['conference call', 'earnings call', 'operator', 'exhibit 99']):
                start_idx = max(0, i - 10)
                break
        
        for i in range(len(lines) - 1, start_idx, -1):
            if any(keyword in lines[i].lower() for keyword in ['end of call', 'conclusion', 'thank you', '</text>', '</document>']):
                end_idx = i + 10
                break
        
        text = '\n'.join(lines[start_idx:end_idx])
    
    return text


def extract_ex99_content(content):
    """
    Extract content from EX-99 exhibits (press releases, etc.)
    """
    ex99_pattern = r'<TYPE>EX-99[^>]*>.*?<TEXT>(.*?)</TEXT>'
    matches = re.findall(ex99_pattern, content, re.DOTALL | re.IGNORECASE)
    
    if matches:
        return '\n\n'.join(matches)
    return content


def extract_narrative_content(content):
    """
    Extract narrative text content from the filing, excluding XBRL tags.
    """
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(content, 'html.parser')
    text = soup.get_text(separator='\n')
    
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 20]
    
    return '\n'.join(lines)


def extract_transcript_from_filing(file_path):
    """
    Extract transcript text from a downloaded SEC filing.
    
    Parameters:
    -----------
    file_path : str
        Path to the filing file
        
    Returns:
    --------
    str
        Extracted transcript text
    """
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        text = extract_transcript_section(content)
        text = clean_transcript_text(text)
        return text
        
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


def clean_transcript_text(text):
    """
    Clean extracted transcript text.
    
    Parameters:
    -----------
    text : str
        Raw extracted text
        
    Returns:
    --------
    str
        Cleaned text
    """
    
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'Page \d+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = text.strip()
    
    return text


def fetch_transcript_for_ticker(downloader, ticker, cik, start_date=None, end_date=None, max_filings=5):
    """
    Fetch earnings call transcript for a single ticker.
    
    Parameters:
    -----------
    downloader : sec_edgar_downloader.Downloader
        Configured downloader instance
    ticker : str
        Stock ticker symbol
    cik : str
        CIK number for the company
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format
    max_filings : int
        Maximum number of filings to download
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: ticker, date, transcript_text
    """
    
    print(f"\nFetching transcripts for {ticker} (CIK: {cik})...")
    
    try:
        if start_date and end_date:
            downloader.get("8-K", cik, after=start_date, before=end_date)
        else:
            downloader.get("8-K", cik)
        
        transcripts = []
        
        base_dir = "sec-edgar-filings"
        ticker_dir = os.path.join(base_dir, cik, "8-K")
        
        if not os.path.exists(ticker_dir):
            print(f"  No filings downloaded for {ticker}")
            return pd.DataFrame(columns=['ticker', 'date', 'transcript_text'])
        
        for root, dirs, files in os.walk(ticker_dir):
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
            print(f"Successfully fetched {len(df)} transcripts for {ticker}")
            return df
        else:
            print(f"No transcripts found for {ticker}")
            return pd.DataFrame(columns=['ticker', 'date', 'transcript_text'])
            
    except Exception as e:
        print(f"Error fetching transcripts for {ticker}: {e}")
        return pd.DataFrame(columns=['ticker', 'date', 'transcript_text'])


def fetch_transcripts_for_tickers(tickers, start_date=None, end_date=None, 
                                  max_filings_per_ticker=3, email='research@example.com'):
    """
    Fetch earnings call transcripts for multiple tickers.
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format
    max_filings_per_ticker : int
        Maximum filings to download per ticker
    email : str
        Email for SEC EDGAR (required by library)
        
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with all transcripts
    """
    
    ticker_to_cik = get_ticker_to_cik()
    
    downloader = Downloader("Earnings Call NLP Project", email)
    
    all_transcripts = []
    
    for ticker in tickers:
        cik = ticker_to_cik.get(ticker.upper())
        if not cik:
            print(f"Warning: CIK not found for {ticker}, skipping...")
            continue
        
        ticker_transcripts = fetch_transcript_for_ticker(
            downloader, ticker, cik, start_date, end_date, max_filings_per_ticker
        )
        
        if not ticker_transcripts.empty:
            all_transcripts.append(ticker_transcripts)
        
        time.sleep(1)
    
    if all_transcripts:
        combined = pd.concat(all_transcripts, ignore_index=True)
        combined = combined.sort_values('date')
        print(f"\nTotal transcripts fetched: {len(combined)}")
        return combined
    else:
        print("\nNo transcripts fetched")
        return pd.DataFrame(columns=['ticker', 'date', 'transcript_text'])
