"""
Test script for data loading and alignment functions.
"""

import sys
import os
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.load_data import (
    load_transcripts,
    load_prices,
    align_transcripts_with_prices
)

print("=" * 60)
print("STEP 1: Loading transcripts...")
print("=" * 60)

transcripts_df = load_transcripts()

print("\n" + "=" * 60)
print("STEP 2: Loading prices...")
print("=" * 60)

prices_df = load_prices()

print("\n" + "=" * 60)
print("STEP 3: Aligning transcripts with prices...")
print("=" * 60)

aligned_df = align_transcripts_with_prices(transcripts_df, prices_df)

print(f"\nAligned {len(aligned_df)} transcripts with price data")

print("\n" + "=" * 60)
print("ALIGNED DATA PREVIEW:")
print("=" * 60)
print(aligned_df.head())

print("\n" + "=" * 60)
print("RETURN STATISTICS:")
print("=" * 60)
print(f"1-day returns:")
print(f"  Mean: {aligned_df['return_1d'].mean():.4f} ({aligned_df['return_1d'].mean()*100:.2f}%)")
print(f"  Std: {aligned_df['return_1d'].std():.4f}")
print(f"  Min: {aligned_df['return_1d'].min():.4f} ({aligned_df['return_1d'].min()*100:.2f}%)")
print(f"  Max: {aligned_df['return_1d'].max():.4f} ({aligned_df['return_1d'].max()*100:.2f}%)")

print(f"\n3-day returns:")
print(f"  Mean: {aligned_df['return_3d'].mean():.4f} ({aligned_df['return_3d'].mean()*100:.2f}%)")
print(f"  Std: {aligned_df['return_3d'].std():.4f}")

print(f"\n5-day returns:")
print(f"  Mean: {aligned_df['return_5d'].mean():.4f} ({aligned_df['return_5d'].mean()*100:.2f}%)")
print(f"  Std: {aligned_df['return_5d'].std():.4f}")

print("\n" + "=" * 60)
print("VOLATILITY STATISTICS:")
print("=" * 60)
volatility_data = aligned_df[aligned_df['pre_volatility'].notna() & aligned_df['post_volatility'].notna()]

if len(volatility_data) > 0:
    print(f"Pre-earnings volatility:")
    print(f"  Mean: {volatility_data['pre_volatility'].mean():.4f}")
    print(f"  Std: {volatility_data['pre_volatility'].std():.4f}")
    
    print(f"\nPost-earnings volatility:")
    print(f"  Mean: {volatility_data['post_volatility'].mean():.4f}")
    print(f"  Std: {volatility_data['post_volatility'].std():.4f}")
    
    print(f"\nVolatility change (post - pre):")
    print(f"  Mean: {volatility_data['volatility_change'].mean():.4f}")
    print(f"  Std: {volatility_data['volatility_change'].std():.4f}")
    print(f"  Positive changes: {(volatility_data['volatility_change'] > 0).sum()} ({(volatility_data['volatility_change'] > 0).sum() / len(volatility_data) * 100:.1f}%)")
    print(f"  Negative changes: {(volatility_data['volatility_change'] < 0).sum()} ({(volatility_data['volatility_change'] < 0).sum() / len(volatility_data) * 100:.1f}%)")
else:
    print("No volatility data available (need at least 10 trading days before and after each call)")

print("\n" + "=" * 60)
print("SAMPLE ALIGNED RECORDS:")
print("=" * 60)
for idx, row in aligned_df.head(3).iterrows():
    print(f"\nTicker: {row['ticker']}")
    print(f"Date: {row['date'].strftime('%Y-%m-%d')}")
    print(f"Reference Price: ${row['reference_price']:.2f}")
    if pd.notna(row['return_1d']):
        print(f"1-day return: {row['return_1d']*100:.2f}%")
    if pd.notna(row['return_3d']):
        print(f"3-day return: {row['return_3d']*100:.2f}%")
    if pd.notna(row['return_5d']):
        print(f"5-day return: {row['return_5d']*100:.2f}%")
    if pd.notna(row['pre_volatility']):
        print(f"Pre-volatility: {row['pre_volatility']:.4f}")
    if pd.notna(row['post_volatility']):
        print(f"Post-volatility: {row['post_volatility']:.4f}")
    if pd.notna(row['volatility_change']):
        change_str = "increased" if row['volatility_change'] > 0 else "decreased"
        print(f"Volatility change: {row['volatility_change']:.4f} ({change_str})")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)

