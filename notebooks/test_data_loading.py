"""
Test script for data loading and alignment functions.
"""

import sys
import os

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
print("LABEL DISTRIBUTION:")
print("=" * 60)
print(f"Label 0 (price down or same): {(aligned_df['label'] == 0).sum()}")
print(f"Label 1 (price up): {(aligned_df['label'] == 1).sum()}")

print("\n" + "=" * 60)
print("SAMPLE ALIGNED RECORDS:")
print("=" * 60)
for idx, row in aligned_df.head(3).iterrows():
    print(f"\nTicker: {row['ticker']}")
    print(f"Date: {row['date'].strftime('%Y-%m-%d')}")
    print(f"Reference Price: ${row['reference_price']:.2f}")
    print(f"Target Price: ${row['target_price']:.2f}")
    print(f"Label: {row['label']} ({'UP' if row['label'] == 1 else 'DOWN'})")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)

