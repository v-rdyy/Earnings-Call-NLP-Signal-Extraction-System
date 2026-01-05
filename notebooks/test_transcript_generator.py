"""
Test script for the transcript generator functions.

This script tests that we can generate and save sample transcripts.
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.generate_sample_data import generate_and_save_sample_data
import pandas as pd

print("=" * 60)
print("Testing complete data generation pipeline...")
print("=" * 60)

transcripts_df, prices_df = generate_and_save_sample_data(num_calls=10)

print("\n" + "=" * 60)
print("TRANSCRIPTS PREVIEW:")
print("=" * 60)
print(transcripts_df.head())

print("\n" + "=" * 60)
print("PRICES PREVIEW:")
print("=" * 60)
print(prices_df.head(10))

print("\n" + "=" * 60)
print("VERIFICATION:")
print("=" * 60)

transcripts_path = 'data/raw/sample_transcripts.csv'
prices_path = 'data/raw/sample_prices.csv'

if os.path.exists(transcripts_path):
    print(f"Transcripts file exists: {transcripts_path}")
    loaded_transcripts = pd.read_csv(transcripts_path)
    print(f"  Loaded {len(loaded_transcripts)} rows")
else:
    print(f"Transcripts file not found: {transcripts_path}")

if os.path.exists(prices_path):
    print(f"Prices file exists: {prices_path}")
    loaded_prices = pd.read_csv(prices_path)
    print(f"  Loaded {len(loaded_prices)} rows")
else:
    print(f"Prices file not found: {prices_path}")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)