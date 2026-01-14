"""
Test script for NLP feature extraction.
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
from src.features.nlp_features import add_nlp_features_to_dataframe

print("=" * 60)
print("STEP 1: Loading and aligning data...")
print("=" * 60)

transcripts_df = load_transcripts()
prices_df = load_prices()
aligned_df = align_transcripts_with_prices(transcripts_df, prices_df)

print(f"Loaded {len(aligned_df)} aligned transcripts")

print("\n" + "=" * 60)
print("STEP 2: Extracting NLP features...")
print("=" * 60)

features_df = add_nlp_features_to_dataframe(aligned_df)

print(f"Extracted NLP features for {len(features_df)} transcripts")

print("\n" + "=" * 60)
print("ENHANCED DATA PREVIEW:")
print("=" * 60)
print(features_df.head())

print("\n" + "=" * 60)
print("NLP FEATURES SUMMARY:")
print("=" * 60)

print("\nSentiment Features:")
print(f"  Document sentiment - Mean: {features_df['doc_sentiment'].mean():.4f}, Std: {features_df['doc_sentiment'].std():.4f}")
print(f"  Sentence mean - Mean: {features_df['sent_mean'].mean():.4f}, Std: {features_df['sent_mean'].std():.4f}")
print(f"  Sentence variance - Mean: {features_df['sent_variance'].mean():.4f}, Std: {features_df['sent_variance'].std():.4f}")

print("\nUncertainty Features:")
print(f"  Uncertainty count - Mean: {features_df['uncertainty_count'].mean():.2f}, Max: {features_df['uncertainty_count'].max()}")
print(f"  Uncertainty density - Mean: {features_df['uncertainty_density'].mean():.2f}, Max: {features_df['uncertainty_density'].max():.2f}")

print("\nForward/Backward Looking Features:")
print(f"  Forward density - Mean: {features_df['forward_density'].mean():.2f}, Max: {features_df['forward_density'].max():.2f}")
print(f"  Backward density - Mean: {features_df['backward_density'].mean():.2f}, Max: {features_df['backward_density'].max():.2f}")
print(f"  Forward/Backward ratio - Mean: {features_df['forward_backward_ratio'].mean():.2f}, Max: {features_df['forward_backward_ratio'].max():.2f}")

print("\n" + "=" * 60)
print("SAMPLE RECORD WITH ALL FEATURES:")
print("=" * 60)

sample = features_df.iloc[0]
print(f"\nTicker: {sample['ticker']}")
print(f"Date: {sample['date'].strftime('%Y-%m-%d')}")
print(f"\nNLP Features:")
print(f"  Document sentiment: {sample['doc_sentiment']:.4f}")
print(f"  Sentence mean: {sample['sent_mean']:.4f}")
print(f"  Sentence variance: {sample['sent_variance']:.4f}")
print(f"  Uncertainty density: {sample['uncertainty_density']:.2f}")
print(f"  Forward density: {sample['forward_density']:.2f}")
print(f"  Backward density: {sample['backward_density']:.2f}")
print(f"  Forward/Backward ratio: {sample['forward_backward_ratio']:.2f}")

print("\n" + "=" * 60)
print("COLUMN NAMES:")
print("=" * 60)
print(f"Total columns: {len(features_df.columns)}")
print(f"Original columns: ticker, date, transcript_text, reference_price, return_1d, return_3d, return_5d, pre_volatility, post_volatility, volatility_change")
print(f"NLP feature columns: {', '.join([col for col in features_df.columns if col not in ['ticker', 'date', 'transcript_text', 'reference_price', 'return_1d', 'return_3d', 'return_5d', 'pre_volatility', 'post_volatility', 'volatility_change']])}")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)

