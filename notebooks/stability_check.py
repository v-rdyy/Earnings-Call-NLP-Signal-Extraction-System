"""
Stability Check: Performance by Time Period

Evaluates model consistency across different time periods to assess
signal persistence and robustness.
"""

import sys
import os
import pandas as pd
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.load_data import load_transcripts, load_prices, align_transcripts_with_prices
from src.features.nlp_features import add_nlp_features_to_dataframe
from src.models.prepare_data import prepare_modeling_data
from src.models.train_models import create_binary_target, train_gradient_boosting
from src.models.calibration import calibrate_probabilities, apply_calibration
from src.evaluation.evaluate_models import evaluate_probabilistic_predictions

print("=" * 80)
print("STABILITY CHECK: PERFORMANCE BY TIME PERIOD")
print("=" * 80)

print("\nLoading data...")
transcripts_df = load_transcripts('data/raw/real_transcripts.csv')
prices_df = load_prices('data/raw/real_prices.csv')
aligned_df = align_transcripts_with_prices(transcripts_df, prices_df)

print("\nExtracting NLP features...")
features_cache_path = 'data/processed/features_with_nlp.csv'
if os.path.exists(features_cache_path):
    features_df = pd.read_csv(features_cache_path)
    features_df['date'] = pd.to_datetime(features_df['date'])
else:
    features_df = add_nlp_features_to_dataframe(aligned_df)
    os.makedirs('data/processed', exist_ok=True)
    features_df.to_csv(features_cache_path, index=False)

X, y_vol = prepare_modeling_data(features_df, target_column='volatility_change')
features_df_filtered = features_df.loc[X.index].copy()

features_df_filtered = features_df_filtered.sort_values('date')

features_df_filtered['year'] = features_df_filtered['date'].dt.year

volatility_threshold = features_df_filtered['volatility_change'].quantile(0.75)

print(f"\nVolatility spike threshold (75th percentile): {volatility_threshold:.6f}")

print("\n" + "=" * 80)
print("STABILITY ANALYSIS BY YEAR")
print("=" * 80)
print("\nNOTE: This analysis trains and tests on the same year's data.")
print("High ROC-AUC scores indicate in-sample fit, not generalization.")
print("See TIME-SPLIT ANALYSIS below for realistic out-of-time performance.")

yearly_results = []

for year in sorted(features_df_filtered['year'].unique()):
    year_data = features_df_filtered[features_df_filtered['year'] == year].copy()
    
    if len(year_data) < 20:
        print(f"\nSkipping {year}: Insufficient samples ({len(year_data)})")
        continue
    
    X_year = X.loc[year_data.index]
    y_vol_year = year_data['volatility_change']
    y_vol_year_binary = create_binary_target(y_vol_year, threshold=volatility_threshold)
    
    if y_vol_year_binary.sum() < 3 or (len(y_vol_year_binary) - y_vol_year_binary.sum()) < 3:
        print(f"\nSkipping {year}: Insufficient class balance")
        continue
    
    gb_year = train_gradient_boosting(X_year, y_vol_year_binary, X_year)
    gb_year_cal = calibrate_probabilities(y_vol_year_binary, gb_year['train_probs'])
    gb_year_cal_probs = apply_calibration(gb_year_cal['calibrator'], gb_year['test_probs'])
    
    eval_year = evaluate_probabilistic_predictions(y_vol_year_binary, gb_year_cal_probs)
    
    yearly_results.append({
        'Year': year,
        'Samples': len(year_data),
        'Volatility Spikes': y_vol_year_binary.sum(),
        'Spike Rate': y_vol_year_binary.mean(),
        'ROC-AUC': eval_year['roc_auc'],
        'Brier Score': eval_year['brier_score']
    })
    
    print(f"\n{year}: {len(year_data)} samples, ROC-AUC: {eval_year['roc_auc']:.4f}")

if yearly_results:
    results_df = pd.DataFrame(yearly_results)
    print("\n" + "=" * 80)
    print("YEARLY PERFORMANCE SUMMARY")
    print("=" * 80)
    print("\n" + results_df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("STABILITY METRICS")
    print("=" * 80)
    print(f"Mean ROC-AUC: {results_df['ROC-AUC'].mean():.4f}")
    print(f"Std ROC-AUC: {results_df['ROC-AUC'].std():.4f}")
    print(f"Min ROC-AUC: {results_df['ROC-AUC'].min():.4f} ({results_df.loc[results_df['ROC-AUC'].idxmin(), 'Year']})")
    print(f"Max ROC-AUC: {results_df['ROC-AUC'].max():.4f} ({results_df.loc[results_df['ROC-AUC'].idxmax(), 'Year']})")
    
    if results_df['ROC-AUC'].std() < 0.15:
        print("\n✓ In-sample performance appears stable across years (low variance)")
        print("  NOTE: These are in-sample metrics. See TIME-SPLIT ANALYSIS for generalization.")
    else:
        print("\n⚠ In-sample performance shows variability across years")
        print("  NOTE: These are in-sample metrics. See TIME-SPLIT ANALYSIS for generalization.")

print("\n" + "=" * 80)
print("TIME-SPLIT ANALYSIS (Out-of-Time Test)")
print("=" * 80)
print("\nThis is the REAL stability check - training on one period, testing on a later period.")
print("This measures generalization, not just in-sample fit.")

print("\nTraining on 2022-2023, testing on 2024...")

train_data = features_df_filtered[features_df_filtered['year'] <= 2023]
test_data = features_df_filtered[features_df_filtered['year'] == 2024]

if len(train_data) >= 100 and len(test_data) >= 20:
    X_train_split = X.loc[train_data.index]
    X_test_split = X.loc[test_data.index]
    
    y_vol_train_split = train_data['volatility_change']
    y_vol_test_split = test_data['volatility_change']
    
    volatility_threshold_split = y_vol_train_split.quantile(0.75)
    
    y_vol_train_split_binary = create_binary_target(y_vol_train_split, threshold=volatility_threshold_split)
    y_vol_test_split_binary = create_binary_target(y_vol_test_split, threshold=volatility_threshold_split)
    
    print(f"Train: {len(train_data)} samples, {y_vol_train_split_binary.sum()} spikes ({y_vol_train_split_binary.mean():.2%})")
    print(f"Test: {len(test_data)} samples, {y_vol_test_split_binary.sum()} spikes ({y_vol_test_split_binary.mean():.2%})")
    
    gb_split = train_gradient_boosting(X_train_split, y_vol_train_split_binary, X_test_split)
    gb_split_cal = calibrate_probabilities(y_vol_train_split_binary, gb_split['train_probs'])
    gb_split_test_cal = apply_calibration(gb_split_cal['calibrator'], gb_split['test_probs'])
    
    eval_split = evaluate_probabilistic_predictions(y_vol_test_split_binary, gb_split_test_cal)
    
    print(f"\nTest Performance (2024):")
    print(f"  ROC-AUC: {eval_split['roc_auc']:.4f}")
    print(f"  Brier Score: {eval_split['brier_score']:.4f}")
    print(f"  Log Loss: {eval_split['log_loss']:.4f}")
    
    if eval_split['roc_auc'] > 0.50:
        print("\n✓ Signal persists in out-of-time test (2024)")
    else:
        print("\n⚠ Signal degraded in out-of-time test (may be time-sensitive)")
else:
    print("Insufficient data for time-split analysis")
