"""
Complete Pipeline with Real Data

This script runs the complete pipeline using real earnings call transcripts
and stock prices from SEC EDGAR and yfinance.
"""

import sys
import os
import pandas as pd
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.load_data import (
    load_transcripts,
    load_prices,
    align_transcripts_with_prices
)
from src.features.nlp_features import add_nlp_features_to_dataframe
from src.models.prepare_data import (
    prepare_modeling_data,
    time_aware_train_test_split
)
from src.models.train_models import (
    create_binary_target,
    train_logistic_regression,
    train_gradient_boosting,
    get_feature_importance
)
from src.models.calibration import (
    measure_calibration,
    calibrate_probabilities,
    apply_calibration
)
from src.evaluation.evaluate_models import (
    evaluate_binary_predictions,
    evaluate_probabilistic_predictions,
    evaluate_by_probability_buckets,
    evaluate_volatility_by_signal
)

print("=" * 80)
print("EARNINGS CALL NLP SIGNAL EXTRACTION - REAL DATA PIPELINE")
print("=" * 80)

print("\n" + "=" * 80)
print("STEP 1: Load Real Data")
print("=" * 80)

try:
    transcripts_df = load_transcripts('data/raw/real_transcripts.csv')
except FileNotFoundError:
    print("Error: real_transcripts.csv not found")
    print("Run notebooks/fetch_real_transcripts.py first to download data")
    sys.exit(1)

try:
    prices_df = load_prices('data/raw/real_prices.csv')
except FileNotFoundError:
    print("Error: real_prices.csv not found")
    print("Prices will be fetched automatically...")
    from src.data.load_real_data import fetch_prices_for_tickers
    
    unique_tickers = transcripts_df['ticker'].unique().tolist()
    date_range_start = (transcripts_df['date'].min() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    date_range_end = (transcripts_df['date'].max() + pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    
    prices_df = fetch_prices_for_tickers(unique_tickers, date_range_start, date_range_end)
    prices_df.to_csv('data/raw/real_prices.csv', index=False)

aligned_df = align_transcripts_with_prices(transcripts_df, prices_df)

print(f"\nAligned {len(aligned_df)} earnings calls with price data")

print("\n" + "=" * 80)
print("STEP 2: Extract NLP Features")
print("=" * 80)

features_cache_path = 'data/processed/features_with_nlp.csv'

if os.path.exists(features_cache_path):
    print(f"Loading cached NLP features from {features_cache_path}...")
    features_df = pd.read_csv(features_cache_path)
    features_df['date'] = pd.to_datetime(features_df['date'])
    print(f"Loaded {len(features_df)} transcripts with cached features")
else:
    print("Extracting NLP features (this may take a few minutes)...")
    features_df = add_nlp_features_to_dataframe(aligned_df)
    
    os.makedirs('data/processed', exist_ok=True)
    features_df.to_csv(features_cache_path, index=False)
    print(f"Extracted NLP features for {len(features_df)} transcripts")
    print(f"Cached features to {features_cache_path} for faster future runs")

print("\n" + "=" * 80)
print("STEP 3: Prepare Data for Modeling")
print("=" * 80)

target_column = 'volatility_change'
X, y = prepare_modeling_data(features_df, target_column=target_column)

print(f"\nPrepared {len(X)} samples with {len(X.columns)} features")
print(f"Target: {target_column} (Post-earnings volatility change)")
print(f"Target statistics:")
print(f"  Mean: {y.mean():.6f}")
print(f"  Std: {y.std():.6f}")
print(f"  Min: {y.min():.6f}, Max: {y.max():.6f}")

print("\n" + "=" * 80)
print("STEP 4: Time-Aware Train/Test Split")
print("=" * 80)

dates = features_df.loc[X.index, 'date']
X_train, X_test, y_train, y_test = time_aware_train_test_split(
    X, y, dates, test_size=0.2
)

print("\n" + "=" * 80)
print("STEP 5: Create Binary Targets (Volatility Spike Detection)")
print("=" * 80)

volatility_threshold = y_train.quantile(0.75)
print(f"Volatility spike threshold (75th percentile of training data): {volatility_threshold:.6f}")
print(f"Predicting: volatility_change > {volatility_threshold:.6f} (volatility spike)")

y_train_binary = create_binary_target(y_train, threshold=volatility_threshold)
y_test_binary = create_binary_target(y_test, threshold=volatility_threshold)

print(f"Training: {y_train_binary.sum()} volatility spikes ({y_train_binary.mean():.2%})")
print(f"Test: {y_test_binary.sum()} volatility spikes ({y_test_binary.mean():.2%})")

print("\n" + "=" * 80)
print("STEP 6: Train Models")
print("=" * 80)

print("\n--- Logistic Regression ---")
lr_results = train_logistic_regression(X_train, y_train_binary, X_test)

print("\n--- Gradient Boosting ---")
gb_results = train_gradient_boosting(X_train, y_train_binary, X_test)

print("\n" + "=" * 80)
print("STEP 6.5: Feature Importance Analysis")
print("=" * 80)

print("\n--- Logistic Regression Feature Importance (Top 10) ---")
lr_importance = get_feature_importance(
    lr_results['model'], 
    X_train.columns.tolist(), 
    model_type='logistic_regression'
)
print(lr_importance.head(10).to_string(index=False))

print("\n--- Gradient Boosting Feature Importance (Top 10) ---")
gb_importance = get_feature_importance(
    gb_results['model'], 
    X_train.columns.tolist(), 
    model_type='gradient_boosting'
)
print(gb_importance.head(10).to_string(index=False))

print("\n" + "=" * 80)
print("STEP 7: Measure Calibration (Before)")
print("=" * 80)

print("\n--- Logistic Regression ---")
lr_cal_before = measure_calibration(y_train_binary, lr_results['train_probs'])
print(f"Calibration error: {lr_cal_before['calibration_error']:.4f}")

print("\n--- Gradient Boosting ---")
gb_cal_before = measure_calibration(y_train_binary, gb_results['train_probs'])
print(f"Calibration error: {gb_cal_before['calibration_error']:.4f}")

print("\n" + "=" * 80)
print("STEP 8: Calibrate Probabilities")
print("=" * 80)

print("\n--- Logistic Regression ---")
lr_calibrated = calibrate_probabilities(y_train_binary, lr_results['train_probs'], method='isotonic')
lr_test_calibrated = apply_calibration(lr_calibrated['calibrator'], lr_results['test_probs'], method='isotonic')

print("\n--- Gradient Boosting ---")
gb_calibrated = calibrate_probabilities(y_train_binary, gb_results['train_probs'], method='isotonic')
gb_test_calibrated = apply_calibration(gb_calibrated['calibrator'], gb_results['test_probs'], method='isotonic')

print("\n" + "=" * 80)
print("STEP 9: Evaluate Models")
print("=" * 80)

print("\n" + "-" * 80)
print("LOGISTIC REGRESSION - BINARY PREDICTIONS")
print("-" * 80)
lr_binary_pred = (lr_results['test_probs'] > 0.5).astype(int)
lr_binary_eval = evaluate_binary_predictions(y_test_binary, lr_binary_pred)

print("\n" + "-" * 80)
print("LOGISTIC REGRESSION - PROBABILISTIC PREDICTIONS (Uncalibrated)")
print("-" * 80)
lr_prob_eval = evaluate_probabilistic_predictions(y_test_binary, lr_results['test_probs'])

print("\n" + "-" * 80)
print("LOGISTIC REGRESSION - PROBABILISTIC PREDICTIONS (Calibrated)")
print("-" * 80)
lr_cal_prob_eval = evaluate_probabilistic_predictions(y_test_binary, lr_test_calibrated)

print("\n" + "-" * 80)
print("GRADIENT BOOSTING - BINARY PREDICTIONS")
print("-" * 80)
gb_binary_pred = (gb_results['test_probs'] > 0.5).astype(int)
gb_binary_eval = evaluate_binary_predictions(y_test_binary, gb_binary_pred)

print("\n" + "-" * 80)
print("GRADIENT BOOSTING - PROBABILISTIC PREDICTIONS (Uncalibrated)")
print("-" * 80)
gb_prob_eval = evaluate_probabilistic_predictions(y_test_binary, gb_results['test_probs'])

print("\n" + "-" * 80)
print("GRADIENT BOOSTING - PROBABILISTIC PREDICTIONS (Calibrated)")
print("-" * 80)
gb_cal_prob_eval = evaluate_probabilistic_predictions(y_test_binary, gb_test_calibrated)

print("\n" + "=" * 80)
print("STEP 10: Probability Bucket Analysis (Event Study)")
print("=" * 80)

print("\n--- Logistic Regression (Calibrated) ---")
lr_buckets = evaluate_by_probability_buckets(y_test_binary, lr_test_calibrated)

print("\n--- Gradient Boosting (Calibrated) ---")
gb_buckets = evaluate_by_probability_buckets(y_test_binary, gb_test_calibrated)

print("\n" + "=" * 80)
print("STEP 11: Absolute Return Magnitude Analysis")
print("=" * 80)

test_indices = X_test.index
test_abs_return = np.abs(features_df.loc[test_indices, 'return_1d'].values)

print("\n--- Logistic Regression (Calibrated) ---")
print("Absolute Return Magnitude by Signal Strength:")
lr_abs_return = evaluate_volatility_by_signal(
    y_test_binary, test_abs_return, lr_test_calibrated
)

print("\n--- Gradient Boosting (Calibrated) ---")
print("Absolute Return Magnitude by Signal Strength:")
gb_abs_return = evaluate_volatility_by_signal(
    y_test_binary, test_abs_return, gb_test_calibrated
)

print("\n" + "=" * 80)
print("PIPELINE COMPLETE!")
print("=" * 80)

print("\nSummary:")
print(f"  - Total samples: {len(features_df)}")
print(f"  - Training samples: {len(X_train)}")
print(f"  - Test samples: {len(X_test)}")
print(f"  - Features: {len(X.columns)}")
print(f"  - Target: Volatility Spike Detection (volatility_change > 75th percentile)")
print(f"  - Models trained: Logistic Regression, Gradient Boosting")
print(f"  - Calibration: Applied (Isotonic Regression)")
print(f"  - Evaluation: Complete (Binary, Probabilistic, Buckets, Absolute Return Magnitude)")
