"""
Complete Pipeline: Earnings Call NLP Signal Extraction

This script runs the complete pipeline:
1. Load and align data
2. Extract NLP features
3. Prepare for modeling
4. Train models
5. Calibrate probabilities
6. Evaluate performance
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
    train_gradient_boosting
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
print("EARNINGS CALL NLP SIGNAL EXTRACTION - COMPLETE PIPELINE")
print("=" * 80)

print("\n" + "=" * 80)
print("STEP 1: Load and Align Data")
print("=" * 80)

transcripts_df = load_transcripts()
prices_df = load_prices()
aligned_df = align_transcripts_with_prices(transcripts_df, prices_df)

print(f"\nAligned {len(aligned_df)} earnings calls with price data")

print("\n" + "=" * 80)
print("STEP 2: Extract NLP Features")
print("=" * 80)

features_df = add_nlp_features_to_dataframe(aligned_df)

print(f"Extracted NLP features for {len(features_df)} transcripts")
print(f"Features: {', '.join([col for col in features_df.columns if col not in ['ticker', 'date', 'transcript_text', 'reference_price', 'return_1d', 'return_3d', 'return_5d', 'pre_volatility', 'post_volatility', 'volatility_change']])}")

print("\n" + "=" * 80)
print("STEP 3: Prepare Data for Modeling")
print("=" * 80)

target_column = 'return_1d'
X, y = prepare_modeling_data(features_df, target_column=target_column)

print(f"\nPrepared {len(X)} samples with {len(X.columns)} features")
print(f"Target: {target_column}")

print("\n" + "=" * 80)
print("STEP 4: Time-Aware Train/Test Split")
print("=" * 80)

dates = features_df.loc[X.index, 'date']
X_train, X_test, y_train, y_test = time_aware_train_test_split(
    X, y, dates, test_size=0.2
)

print("\n" + "=" * 80)
print("STEP 5: Create Binary Targets")
print("=" * 80)

y_train_binary = create_binary_target(y_train, threshold=0.0)
y_test_binary = create_binary_target(y_test, threshold=0.0)

print(f"Training: {y_train_binary.sum()} positive ({y_train_binary.mean():.2%})")
print(f"Test: {y_test_binary.sum()} positive ({y_test_binary.mean():.2%})")

print("\n" + "=" * 80)
print("STEP 6: Train Models")
print("=" * 80)

print("\n--- Logistic Regression ---")
lr_results = train_logistic_regression(X_train, y_train_binary, X_test)

print("\n--- Gradient Boosting ---")
gb_results = train_gradient_boosting(X_train, y_train_binary, X_test)

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
print("STEP 11: Volatility Reaction Analysis")
print("=" * 80)

test_indices = X_test.index
test_volatility = features_df.loc[test_indices, 'volatility_change'].values

print("\n--- Logistic Regression (Calibrated) ---")
lr_volatility = evaluate_volatility_by_signal(
    y_test_binary, test_volatility, lr_test_calibrated
)

print("\n--- Gradient Boosting (Calibrated) ---")
gb_volatility = evaluate_volatility_by_signal(
    y_test_binary, test_volatility, gb_test_calibrated
)

print("\n" + "=" * 80)
print("PIPELINE COMPLETE!")
print("=" * 80)

print("\nSummary:")
print(f"  - Total samples: {len(features_df)}")
print(f"  - Training samples: {len(X_train)}")
print(f"  - Test samples: {len(X_test)}")
print(f"  - Features: {len(X.columns)}")
print(f"  - Models trained: Logistic Regression, Gradient Boosting")
print(f"  - Calibration: Applied (Isotonic Regression)")
print(f"  - Evaluation: Complete (Binary, Probabilistic, Buckets, Volatility)")
