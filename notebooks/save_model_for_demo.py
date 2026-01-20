"""
Save trained model for Streamlit demo

This script trains a model and saves it for use in the Streamlit demo.
Run this once before using the Streamlit app.
"""

import sys
import os
import pickle
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.load_data import load_transcripts, load_prices, align_transcripts_with_prices
from src.features.nlp_features import add_nlp_features_to_dataframe
from src.models.prepare_data import prepare_modeling_data
from src.models.train_models import create_binary_target, train_gradient_boosting
from src.models.calibration import calibrate_probabilities
from sklearn.preprocessing import StandardScaler

print("Loading data and training model for demo...")

transcripts_df = load_transcripts('data/raw/real_transcripts.csv')
prices_df = load_prices('data/raw/real_prices.csv')
aligned_df = align_transcripts_with_prices(transcripts_df, prices_df)

features_cache_path = 'data/processed/features_with_nlp.csv'
if os.path.exists(features_cache_path):
    features_df = pd.read_csv(features_cache_path)
    features_df['date'] = pd.to_datetime(features_df['date'])
else:
    features_df = add_nlp_features_to_dataframe(aligned_df)
    os.makedirs('data/processed', exist_ok=True)
    features_df.to_csv(features_cache_path, index=False)

target_column = 'volatility_change'
X, y = prepare_modeling_data(features_df, target_column=target_column)

volatility_threshold = y.quantile(0.75)
y_binary = create_binary_target(y, threshold=volatility_threshold)

gb_results = train_gradient_boosting(X, y_binary, X)
gb_cal = calibrate_probabilities(y_binary, gb_results['train_probs'])

model_dict = {
    'model': gb_results['model'],
    'calibrator': gb_cal['calibrator'],
    'scaler': None,
    'threshold': volatility_threshold,
    'feature_columns': list(X.columns)
}

os.makedirs('outputs/models', exist_ok=True)
model_path = 'outputs/models/demo_model.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(model_dict, f)

print(f"Model saved to {model_path}")
print(f"Trained on {len(X)} samples")
print(f"Volatility spike threshold: {volatility_threshold:.6f}")
