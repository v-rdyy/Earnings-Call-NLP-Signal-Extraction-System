"""
Streamlit Demo: Earnings Call NLP Signal Inspection Tool

Lightweight interface for inspecting extracted NLP features and 
calibrated probabilistic outputs on unseen earnings disclosures.
"""

import sys
import os
import streamlit as st
import pandas as pd
import pickle
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.features.nlp_features import extract_all_nlp_features
from src.models.calibration import apply_calibration

st.set_page_config(
    page_title="Earnings Call NLP Signal Inspection",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_model():
    """Load trained model and calibrator"""
    model_path = 'outputs/models/demo_model.pkl'
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please run notebooks/save_model_for_demo.py first.")
        st.stop()
    
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    
    return model_dict

def extract_core_features(nlp_features):
    """Extract 8-10 core features for display"""
    core_features = {
        'Sentiment': {
            'doc_sentiment': nlp_features.get('doc_sentiment', 0.0),
            'sent_mean': nlp_features.get('sent_mean', 0.0),
            'sent_variance': nlp_features.get('sent_variance', 0.0)
        },
        'Uncertainty': {
            'uncertainty_density': nlp_features.get('uncertainty_density', 0.0)
        },
        'Outlook': {
            'forward_backward_ratio': nlp_features.get('forward_backward_ratio', 0.0)
        },
        'Financial Language': {
            'guidance_density': nlp_features.get('guidance_density', 0.0),
            'surprise_density': nlp_features.get('surprise_density', 0.0)
        },
        'Text Scale': {
            'word_count': nlp_features.get('word_count', 0)
        }
    }
    return core_features

def main():
    model_dict = load_model()
    model = model_dict['model']
    calibrator = model_dict['calibrator']
    feature_columns = model_dict['feature_columns']
    
    st.title("Earnings Call NLP Signal Inspection Tool")
    st.subheader("Research demo for inspecting probabilistic language-based signals from earnings disclosures")
    
    st.warning("**This tool is for research and analysis only. Outputs are not trading signals.**")
    
    st.markdown("---")
    
    st.header("Transcript Input")
    transcript_text = st.text_area(
        "Paste earnings call text or press release",
        height=250,
        placeholder="Paste earnings call transcript or press release text here..."
    )
    
    if not transcript_text.strip():
        st.info("Please paste earnings call text above to see extracted features and model output.")
        st.stop()
    
    st.markdown("---")
    
    st.header("Extracted NLP Signals")
    
    nlp_features = extract_all_nlp_features(transcript_text)
    
    core_features = extract_core_features(nlp_features)
    
    display_data = []
    for category, features in core_features.items():
        for feature_name, value in features.items():
            display_data.append({
                'Category': category,
                'Feature': feature_name,
                'Value': value
            })
    
    feature_df = pd.DataFrame(display_data)
    st.dataframe(feature_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.header("Model Output (Calibrated Probability)")
    
    feature_dict = {col: nlp_features.get(col, 0.0) for col in feature_columns}
    feature_vector = pd.DataFrame([feature_dict])
    
    try:
        uncalibrated_prob = model.predict_proba(feature_vector)[0, 1]
        calibrated_prob = apply_calibration(calibrator, np.array([uncalibrated_prob]))[0]
    except Exception as e:
        st.error(f"Error generating prediction: {e}")
        st.stop()
    
    is_clipped = calibrated_prob == 0.0 and uncalibrated_prob > 0.0
    
    if is_clipped:
        display_prob = uncalibrated_prob
        st.warning("⚠️ Calibrated probability was clipped to 0.0. Showing uncalibrated probability below.")
    else:
        display_prob = calibrated_prob
    
    st.metric(
        "Probability of Post-Earnings Volatility Spike",
        f"{display_prob:.2f}",
        delta=None
    )
    
    st.caption("This probability is calibrated using isotonic regression for reliability." if not is_clipped else "Uncalibrated probability shown (calibration clipped to 0.0).")
    
    with st.expander("View both probabilities"):
        st.write(f"**Calibrated probability:** {calibrated_prob:.4f}")
        st.write(f"**Uncalibrated probability:** {uncalibrated_prob:.4f}")
        if is_clipped:
            st.info("The calibrated probability was clipped because it's below the training range. For new transcripts, the uncalibrated probability above may be more informative.")
        else:
            st.caption("Calibrated probability is more reliable as it's adjusted for model calibration.")
    
    st.markdown("---")
    
    st.header("Interpretation Notes")
    
    interpretation_text = """
    • **Higher probabilities indicate elevated uncertainty in language.**
    • The model captures risk signals, not directional predictions.
    • Language tends to correlate more with volatility than returns.
    """
    
    st.markdown(interpretation_text)

if __name__ == "__main__":
    main()
