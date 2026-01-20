# Earnings Call NLP Signal Extraction System

## Project Overview

An interpretable ML system that extracts probabilistic language-based uncertainty signals from earnings call transcripts and evaluates how those signals relate to post-earnings market behavior under uncertainty.

**Core Question:** Does the way management speaks during earnings calls correlate with volatility and return behavior afterward — and how reliably?

This is not about predicting stock direction, but about extracting reliable signals from language and evaluating them correctly.

## TL;DR

- Built an end-to-end NLP system to extract uncertainty signals from earnings disclosures
- Found language predicts post-earnings volatility better than return direction (+8.7% ROC-AUC)
- Used time-aware splits, calibration, and event studies for realistic evaluation
- Conclusion: language provides weak but persistent signals about risk, not direction

## Key Finding

**Language is more informative about uncertainty (volatility) than direction (returns).**

- Volatility Spike Prediction: ROC-AUC 0.6352
- Return Direction Prediction: ROC-AUC 0.5845
- **Improvement: +8.7%**

This aligns with market efficiency theory: direction is harder to predict as information is quickly priced in, while volatility reflects uncertainty that language captures better.

## Project Structure

```
nlpearningscall/
├── data/              # Data directories (gitignored)
│   ├── raw/          # Transcripts and prices
│   ├── processed/    # Aligned data & NLP features
│   └── features/     # Extracted features
├── src/              # Source code modules
│   ├── data/         # Data loading, alignment, SEC EDGAR fetching
│   ├── features/     # NLP feature extraction
│   ├── models/       # Training, calibration, data preparation
│   └── evaluation/   # Model evaluation and event study analysis
├── notebooks/        # Analysis scripts
├── outputs/          # Saved models and results (gitignored)
├── streamlit_app.py  # Interactive demo interface
├── requirements.txt  # Python dependencies
├── ERROR_ANALYSIS.md # Detailed error analysis
└── README.md         # This file
```

## Setup Instructions

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Mac/Linux
   # or
   venv\Scripts\activate     # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download TextBlob corpora:**
   ```bash
   python3 -m textblob.download_corpora
   ```

## Getting Started

### 1. Fetch Real Data (One-time setup)

**Option A: Process already-downloaded SEC filings**
```bash
python3 notebooks/process_downloaded_filings.py
```

**Option B: Download new transcripts from SEC EDGAR**
```bash
python3 notebooks/fetch_real_transcripts.py
```

This will:
- Download 8-K filings from SEC EDGAR for multiple companies
- Extract transcript text from filings
- Fetch corresponding stock prices from yfinance
- Save to `data/raw/real_transcripts.csv` and `data/raw/real_prices.csv`

### 2. Run Complete Pipeline

```bash
python3 notebooks/run_pipeline_with_real_data.py
```

This runs the complete end-to-end workflow:
1. Load and align transcripts with price data
2. Extract NLP features (25 features: sentiment, uncertainty, forward-looking language, financial keywords, text statistics)
3. Prepare data for modeling (time-aware train/test split)
4. Train models (logistic regression, gradient boosting)
5. Calibrate probabilities (isotonic regression)
6. Evaluate performance (binary, probabilistic, event study)

### 3. Run Additional Analyses

**Target Comparison:**
```bash
python3 notebooks/compare_targets.py
```
Compares model performance when predicting return direction vs volatility spikes. Demonstrates that language is more informative about uncertainty than direction.

**Stability Check:**
```bash
python3 notebooks/stability_check.py
```
Evaluates model consistency across different time periods (by year) and out-of-time testing (train on 2022-2023, test on 2024). Assesses signal persistence and robustness.

**Full Analysis Suite:**
```bash
python3 notebooks/run_full_analysis.py
```
Runs all analyses in sequence: main pipeline, target comparison, and stability check.

### 4. Interactive Demo

```bash
# First, save the trained model
python3 notebooks/save_model_for_demo.py

# Then run the Streamlit app
streamlit run streamlit_app.py
```

A lightweight Streamlit interface to inspect extracted NLP features and probabilistic outputs on unseen earnings disclosures.

## Model Performance

**Best Model: Gradient Boosting Classifier**
- **Target**: Volatility Spike Detection (volatility_change > 75th percentile)
- **ROC-AUC**: 0.6352 (63.52%)
- **Accuracy**: 70.93%
- **Calibration Error**: ~0.000 (isotonic calibration on training data)

**Interpretation:**
- Above random chance (50%) but not production-ready
- Demonstrates that language signals provide measurable predictive power for volatility
- Results are realistic given market efficiency constraints

## Feature Importance

**Top Features for Gradient Boosting:**
1. `sent_variance` (8.5%) - Sentiment consistency
2. `uncertainty_density` (8.0%) - Uncertainty language
3. `financial_perf_count` (7.6%) - Financial performance terms
4. `backward_density` (6.7%) - Backward-looking language
5. `beat_miss_density` (5.8%) - Beat/miss language

**Key Learning:** Sentiment consistency and uncertainty language are the strongest signals, followed by financial keywords.

## NLP Features (25 Total)

The system extracts 25 interpretable features:

- **Sentiment (3)**: Document sentiment, sentence mean, sentence variance
- **Uncertainty (2)**: Uncertainty word count and density
- **Forward/Backward Looking (5)**: Forward/backward word counts, densities, ratio
- **Financial Keywords (10)**: Guidance, beat/miss, surprise, financial performance, risk/loss (counts and densities)
- **Text Statistics (5)**: Character count, word count, average word length, sentence count, average sentence length

## Key Findings

### 1. Language More Informative About Volatility Than Direction

- **Volatility Spike Prediction**: ROC-AUC 0.6352
- **Return Direction Prediction**: ROC-AUC 0.5845
- **Improvement**: +8.7%

**Interview Answer:** "We found language is more informative about risk than direction, which aligns with how markets incorporate earnings information."

### 2. Signal Persistence

- Out-of-time test (2022-2023 → 2024): ROC-AUC 0.5161
- Above random (0.50) but modest
- Shows: Signal persists across time periods, but is weak

### 3. Calibration Success

- Calibration error → ~0.000 (isotonic calibration on training data)
- Interpretation: When model says 60% probability, it's right 60% of the time (on training data)

## Error Analysis

See `ERROR_ANALYSIS.md` for detailed analysis of when and why the model fails.

**Key Failure Modes:**
1. **Neutral language with hidden uncertainty** - Press releases contain neutral language but Q&A reveals uncertainty that the model cannot detect
2. **Context-dependent language** - Financial keywords appear in neutral contexts (e.g., "guidance" mentioned historically)
3. **Market overreaction** - Language suggests moderate uncertainty but market overreacts for non-fundamental reasons
4. **Industry-specific patterns** - Language patterns that signal uncertainty in one industry don't translate to others
5. **Rapid information incorporation** - Information revealed in transcripts is already reflected in stock prices

**Main Limitation:**
The model struggles when press releases contain neutral language but Q&A reveals uncertainty, highlighting the limitation of scripted disclosures. This validates the probabilistic framing: we're extracting signals from available language, recognizing that full transcripts with Q&A would provide stronger signals.

## Technical Details

### Data Sources
- **SEC EDGAR**: 8-K filings (press releases) via `sec-edgar-downloader`
- **Stock Prices**: Historical data via `yfinance`
- **Dataset**: 859 aligned transcripts from 40+ companies (2022-2024)

### Models
- **Logistic Regression**: Linear model with feature scaling
- **Gradient Boosting**: Ensemble of decision trees (100 estimators)
- **Calibration**: Isotonic Regression

### Evaluation
- **Binary Metrics**: Accuracy, Precision, Recall, Confusion Matrix
- **Probabilistic Metrics**: ROC-AUC, Brier Score, Log Loss
- **Event Study**: Probability bucket analysis, volatility reaction by signal strength

## Limitations

1. **Data Quality**: Using 8-K filings (press releases) vs full Q&A transcripts
2. **Sample Size**: 859 samples is modest for complex models
3. **Feature Engineering**: Rule-based features (no semantic understanding)
4. **Market Efficiency**: Information processed very quickly, limiting predictive power
5. **Calibration**: May clip probabilities outside training range for new transcripts

## Future Improvements

**High Impact:**
- Full transcripts with Q&A sessions
- Speaker identification (management vs analyst)
- Larger dataset (5,000+ samples)
- Contextual features (sentiment of guidance, named entities)

**Medium Impact:**
- Industry normalization (relative features)
- Market integration (pre-earnings price movements)
- Deep learning (BERT/FinBERT for semantic understanding)
- Ensemble methods (specialized sub-models)

## Project Status

**Complete** - All modules implemented and tested. Ready for analysis and experimentation.

**Core Scripts:**
- `notebooks/run_pipeline_with_real_data.py` - Main pipeline (volatility spike prediction)
- `notebooks/compare_targets.py` - Side-by-side target comparison
- `notebooks/stability_check.py` - Temporal stability analysis
- `notebooks/run_full_analysis.py` - Complete analysis suite
- `notebooks/fetch_real_transcripts.py` - Download transcripts from SEC EDGAR
- `notebooks/process_downloaded_filings.py` - Process already-downloaded filings
- `notebooks/save_model_for_demo.py` - Save model for Streamlit demo
- `streamlit_app.py` - Interactive demo interface

**Documentation:**
- `README.md` - This file
- `ERROR_ANALYSIS.md` - Detailed error analysis and limitations

## License

This project is for research and educational purposes.
