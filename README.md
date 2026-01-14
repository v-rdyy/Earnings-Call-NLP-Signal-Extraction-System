# Earnings Call NLP Signal Extraction System

## Project Overview

An ML system that extracts probabilistic language-based signals from earnings call transcripts and evaluates how those signals relate to post-earnings market behavior under uncertainty.

**Core Question:** Does the way management speaks during earnings calls correlate with volatility and return behavior afterward — and how reliably?

This is not about predicting stock direction, but about extracting reliable signals from language and evaluating them correctly.

## Project Structure

- `data/`: Raw and processed data
  - `raw/`: Sample transcripts and price data
  - `processed/`: Cleaned and aligned data
  - `features/`: Extracted features
- `src/`: Source code modules
  - `data/`: Data generation, loading, and alignment
  - `features/`: NLP feature extraction
  - `models/`: Model training (coming soon)
  - `evaluation/`: Evaluation scripts (coming soon)
- `notebooks/`: Test scripts and exploration
- `outputs/`: Saved models and results

## Setup Instructions

1. Create virtual environment: `python3 -m venv venv`
2. Activate virtual environment: `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Download TextBlob corpora: `python3 -m textblob.download_corpora`

## Current Status

### Completed Modules

1. **Data Generation** (`src/data/generate_sample_data.py`)
   - Sample earnings call transcript generation
   - Sample stock price data generation
   - CSV export functionality

2. **Data Loading & Alignment** (`src/data/load_data.py`)
   - Load transcripts and prices from CSV
   - Align transcripts with price data
   - Compute continuous returns (1d, 3d, 5d)
   - Compute volatility metrics (pre/post earnings)

3. **NLP Feature Extraction** (`src/features/nlp_features.py`)
   - Sentiment features (document and sentence-level)
   - Uncertainty language detection
   - Forward-looking vs backward-looking language analysis

### Next Steps

- Probabilistic modeling (logistic regression, gradient boosting)
- Model calibration and reliability curves
- Event study evaluation methodology

## Usage

### Generate Sample Data
```python
from src.data.generate_sample_data import generate_and_save_sample_data
transcripts, prices = generate_and_save_sample_data(num_calls=20)
```

### Load and Align Data
```python
from src.data.load_data import load_transcripts, load_prices, align_transcripts_with_prices
transcripts = load_transcripts()
prices = load_prices()
aligned = align_transcripts_with_prices(transcripts, prices)
```

### Extract NLP Features
```python
from src.features.nlp_features import add_nlp_features_to_dataframe
features_df = add_nlp_features_to_dataframe(aligned)
```

## Project Status

In Development - Data pipeline and NLP feature extraction complete. Probabilistic modeling in progress.