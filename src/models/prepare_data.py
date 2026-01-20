"""
Data Preparation for Modeling

This module prepares data for probabilistic modeling.
"""

import pandas as pd
import numpy as np


def prepare_modeling_data(df, target_column='return_1d', feature_columns=None):
    """
    Prepare DataFrame for modeling by separating features and targets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with NLP features and target variables
    target_column : str
        Name of the target column to predict
    feature_columns : list, optional
        List of feature column names. If None, uses all NLP feature columns.
        
    Returns:
    --------
    tuple
        (X, y) where X is features DataFrame and y is target Series
    """
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    if feature_columns is None:
        feature_columns = [
            'doc_sentiment', 'sent_mean', 'sent_variance',
            'uncertainty_count', 'uncertainty_density',
            'forward_count', 'forward_density',
            'backward_count', 'backward_density',
            'forward_backward_ratio',
            'guidance_count', 'guidance_density',
            'beat_miss_count', 'beat_miss_density',
            'surprise_count', 'surprise_density',
            'financial_perf_count', 'financial_perf_density',
            'risk_loss_count', 'risk_loss_density',
            'char_count', 'word_count', 'avg_word_length',
            'sentence_count', 'avg_sentence_length'
        ]
    
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Feature columns not found: {missing_features}")
    
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    valid_mask = ~(y.isna() | X.isna().any(axis=1))
    
    X_clean = X[valid_mask].copy()
    y_clean = y[valid_mask].copy()
    
    print(f"Prepared data: {len(X_clean)} samples, {len(feature_columns)} features")
    print(f"Removed {len(df) - len(X_clean)} samples with missing values")
    
    return X_clean, y_clean


def select_features_by_importance(X, y, model_type='gradient_boosting', n_features=20, random_state=42):
    """
    Select top N most important features based on a quick model fit.
    
    This helps reduce overfitting by keeping only the most predictive features.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features DataFrame
    y : pd.Series
        Target Series
    model_type : str
        Type of model to use for feature selection: 'gradient_boosting' or 'logistic_regression'
    n_features : int
        Number of top features to keep (default: 20)
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    list
        List of selected feature names
    """
    
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    if model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=random_state)
        model.fit(X, y)
        importances = model.feature_importances_
    elif model_type == 'logistic_regression':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(random_state=random_state, max_iter=500)
        model.fit(X_scaled, y)
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    selected_features = feature_importance.head(n_features)['feature'].tolist()
    
    return selected_features


def time_aware_train_test_split(X, y, dates, test_size=0.2):
    """
    Split data by time, not randomly.
    
    This is critical for time-series data to prevent data leakage.
    Training data consists of earlier earnings calls.
    Test data consists of later earnings calls.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features DataFrame
    y : pd.Series
        Target Series
    dates : pd.Series
        Dates corresponding to each sample (must align with X and y indices)
    test_size : float
        Proportion of data to use for testing (default: 0.2 = 20%)
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test) split by time
    """
    
    if len(X) != len(y) or len(X) != len(dates):
        raise ValueError("X, y, and dates must have the same length")
    
    combined = pd.DataFrame({
        'date': dates.values,
        'target': y.values
    }, index=X.index)
    
    for col in X.columns:
        combined[col] = X[col].values
    
    combined = combined.sort_values('date')
    
    split_idx = int(len(combined) * (1 - test_size))
    
    train_data = combined.iloc[:split_idx]
    test_data = combined.iloc[split_idx:]
    
    X_train = train_data[X.columns]
    y_train = train_data['target']
    X_test = test_data[X.columns]
    y_test = test_data['target']
    
    print(f"Train set: {len(X_train)} samples (earliest: {train_data['date'].min()}, latest: {train_data['date'].max()})")
    print(f"Test set: {len(X_test)} samples (earliest: {test_data['date'].min()}, latest: {test_data['date'].max()})")
    
    return X_train, X_test, y_train, y_test

