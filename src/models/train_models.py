"""
Model Training for Probabilistic Predictions

This module trains models that output probabilities, not binary predictions.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


def create_binary_target(y, threshold=0.0):
    """
    Convert continuous target to binary for classification.
    
    For returns: 1 if return > threshold, 0 otherwise
    For volatility: 1 if change > threshold, 0 otherwise
    
    Parameters:
    -----------
    y : pd.Series
        Continuous target values (returns or volatility changes)
    threshold : float
        Threshold for binary classification (default: 0.0)
        
    Returns:
    --------
    pd.Series
        Binary target: 1 if y > threshold, 0 otherwise
    """
    
    binary_target = (y > threshold).astype(int)
    
    return binary_target


def train_logistic_regression(X_train, y_train_binary, X_test=None):
    """
    Train logistic regression model for probabilistic predictions.
    
    Logistic regression outputs probabilities (0.0 to 1.0) representing
    the probability of the positive class (y=1).
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train_binary : pd.Series
        Binary training targets (0 or 1)
    X_test : pd.DataFrame, optional
        Test features for prediction
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'model': Trained model
        - 'scaler': Feature scaler
        - 'train_probs': Training set probabilities
        - 'test_probs': Test set probabilities (if X_test provided)
    """
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train_binary)
    
    train_probs = model.predict_proba(X_train_scaled)[:, 1]
    
    result = {
        'model': model,
        'scaler': scaler,
        'train_probs': train_probs
    }
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        test_probs = model.predict_proba(X_test_scaled)[:, 1]
        result['test_probs'] = test_probs
    
    print(f"Trained logistic regression on {len(X_train)} samples")
    print(f"Positive class rate: {y_train_binary.mean():.2%}")
    
    return result


def train_gradient_boosting(X_train, y_train_binary, X_test=None):
    """
    Train gradient boosting model for probabilistic predictions.
    
    Gradient boosting is an ensemble method that combines multiple
    weak learners (decision trees) to make stronger predictions.
    Outputs probabilities like logistic regression.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train_binary : pd.Series
        Binary training targets (0 or 1)
    X_test : pd.DataFrame, optional
        Test features for prediction
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'model': Trained model
        - 'train_probs': Training set probabilities
        - 'test_probs': Test set probabilities (if X_test provided)
    """
    
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    model.fit(X_train, y_train_binary)
    
    train_probs = model.predict_proba(X_train)[:, 1]
    
    result = {
        'model': model,
        'train_probs': train_probs
    }
    
    if X_test is not None:
        test_probs = model.predict_proba(X_test)[:, 1]
        result['test_probs'] = test_probs
    
    print(f"Trained gradient boosting on {len(X_train)} samples")
    print(f"Positive class rate: {y_train_binary.mean():.2%}")
    
    return result


def get_feature_importance(model, feature_names, model_type='gradient_boosting'):
    """
    Extract feature importance from trained model.
    
    For gradient boosting: Uses built-in feature_importances_
    For logistic regression: Uses absolute value of coefficients
    
    Parameters:
    -----------
    model : Trained model
        Either GradientBoostingClassifier or LogisticRegression
    feature_names : list
        List of feature names (must match order in training data)
    model_type : str
        Type of model: 'gradient_boosting' or 'logistic_regression'
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with features and their importance scores, sorted by importance
    """
    
    if model_type == 'gradient_boosting':
        importances = model.feature_importances_
    elif model_type == 'logistic_regression':
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    feature_importance_df['importance_pct'] = (
        feature_importance_df['importance'] / feature_importance_df['importance'].sum() * 100
    )
    
    return feature_importance_df
