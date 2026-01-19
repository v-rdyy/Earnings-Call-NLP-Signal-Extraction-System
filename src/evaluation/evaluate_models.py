"""
Model Evaluation

This module evaluates model performance using metrics appropriate for
probabilistic predictions and event study methodology.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score


def evaluate_binary_predictions(y_true, y_pred_binary):
    """
    Evaluate binary predictions (0 or 1) using standard classification metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1)
    y_pred_binary : array-like
        Predicted binary labels (0 or 1)
        
    Returns:
    --------
    dict
        Dictionary containing accuracy, precision, recall, and confusion matrix
    """
    
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    cm = confusion_matrix(y_true, y_pred_binary)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm
    }
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Negative  Positive")
    print(f"Actual Negative    {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"       Positive    {cm[1,0]:4d}      {cm[1,1]:4d}")
    
    return results


def evaluate_probabilistic_predictions(y_true, y_probs):
    """
    Evaluate probabilistic predictions using metrics appropriate for probabilities.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1)
    y_probs : array-like
        Predicted probabilities (0.0 to 1.0)
        
    Returns:
    --------
    dict
        Dictionary containing ROC-AUC, Brier score, and log loss
    """
    
    roc_auc = roc_auc_score(y_true, y_probs)
    
    y_true_array = np.array(y_true)
    y_probs_array = np.array(y_probs)
    
    brier_score = np.mean((y_probs_array - y_true_array) ** 2)
    
    epsilon = 1e-15
    y_probs_clipped = np.clip(y_probs_array, epsilon, 1 - epsilon)
    log_loss = -np.mean(y_true_array * np.log(y_probs_clipped) + 
                       (1 - y_true_array) * np.log(1 - y_probs_clipped))
    
    results = {
        'roc_auc': roc_auc,
        'brier_score': brier_score,
        'log_loss': log_loss
    }
    
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Brier Score: {brier_score:.4f} (lower is better)")
    print(f"Log Loss: {log_loss:.4f} (lower is better)")
    
    return results


def evaluate_by_probability_buckets(y_true, y_probs, n_buckets=10):
    """
    Evaluate model performance by probability buckets (event study style).
    
    Groups predictions into buckets (0-10%, 10-20%, etc.) and calculates
    actual frequency in each bucket. This shows conditional return distributions.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1)
    y_probs : array-like
        Predicted probabilities (0.0 to 1.0)
    n_buckets : int
        Number of probability buckets (default: 10)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: bucket, predicted_prob, actual_freq, count
    """
    
    y_true_array = np.array(y_true)
    y_probs_array = np.array(y_probs)
    
    bucket_edges = np.linspace(0, 1, n_buckets + 1)
    bucket_labels = [f"{bucket_edges[i]:.1f}-{bucket_edges[i+1]:.1f}" 
                     for i in range(n_buckets)]
    
    bucket_indices = np.digitize(y_probs_array, bucket_edges) - 1
    bucket_indices = np.clip(bucket_indices, 0, n_buckets - 1)
    
    results = []
    
    for i in range(n_buckets):
        mask = bucket_indices == i
        if np.sum(mask) > 0:
            bucket_probs = y_probs_array[mask]
            bucket_labels_true = y_true_array[mask]
            
            predicted_prob = np.mean(bucket_probs)
            actual_freq = np.mean(bucket_labels_true)
            count = np.sum(mask)
            
            results.append({
                'bucket': bucket_labels[i],
                'predicted_prob': predicted_prob,
                'actual_freq': actual_freq,
                'count': count,
                'difference': abs(predicted_prob - actual_freq)
            })
    
    results_df = pd.DataFrame(results)
    
    print("\nProbability Bucket Analysis:")
    print("=" * 70)
    print(f"{'Bucket':<15} {'Predicted':<12} {'Actual':<12} {'Count':<8} {'Diff':<8}")
    print("=" * 70)
    for _, row in results_df.iterrows():
        print(f"{row['bucket']:<15} {row['predicted_prob']:>10.2%} {row['actual_freq']:>10.2%} "
              f"{int(row['count']):>6} {row['difference']:>7.2%}")
    
    return results_df


def evaluate_volatility_by_signal(y_true, volatility_change, y_probs, n_buckets=5):
    """
    Evaluate volatility reaction by signal strength (event study methodology).
    
    Groups predictions into buckets and calculates average volatility change
    in each bucket. Shows how volatility reacts to different signal strengths.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1)
    volatility_change : array-like
        Volatility change values (pre vs post earnings)
    y_probs : array-like
        Predicted probabilities (0.0 to 1.0)
    n_buckets : int
        Number of signal strength buckets (default: 5)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with volatility statistics by signal bucket
    """
    
    y_probs_array = np.array(y_probs)
    volatility_array = np.array(volatility_change)
    
    bucket_edges = np.linspace(0, 1, n_buckets + 1)
    bucket_indices = np.digitize(y_probs_array, bucket_edges) - 1
    bucket_indices = np.clip(bucket_indices, 0, n_buckets - 1)
    
    results = []
    
    for i in range(n_buckets):
        mask = bucket_indices == i
        if np.sum(mask) > 0:
            bucket_volatility = volatility_array[mask]
            bucket_probs = y_probs_array[mask]
            
            results.append({
                'bucket': f"{bucket_edges[i]:.1f}-{bucket_edges[i+1]:.1f}",
                'signal_strength': np.mean(bucket_probs),
                'mean_volatility_change': np.mean(bucket_volatility),
                'std_volatility_change': np.std(bucket_volatility),
                'count': np.sum(mask)
            })
    
    results_df = pd.DataFrame(results)
    
    print("\nVolatility Reaction by Signal Strength:")
    print("=" * 80)
    print(f"{'Bucket':<15} {'Signal':<12} {'Mean Vol Change':<18} {'Std':<12} {'Count':<8}")
    print("=" * 80)
    for _, row in results_df.iterrows():
        print(f"{row['bucket']:<15} {row['signal_strength']:>10.2%} "
              f"{row['mean_volatility_change']:>15.4f} {row['std_volatility_change']:>10.4f} "
              f"{int(row['count']):>6}")
    
    return results_df
