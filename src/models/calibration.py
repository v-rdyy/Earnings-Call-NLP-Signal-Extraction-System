"""
Model Calibration

This module calibrates model probabilities to ensure they are reliable.
Calibration ensures that if a model predicts 60%, it's actually right 60% of the time.
"""

import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression


def measure_calibration(y_true, y_probs, n_bins=10):
    """
    Measure how well-calibrated the model probabilities are.
    
    Creates a reliability diagram showing predicted probability vs actual frequency.
    Perfect calibration: predicted probability = actual frequency.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1)
    y_probs : array-like
        Predicted probabilities (0.0 to 1.0)
    n_bins : int
        Number of bins for calibration curve (default: 10)
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'fraction_of_positives': Actual frequency in each bin
        - 'mean_predicted_value': Mean predicted probability in each bin
        - 'calibration_error': Mean absolute difference (lower is better)
    """
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_probs, n_bins=n_bins, strategy='uniform'
    )
    
    calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
    
    return {
        'fraction_of_positives': fraction_of_positives,
        'mean_predicted_value': mean_predicted_value,
        'calibration_error': calibration_error
    }


def calibrate_probabilities(y_true, y_probs, method='isotonic'):
    """
    Calibrate model probabilities to make them reliable.
    
    Takes uncalibrated probabilities and adjusts them so that
    predicted probability matches actual frequency.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1) - used for calibration
    y_probs : array-like
        Uncalibrated probabilities (0.0 to 1.0)
    method : str
        Calibration method: 'isotonic' (non-parametric) or 'sigmoid' (parametric)
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'calibrator': Fitted calibration model
        - 'calibrated_probs': Calibrated probabilities
        - 'calibration_error_before': Error before calibration
        - 'calibration_error_after': Error after calibration
    """
    
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    before_calibration = measure_calibration(y_true, y_probs)
    error_before = before_calibration['calibration_error']
    
    if method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
    elif method == 'sigmoid':
        from sklearn.linear_model import LogisticRegression
        calibrator = LogisticRegression()
        y_probs = y_probs.reshape(-1, 1)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'isotonic' or 'sigmoid'")
    
    calibrator.fit(y_probs, y_true)
    
    if method == 'sigmoid':
        calibrated_probs = calibrator.predict_proba(y_probs)[:, 1]
    else:
        calibrated_probs = calibrator.predict(y_probs)
    
    after_calibration = measure_calibration(y_true, calibrated_probs)
    error_after = after_calibration['calibration_error']
    
    print(f"Calibration error: {error_before:.4f} → {error_after:.4f}")
    print(f"Improvement: {((error_before - error_after) / error_before * 100):.1f}%")
    
    return {
        'calibrator': calibrator,
        'calibrated_probs': calibrated_probs,
        'calibration_error_before': error_before,
        'calibration_error_after': error_after
    }


def apply_calibration(calibrator, y_probs, method='isotonic'):
    """
    Apply fitted calibrator to new probabilities.
    
    After calibrating on training data, use this to calibrate
    test data probabilities with the same calibrator.
    
    Parameters:
    -----------
    calibrator : fitted calibrator model
        Calibrator fitted on training data
    y_probs : array-like
        New probabilities to calibrate
    method : str
        Method used: 'isotonic' or 'sigmoid'
        
    Returns:
    --------
    array
        Calibrated probabilities
    """
    
    y_probs = np.array(y_probs)
    
    if method == 'sigmoid':
        y_probs = y_probs.reshape(-1, 1)
        calibrated_probs = calibrator.predict_proba(y_probs)[:, 1]
    else:
        calibrated_probs = calibrator.predict(y_probs)
    
    return calibrated_probs
