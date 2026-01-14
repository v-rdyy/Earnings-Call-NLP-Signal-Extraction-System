"""
NLP Feature Extraction Module

This module extracts language-based signals from earnings call transcripts.
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
import re


def extract_sentiment_features(transcript_text):
    """
    Extract sentiment features from transcript text.
    
    Computes:
    - Document-level sentiment (overall tone)
    - Sentence-level sentiment statistics (mean, variance)
    
    Parameters:
    -----------
    transcript_text : str
        The transcript text to analyze
        
    Returns:
    --------
    dict
        Dictionary with keys: doc_sentiment, sent_mean, sent_variance
    """
    
    if pd.isna(transcript_text) or transcript_text == '':
        return {
            'doc_sentiment': 0.0,
            'sent_mean': 0.0,
            'sent_variance': 0.0
        }
    
    blob = TextBlob(transcript_text)
    
    doc_sentiment = blob.sentiment.polarity
    
    sentences = blob.sentences
    sentence_sentiments = []
    
    for sentence in sentences:
        sent_blob = TextBlob(str(sentence))
        sentence_sentiments.append(sent_blob.sentiment.polarity)
    
    if len(sentence_sentiments) > 0:
        sent_mean = np.mean(sentence_sentiments)
        sent_variance = np.var(sentence_sentiments)
    else:
        sent_mean = 0.0
        sent_variance = 0.0
    
    return {
        'doc_sentiment': doc_sentiment,
        'sent_mean': sent_mean,
        'sent_variance': sent_variance
    }


def extract_uncertainty_features(transcript_text):
    """
    Extract uncertainty language features from transcript text.
    
    Counts hedging words that indicate uncertainty or caution.
    Computes:
    - Uncertainty word count
    - Uncertainty word density (per 100 words)
    
    Parameters:
    -----------
    transcript_text : str
        The transcript text to analyze
        
    Returns:
    --------
    dict
        Dictionary with keys: uncertainty_count, uncertainty_density
    """
    
    if pd.isna(transcript_text) or transcript_text == '':
        return {
            'uncertainty_count': 0,
            'uncertainty_density': 0.0
        }
    
    uncertainty_words = [
        'may', 'might', 'could', 'possibly', 'perhaps', 'maybe',
        'uncertain', 'uncertainty', 'unclear', 'unpredictable',
        'volatile', 'volatility', 'risk', 'risks', 'risky',
        'challenge', 'challenges', 'difficult', 'difficulty',
        'concern', 'concerns', 'worry', 'worries', 'caution',
        'cautious', 'hedge', 'hedging', 'contingent', 'contingency'
    ]
    
    text_lower = transcript_text.lower()
    
    words = text_lower.split()
    total_words = len(words)
    
    uncertainty_count = 0
    
    for word in uncertainty_words:
        uncertainty_count += text_lower.count(word)
    
    if total_words > 0:
        uncertainty_density = (uncertainty_count / total_words) * 100
    else:
        uncertainty_density = 0.0
    
    return {
        'uncertainty_count': uncertainty_count,
        'uncertainty_density': uncertainty_density
    }


def extract_forward_looking_features(transcript_text):
    """
    Extract forward-looking vs backward-looking language features.
    
    Distinguishes between:
    - Forward-looking: guidance, outlook, future plans
    - Backward-looking: past results, historical performance
    
    Computes:
    - Forward-looking word count and density
    - Backward-looking word count and density
    - Forward/backward ratio
    
    Parameters:
    -----------
    transcript_text : str
        The transcript text to analyze
        
    Returns:
    --------
    dict
        Dictionary with keys: forward_count, forward_density, 
        backward_count, backward_density, forward_backward_ratio
    """
    
    if pd.isna(transcript_text) or transcript_text == '':
        return {
            'forward_count': 0,
            'forward_density': 0.0,
            'backward_count': 0,
            'backward_density': 0.0,
            'forward_backward_ratio': 0.0
        }
    
    forward_words = [
        'will', 'expect', 'expects', 'expected', 'expecting',
        'outlook', 'guidance', 'forecast', 'forecasts', 'forecasting',
        'plan', 'plans', 'planning', 'strategy', 'strategic',
        'future', 'ahead', 'going forward', 'next quarter',
        'next year', 'upcoming', 'anticipate', 'anticipates',
        'anticipating', 'project', 'projects', 'projected',
        'target', 'targets', 'goal', 'goals', 'objective', 'objectives'
    ]
    
    backward_words = [
        'was', 'were', 'had', 'previous', 'prior', 'last quarter',
        'last year', 'past', 'historical', 'history', 'achieved',
        'accomplished', 'completed', 'finished', 'ended',
        'reported', 'announced', 'released', 'delivered',
        'resulted', 'resulting', 'performance', 'performed'
    ]
    
    text_lower = transcript_text.lower()
    
    words = text_lower.split()
    total_words = len(words)
    
    forward_count = 0
    backward_count = 0
    
    for word in forward_words:
        forward_count += text_lower.count(word)
    
    for word in backward_words:
        backward_count += text_lower.count(word)
    
    if total_words > 0:
        forward_density = (forward_count / total_words) * 100
        backward_density = (backward_count / total_words) * 100
    else:
        forward_density = 0.0
        backward_density = 0.0
    
    if backward_count > 0:
        forward_backward_ratio = forward_count / backward_count
    else:
        forward_backward_ratio = forward_count if forward_count > 0 else 0.0
    
    return {
        'forward_count': forward_count,
        'forward_density': forward_density,
        'backward_count': backward_count,
        'backward_density': backward_density,
        'forward_backward_ratio': forward_backward_ratio
    }


def extract_all_nlp_features(transcript_text):
    """
    Extract all NLP features from transcript text.
    
    This is a convenience function that calls all individual
    feature extraction functions and combines their results.
    
    Parameters:
    -----------
    transcript_text : str
        The transcript text to analyze
        
    Returns:
    --------
    dict
        Dictionary containing all NLP features:
        - Sentiment features (doc_sentiment, sent_mean, sent_variance)
        - Uncertainty features (uncertainty_count, uncertainty_density)
        - Forward/backward features (forward_count, forward_density, etc.)
    """
    
    sentiment_features = extract_sentiment_features(transcript_text)
    uncertainty_features = extract_uncertainty_features(transcript_text)
    forward_features = extract_forward_looking_features(transcript_text)
    
    all_features = {}
    all_features.update(sentiment_features)
    all_features.update(uncertainty_features)
    all_features.update(forward_features)
    
    return all_features


def add_nlp_features_to_dataframe(df, transcript_column='transcript_text'):
    """
    Apply NLP feature extraction to all transcripts in a DataFrame.
    
    Takes a DataFrame with transcript text and adds columns for all
    NLP features (sentiment, uncertainty, forward/backward looking).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing transcript data. Must have a column
        with transcript text (default: 'transcript_text')
    transcript_column : str
        Name of the column containing transcript text
        
    Returns:
    --------
    pd.DataFrame
        Original DataFrame with added NLP feature columns
    """
    
    if transcript_column not in df.columns:
        raise ValueError(f"Column '{transcript_column}' not found in DataFrame")
    
    nlp_features_list = []
    
    for idx, row in df.iterrows():
        transcript_text = row[transcript_column]
        features = extract_all_nlp_features(transcript_text)
        nlp_features_list.append(features)
    
    features_df = pd.DataFrame(nlp_features_list)
    
    result_df = pd.concat([df, features_df], axis=1)
    
    return result_df

