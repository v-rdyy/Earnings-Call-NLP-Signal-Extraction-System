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


def extract_text_statistics(transcript_text):
    """
    Extract basic text statistics that may correlate with information content.
    
    Computes:
    - Text length (character count, word count)
    - Average word length
    - Sentence count and average sentence length
    
    Parameters:
    -----------
    transcript_text : str
        The transcript text to analyze
        
    Returns:
    --------
    dict
        Dictionary with text statistics
    """
    
    if pd.isna(transcript_text) or transcript_text == '':
        return {
            'char_count': 0,
            'word_count': 0,
            'avg_word_length': 0.0,
            'sentence_count': 0,
            'avg_sentence_length': 0.0
        }
    
    char_count = len(transcript_text)
    words = transcript_text.split()
    word_count = len(words)
    
    if word_count > 0:
        total_char_in_words = sum(len(word) for word in words)
        avg_word_length = total_char_in_words / word_count
    else:
        avg_word_length = 0.0
    
    blob = TextBlob(transcript_text)
    sentences = blob.sentences
    sentence_count = len(sentences)
    
    if sentence_count > 0:
        avg_sentence_length = word_count / sentence_count
    else:
        avg_sentence_length = 0.0
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'avg_word_length': avg_word_length,
        'sentence_count': sentence_count,
        'avg_sentence_length': avg_sentence_length
    }


def extract_financial_keyword_features(transcript_text):
    """
    Extract financial-domain specific keyword features.
    
    Looks for:
    - Guidance mentions (guidance, outlook, forecast)
    - Beat/miss language (exceeded, beat, missed, below)
    - Surprise indicators (surprise, unexpected, strong, weak)
    - Financial performance (revenue, profit, margin, growth)
    - Risk/loss language (loss, decline, decrease, challenge)
    
    Parameters:
    -----------
    transcript_text : str
        The transcript text to analyze
        
    Returns:
    --------
    dict
        Dictionary with financial keyword counts and densities
    """
    
    if pd.isna(transcript_text) or transcript_text == '':
        return {
            'guidance_count': 0,
            'guidance_density': 0.0,
            'beat_miss_count': 0,
            'beat_miss_density': 0.0,
            'surprise_count': 0,
            'surprise_density': 0.0,
            'financial_perf_count': 0,
            'financial_perf_density': 0.0,
            'risk_loss_count': 0,
            'risk_loss_density': 0.0
        }
    
    guidance_words = [
        'guidance', 'outlook', 'forecast', 'forecasting', 'projection',
        'expectations', 'expected', 'target', 'targets', 'goal', 'goals'
    ]
    
    beat_miss_words = [
        'exceeded', 'exceed', 'beat', 'beats', 'beaten', 'above expectations',
        'below expectations', 'missed', 'miss', 'misses', 'short of',
        'fell short', 'underperformed', 'outperformed', 'surpassed'
    ]
    
    surprise_words = [
        'surprise', 'surprised', 'surprising', 'unexpected', 'unexpectedly',
        'strong', 'strongly', 'weak', 'weakly', 'disappointing', 'disappointed',
        'better than expected', 'worse than expected'
    ]
    
    financial_perf_words = [
        'revenue', 'revenues', 'sales', 'profit', 'profits', 'profitability',
        'margin', 'margins', 'growth', 'growing', 'earnings', 'eps',
        'ebitda', 'cash flow', 'cashflow', 'operating income'
    ]
    
    risk_loss_words = [
        'loss', 'losses', 'decline', 'declined', 'decrease', 'decreased',
        'decreasing', 'challenge', 'challenges', 'difficult', 'difficulty',
        'headwind', 'headwinds', 'risk', 'risks', 'concern', 'concerns'
    ]
    
    text_lower = transcript_text.lower()
    words = text_lower.split()
    total_words = len(words)
    
    guidance_count = sum(text_lower.count(word) for word in guidance_words)
    beat_miss_count = sum(text_lower.count(word) for word in beat_miss_words)
    surprise_count = sum(text_lower.count(word) for word in surprise_words)
    financial_perf_count = sum(text_lower.count(word) for word in financial_perf_words)
    risk_loss_count = sum(text_lower.count(word) for word in risk_loss_words)
    
    if total_words > 0:
        guidance_density = (guidance_count / total_words) * 100
        beat_miss_density = (beat_miss_count / total_words) * 100
        surprise_density = (surprise_count / total_words) * 100
        financial_perf_density = (financial_perf_count / total_words) * 100
        risk_loss_density = (risk_loss_count / total_words) * 100
    else:
        guidance_density = 0.0
        beat_miss_density = 0.0
        surprise_density = 0.0
        financial_perf_density = 0.0
        risk_loss_density = 0.0
    
    return {
        'guidance_count': guidance_count,
        'guidance_density': guidance_density,
        'beat_miss_count': beat_miss_count,
        'beat_miss_density': beat_miss_density,
        'surprise_count': surprise_count,
        'surprise_density': surprise_density,
        'financial_perf_count': financial_perf_count,
        'financial_perf_density': financial_perf_density,
        'risk_loss_count': risk_loss_count,
        'risk_loss_density': risk_loss_density
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
        - Financial keyword features (guidance, beat/miss, surprise, etc.)
        - Text statistics (char_count, word_count, sentence_count, etc.)
    """
    
    sentiment_features = extract_sentiment_features(transcript_text)
    uncertainty_features = extract_uncertainty_features(transcript_text)
    forward_features = extract_forward_looking_features(transcript_text)
    financial_features = extract_financial_keyword_features(transcript_text)
    text_stats = extract_text_statistics(transcript_text)
    
    all_features = {}
    all_features.update(sentiment_features)
    all_features.update(uncertainty_features)
    all_features.update(forward_features)
    all_features.update(financial_features)
    all_features.update(text_stats)
    
    interaction_features = create_interaction_features(
        sentiment_features, forward_features, financial_features
    )
    all_features.update(interaction_features)
    
    return all_features


def create_interaction_features(sentiment_features, forward_features, financial_features):
    """
    Create interaction features between sentiment, forward-looking, and financial keywords.
    
    Interactions can capture more nuanced signals:
    - Positive sentiment + guidance mentions = strong positive signal
    - Negative sentiment + risk/loss mentions = strong negative signal
    - Sentiment × forward-looking ratio = confidence in future outlook
    
    Parameters:
    -----------
    sentiment_features : dict
        Dictionary with sentiment features
    forward_features : dict
        Dictionary with forward/backward looking features
    financial_features : dict
        Dictionary with financial keyword features
        
    Returns:
    --------
    dict
        Dictionary with interaction features
    """
    
    interactions = {}
    
    doc_sentiment = sentiment_features.get('doc_sentiment', 0.0)
    forward_ratio = forward_features.get('forward_backward_ratio', 0.0)
    
    guidance_density = financial_features.get('guidance_density', 0.0)
    surprise_density = financial_features.get('surprise_density', 0.0)
    beat_miss_density = financial_features.get('beat_miss_density', 0.0)
    risk_loss_density = financial_features.get('risk_loss_density', 0.0)
    financial_perf_density = financial_features.get('financial_perf_density', 0.0)
    
    interactions['sentiment_x_guidance'] = doc_sentiment * guidance_density
    interactions['sentiment_x_surprise'] = doc_sentiment * surprise_density
    interactions['sentiment_x_beat_miss'] = doc_sentiment * beat_miss_density
    interactions['sentiment_x_risk_loss'] = doc_sentiment * risk_loss_density
    interactions['sentiment_x_financial_perf'] = doc_sentiment * financial_perf_density
    
    interactions['sentiment_x_forward_ratio'] = doc_sentiment * forward_ratio
    
    interactions['forward_ratio_x_guidance'] = forward_ratio * guidance_density
    interactions['forward_ratio_x_surprise'] = forward_ratio * surprise_density
    
    return interactions


def add_nlp_features_to_dataframe(df, transcript_column='transcript_text'):
    """
    Apply NLP feature extraction to all transcripts in a DataFrame.
    
    Takes a DataFrame with transcript text and adds columns for all
    NLP features (sentiment, uncertainty, forward/backward looking, financial keywords).
    
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
    total = len(df)
    
    for idx, row in df.iterrows():
        transcript_text = row[transcript_column]
        features = extract_all_nlp_features(transcript_text)
        nlp_features_list.append(features)
        
        processed = len(nlp_features_list)
        if processed % 10 == 0 or processed == total:
            print(f"  Processed {processed}/{total} transcripts...", end='\r')
    
    print()  # New line after progress
    features_df = pd.DataFrame(nlp_features_list)
    
    result_df = pd.concat([df, features_df], axis=1)
    
    return result_df

