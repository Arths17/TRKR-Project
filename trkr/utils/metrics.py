"""
TRKR Metrics Utilities
======================
Calculate and format F1 metrics: position errors, pit stop analysis, gaps, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List


def calculate_position_mae(predictions: pd.DataFrame, actuals: pd.DataFrame) -> float:
    """
    Calculate Mean Absolute Error between predicted and actual positions.
    
    Args:
        predictions: DataFrame with 'driver' and 'predicted_position'
        actuals: DataFrame with 'driver' and 'position'
    
    Returns:
        MAE value
    """
    merged = predictions.merge(actuals, on="driver", how="inner")
    if merged.empty:
        return 0.0
    
    errors = (merged['predicted_position'] - merged['position']).abs()
    return errors.mean()


def calculate_time_mae(predictions: pd.DataFrame, actuals: pd.DataFrame) -> float:
    """
    Calculate Mean Absolute Error in race time.
    
    Args:
        predictions: DataFrame with 'driver' and 'predicted_race_time'
        actuals: DataFrame with 'driver' and 'time'
    
    Returns:
        MAE in seconds
    """
    merged = predictions.merge(actuals, on="driver", how="inner")
    if merged.empty:
        return 0.0
    
    errors = (merged['predicted_race_time'] - merged['time']).abs()
    return errors.mean()


def check_winner_accuracy(predicted_winner: str, actual_winner: str) -> bool:
    """
    Check if predicted winner matches actual winner.
    
    Returns:
        True if match, False otherwise
    """
    return predicted_winner.upper() == actual_winner.upper()


def calculate_podium_accuracy(predicted_podium: List[str], actual_podium: List[str]) -> float:
    """
    Calculate podium accuracy as fraction of correct drivers.
    
    Args:
        predicted_podium: List of [P1, P2, P3] driver names
        actual_podium: List of [P1, P2, P3] driver names
    
    Returns:
        Fraction correct (0.0 to 1.0)
    """
    predicted = [d.upper() for d in predicted_podium[:3]]
    actual = [d.upper() for d in actual_podium[:3]]
    
    correct = sum(1 for p, a in zip(predicted, actual) if p == a)
    return correct / 3.0 if len(actual) == 3 else 0.0


def calculate_gap_accuracy(predicted_gaps: pd.DataFrame, actual_gaps: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze gap prediction accuracy per driver.
    
    Args:
        predicted_gaps: DataFrame with 'driver' and 'gap'
        actual_gaps: DataFrame with 'driver' and 'gap'
    
    Returns:
        Dict of {driver: gap_error_in_seconds}
    """
    merged = predicted_gaps.merge(actual_gaps, on="driver", how="inner", suffixes=('_pred', '_actual'))
    
    results = {}
    for _, row in merged.iterrows():
        gap_error = abs(row['gap_pred'] - row['gap_actual'])
        results[row['driver']] = gap_error
    
    return results


def detect_pit_stop_clusters(lap_times: pd.DataFrame, threshold: float = 1.5) -> Dict[str, List[int]]:
    """
    Detect pit stops by identifying lap time spikes.
    
    Args:
        lap_times: DataFrame with 'driver', 'lap', 'time'
        threshold: Multiplier for median lap time (pit stop = slower)
    
    Returns:
        Dict of {driver: [pit_stop_laps]}
    """
    pit_stops = {}
    
    for driver in lap_times['driver'].unique():
        driver_laps = lap_times[lap_times['driver'] == driver].sort_values('lap')
        
        if len(driver_laps) < 2:
            continue
        
        median_time = driver_laps['time'].median()
        slow_laps = driver_laps[driver_laps['time'] > median_time * threshold]['lap'].tolist()
        
        if slow_laps:
            pit_stops[driver] = slow_laps
    
    return pit_stops


def calculate_dnf_probability(gap_to_leader: float, confidence_score: float) -> float:
    """
    Estimate DNF (Did Not Finish) probability based on gap anomaly.
    
    Args:
        gap_to_leader: Gap in seconds
        confidence_score: Prediction confidence (0-100)
    
    Returns:
        DNF probability (0.0 to 1.0)
    """
    # Extreme gaps (>120s) suggest potential DNF
    if gap_to_leader > 120:
        base_prob = 0.7
    elif gap_to_leader > 90:
        base_prob = 0.4
    else:
        base_prob = 0.1
    
    # Adjust by confidence (low confidence = higher DNF prob)
    confidence_factor = (100 - confidence_score) / 100
    
    return min(1.0, base_prob + confidence_factor * 0.2)


def calculate_skill_metric(predictions_df: pd.DataFrame, actuals_df: pd.DataFrame) -> float:
    """
    Calculate overall prediction skill as composite metric (0-100).
    Combines position accuracy, winner prediction, podium accuracy.
    
    Args:
        predictions_df: DataFrame with predictions
        actuals_df: DataFrame with actual results
    
    Returns:
        Skill score (0-100)
    """
    position_mae = calculate_position_mae(predictions_df, actuals_df)
    position_score = max(0, 100 - (position_mae * 10))  # Normalize MAE to 0-100
    
    predicted_winner = predictions_df.iloc[0]['driver'] if not predictions_df.empty else ""
    actual_winner = actuals_df.iloc[0]['driver'] if not actuals_df.empty else ""
    winner_score = 100 if check_winner_accuracy(predicted_winner, actual_winner) else 0
    
    # Average the components
    skill = (position_score + winner_score) / 2
    return min(100, max(0, skill))


def format_metric(value: float, metric_type: str = "default") -> str:
    """
    Format a metric value with appropriate units and precision.
    
    Args:
        value: Numeric value
        metric_type: "mae", "time", "percent", "score"
    
    Returns:
        Formatted string
    """
    if metric_type == "mae":
        return f"±{value:.2f} positions"
    elif metric_type == "time":
        return f"{value:.1f}s"
    elif metric_type == "percent":
        return f"{value * 100:.1f}%"
    elif metric_type == "score":
        return f"{value:.0f}/100"
    else:
        return f"{value:.2f}"
