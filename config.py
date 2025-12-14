"""
Configuration file for F1 Prediction System
"""

# Data settings
CURRENT_SEASON = 2024
HISTORICAL_SEASONS = [2022, 2023, 2024]  # Seasons to fetch for training
MIN_LAPS_FOR_ANALYSIS = 5  # Minimum laps to include in analysis

# Feature engineering settings
RECENT_RACES_WINDOW = 5  # Number of recent races for performance metrics
MIN_RACES_FOR_PREDICTION = 3  # Minimum races needed before making predictions

# Model settings
MODEL_TYPE = 'xgboost'  # Options: 'xgboost', 'lightgbm', 'random_forest'
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200
LEARNING_RATE = 0.1
MAX_DEPTH = 6

# Display settings
TOP_N_PREDICTIONS = 10  # Number of drivers to show in predictions
SHOW_PROBABILITIES = True

# Cache settings
CACHE_DIR = 'cache'
USE_CACHE = True

# Export settings
EXPORT_PREDICTIONS = True
EXPORT_DIR = 'predictions'
