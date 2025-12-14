"""
F1 Prediction System - Quick Reference Guide
============================================

TRAINING
--------
# Train on multiple seasons
python3 main.py --mode train --seasons 2022 2023 2024

# Validate model performance
python3 validate_model.py

PREDICTION
----------
# Predict specific 2025 race
python3 main.py --mode predict --year 2025 --race "Monaco" --load-models

# Export predictions to CSV
python3 main.py --mode predict --year 2025 --race "Qatar" --load-models --export

# Interactive mode
python3 main.py

CURRENT MODEL PERFORMANCE
-------------------------
✅ MAE: 1.35-1.52 positions (varies by season)
✅ Accuracy within ±2 positions: 75%
✅ Accuracy within ±3 positions: 87%
✅ Training data: 1,357 samples from 2022-2024
✅ Features: 28 pre-race features

KEY FEATURES (by importance)
-----------------------------
1. TeamPoints (74%) - Dominant predictor
2. CircuitAvgPosition (6%)
3. CircuitAvgPoints (5%)
4. TeamAvgPosition (3%)
5. GridPosition (2%)
6. RecentPosition (1.5%)
7. Tire performance features (1.1%)

FEATURES AVAILABLE
------------------
✅ Historical driver performance (last 5-10 races)
✅ Recent form (last 3-5 races)
✅ Circuit-specific history
✅ Team performance metrics
✅ Tire compound performance
✅ Grid position (when qualifying available)
⚠️ Weather data (available but low importance - 0%)

PREDICTION OUTPUT
-----------------
✅ Finishing positions (1-20)
✅ Realistic race times with progressive gaps
✅ Podium predictions with time gaps
✅ Dark horse candidates
✅ Team strength analysis
✅ CSV export for further analysis

NEXT IMPROVEMENTS
-----------------
🔄 Add pit stop strategy modeling
🔄 Incorporate live telemetry when available
🔄 Explore ensemble models (Random Forest + XGBoost)
🔄 Create web dashboard for visualization
🔄 Add championship standing predictions
🔄 Real-time model updates during 2025 season

FILES STRUCTURE
---------------
main.py              - Main orchestrator
data_fetcher.py      - FastF1 data fetching
data_processor.py    - Feature engineering
model_trainer.py     - XGBoost training
predictor.py         - Prediction generation
display.py           - Terminal formatting
validate_model.py    - Model validation
config.py            - Configuration

models/              - Trained models
predictions/         - Exported predictions
cache/               - FastF1 cache
"""

print(__doc__)
