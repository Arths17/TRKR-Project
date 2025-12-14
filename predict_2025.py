#!/usr/bin/env python3
"""
Quick script to predict a 2025 race using trained models
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_fetcher import F1DataFetcher
from data_processor import F1DataProcessor
from predictor import F1Predictor
from display import F1Display

def predict_2025_race(race_identifier):
    """
    Predict a 2025 race
    
    Args:
        race_identifier: Race name or round number
    """
    display = F1Display()
    display.show_banner()
    
    print(f"\n{'='*70}")
    print(f"Predicting 2025 Race: {race_identifier}")
    print(f"{'='*70}\n")
    
    # Fetch 2025 race data WITH qualifying
    print("→ Fetching 2025 race data (including qualifying)...")
    fetcher = F1DataFetcher(seasons=[2025])
    race_data = fetcher.fetch_race_with_qualifying(2025, race_identifier)
    
    if not race_data or race_data.get('results', None) is None or race_data['results'].empty:
        print("❌ Could not fetch race data. Check if race has occurred or try a different identifier.")
        return
    
    # Process the data
    print("→ Processing features...")
    processor = F1DataProcessor(race_data)
    processed = processor.process()
    
    if processed.empty:
        print("❌ Failed to process race data")
        return
    
    # Load trained models
    print("→ Loading trained models...")
    try:
        predictor = F1Predictor.load_predictor('models')
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        print("   Have you run training first? (python3 main.py --mode train --seasons 2023 2024)")
        return
    
    # Get race info
    race_info = processed.iloc[0]
    race_name = race_info['Race']
    circuit = race_info['Circuit']
    date = str(race_info['Date'])
    
    display.show_race_header(race_name, circuit, date)
    
    # Make predictions
    print("→ Generating predictions...")
    predictions = predictor.predict_comprehensive(processed)
    
    # Get insights
    insights = predictor.get_insights(processed, predictions)
    
    # Display results
    display.show_winner_prediction(insights)
    display.show_podium_prediction(insights)
    display.show_predictions_table(predictions, top_n=15)
    display.show_insights(insights)
    
    # Export
    filename = f"predictions_2025_{race_name.replace(' ', '_')}.csv"
    filepath = display.export_predictions_table(predictions, filename)
    print(f"\n✓ Predictions exported to: {filepath}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict 2025 F1 Race')
    parser.add_argument('race', help='Race name or round number (e.g., "Abu Dhabi" or 24)')
    
    args = parser.parse_args()
    
    # Try to convert to int if it's a number
    try:
        race_id = int(args.race)
    except ValueError:
        race_id = args.race
    
    predict_2025_race(race_id)
