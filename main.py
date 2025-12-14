"""
Main F1 Prediction Program
Terminal-based F1 race prediction system using AI/ML
"""

import argparse
from pathlib import Path
import pandas as pd
import sys
import warnings
warnings.filterwarnings('ignore')

import config
from data_fetcher import F1DataFetcher
from data_processor import F1DataProcessor
from model_trainer import F1ModelTrainer
from predictor import F1Predictor
from display import F1Display


class F1PredictionSystem:
    """Main orchestrator for F1 prediction system"""
    
    def __init__(self):
        """Initialize the prediction system"""
        self.display = F1Display()
        self.fetcher = None
        self.processor = None
        self.trainer = None
        self.predictor = None
        self.processed_data = None
        
    def fetch_and_process_data(self, seasons=None):
        """
        Fetch and process historical F1 data
        
        Args:
            seasons: List of seasons to fetch (default from config)
        """
        self.display.show_progress("Fetching historical F1 data...")
        
        # Fetch data
        seasons = seasons or config.HISTORICAL_SEASONS
        self.fetcher = F1DataFetcher(seasons=seasons)
        raw_data = self.fetcher.fetch_historical_data()
        
        if raw_data['results'].empty:
            self.display.show_error("No data fetched. Please check your internet connection or FastF1 API.")
            return False
        
        # Process data
        self.display.show_progress("Processing data and engineering features...")
        self.processor = F1DataProcessor(raw_data)
        self.processed_data = self.processor.process()
        
        # Show data summary
        data_summary = {
            'total_races': len(self.processed_data['Race'].unique()),
            'seasons': ', '.join(map(str, seasons)),
            'total_laps': len(raw_data['laps']) if 'laps' in raw_data else 0,
            'unique_drivers': len(self.processed_data['Driver'].unique()),
            'model_type': config.MODEL_TYPE
        }
        self.display.show_data_summary(data_summary)
        
        return True
    
    def train_models(self, save_models=True):
        """
        Train ML models for predictions
        
        Args:
            save_models: Whether to save trained models to disk
        """
        if self.processed_data is None or self.processed_data.empty:
            self.display.show_error("No processed data available. Run fetch_and_process_data first.")
            return False
        
        self.display.show_progress("Training AI models...")
        
        # Get PRE-RACE feature columns only (excludes post-race features)
        feature_cols = self.processor.get_feature_columns(pre_race_only=True)
        
        if not feature_cols:
            self.display.show_error("No feature columns available for training.")
            return False
        
        print(f"\n📊 Using {len(feature_cols)} pre-race features for training")
        print(f"Features include: qualifying, historical performance, team stats, circuit history")
        
        # Create regression target first
        temp_trainer = F1ModelTrainer(self.processed_data, feature_cols)
        self.processed_data = temp_trainer.create_regression_target(self.processed_data)
        
        # Now train with updated data
        self.trainer = F1ModelTrainer(self.processed_data, feature_cols)
        results = self.trainer.train_position_model('FinishPosition')
        # Show MAE
        mae = results['evaluation'].get('mae', None)
        if mae is not None:
            self.display.show_info(f"Model MAE: {mae:.2f} positions")
        
        # Save models
        if save_models:
            self.display.show_progress("Saving models...")
            self.trainer.save_models()
            self.display.show_info("Models saved successfully!")
        
        return True
    
    def load_models(self, model_dir='models'):
        """
        Load pre-trained models
        
        Args:
            model_dir: Directory containing saved models
        """
        try:
            self.display.show_progress("Loading pre-trained models...")
            self.predictor = F1Predictor.load_predictor(model_dir)
            self.display.show_info("Models loaded successfully!")
            return True
        except Exception as e:
            self.display.show_error(f"Failed to load models: {e}")
            return False
    
    def predict_race(self, year, race_identifier, export=False):
        """
        Make predictions for a specific race
        
        Args:
            year: Season year
            race_identifier: Race name or round number
            export: Whether to export predictions to CSV
        """
        # Ensure we have a predictor
        if self.predictor is None:
            if self.trainer is not None:
                # Create predictor from trainer
                self.predictor = F1Predictor(
                    self.trainer.models,
                    self.trainer.scalers,
                    self.trainer.feature_columns
                )
            else:
                self.display.show_error("No models available. Train or load models first.")
                return
        
        # Fetch race data for prediction
        self.display.show_progress(f"Fetching data for {year} - {race_identifier}...")
        
        if self.fetcher is None:
            self.fetcher = F1DataFetcher(seasons=[year])
        
        # Try to fetch with qualifying data first (for better predictions)
        try:
            race_data_dict = self.fetcher.fetch_race_with_qualifying(year, race_identifier)
            if race_data_dict.get('qualifying', pd.DataFrame()).empty:
                print("⚠️  No qualifying data available, using race data only")
        except Exception as e:
            print(f"⚠️  Could not fetch qualifying: {e}")
            race_data_dict = self.fetcher.fetch_all_race_data(year, race_identifier)
        
        if not race_data_dict or race_data_dict.get('results', pd.DataFrame()).empty:
            self.display.show_error(f"Could not fetch data for race: {year} - {race_identifier}")
            return
        
        # Process race data
        temp_processor = F1DataProcessor(race_data_dict)
        race_processed = temp_processor.process()
        
        if race_processed.empty:
            self.display.show_error("Failed to process race data.")
            return
        
        # Get race info
        race_info = race_processed.iloc[0]
        race_name = race_info['Race']
        circuit = race_info['Circuit']
        date = str(race_info['Date'])
        
        # Display race header
        self.display.show_race_header(race_name, circuit, date)
        
        # Make predictions
        self.display.show_progress("Generating predictions...")
        predictions = self.predictor.predict_comprehensive(race_processed)
        
        # Get insights
        insights = self.predictor.get_insights(race_processed, predictions)
        
        # Display results
        self.display.show_winner_prediction(insights)
        self.display.show_podium_prediction(insights)
        self.display.show_predictions_table(predictions)
        self.display.show_insights(insights)
        
        # Export if requested
        if export and config.EXPORT_PREDICTIONS:
            filename = f"predictions_{race_name.replace(' ', '_')}_{year}.csv"
            filepath = self.display.export_predictions_table(predictions, filename)
            self.display.show_info(f"Predictions exported to: {filepath}")
    
    def predict_next_race(self, export=False):
        """
        Predict the next upcoming race
        
        Args:
            export: Whether to export predictions
        """
        current_year = config.CURRENT_SEASON
        
        # Get current season schedule
        self.display.show_progress(f"Finding next race in {current_year} season...")
        
        import fastf1
        try:
            schedule = fastf1.get_event_schedule(current_year)
            
            # Find next race (races with future dates)
            from datetime import datetime
            now = datetime.now()
            
            future_races = schedule[pd.to_datetime(schedule['EventDate']) > now]
            
            if future_races.empty:
                self.display.show_warning(f"No upcoming races found in {current_year} season.")
                # Try last race instead
                last_race = schedule.iloc[-1]
                race_round = last_race['RoundNumber']
                self.display.show_info(f"Showing predictions for last race: {last_race['EventName']}")
            else:
                next_race = future_races.iloc[0]
                race_round = next_race['RoundNumber']
                self.display.show_info(f"Next race: {next_race['EventName']}")
            
            # Predict
            self.predict_race(current_year, race_round, export=export)
            
        except Exception as e:
            self.display.show_error(f"Failed to get schedule: {e}")
    
    def interactive_mode(self):
        """Run the system in interactive mode"""
        self.display.show_banner()
        
        while True:
            self.display.console.print("\n[bold cyan]Options:[/bold cyan]")
            self.display.console.print("1. Fetch and process data")
            self.display.console.print("2. Train models")
            self.display.console.print("3. Load pre-trained models")
            self.display.console.print("4. Predict specific race")
            self.display.console.print("5. Predict next upcoming race")
            self.display.console.print("6. Exit")
            
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == '1':
                self.fetch_and_process_data()
            
            elif choice == '2':
                if self.processed_data is None:
                    self.display.show_warning("No data available. Fetching data first...")
                    if not self.fetch_and_process_data():
                        continue
                self.train_models(save_models=True)
            
            elif choice == '3':
                self.load_models()
            
            elif choice == '4':
                year = input("Enter year: ").strip()
                race = input("Enter race name or round number: ").strip()
                export = input("Export to CSV? (y/n): ").strip().lower() == 'y'
                
                try:
                    year = int(year)
                    try:
                        race = int(race)
                    except ValueError:
                        pass  # Keep as string if not a number
                    
                    self.predict_race(year, race, export=export)
                except ValueError:
                    self.display.show_error("Invalid year format.")
            
            elif choice == '5':
                export = input("Export to CSV? (y/n): ").strip().lower() == 'y'
                self.predict_next_race(export=export)
            
            elif choice == '6':
                self.display.console.print("\n[bold green]Thank you for using F1 Prediction System! 🏁[/bold green]\n")
                break
            
            else:
                self.display.show_error("Invalid choice. Please select 1-6.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='F1 Race Prediction System - AI-powered Formula 1 analytics'
    )
    
    parser.add_argument(
        '--mode',
        choices=['train', 'predict', 'interactive'],
        default='interactive',
        help='Operation mode: train models, predict race, or interactive'
    )
    
    parser.add_argument(
        '--year',
        type=int,
        help='Year for prediction'
    )
    
    parser.add_argument(
        '--race',
        help='Race name or round number for prediction'
    )
    
    parser.add_argument(
        '--seasons',
        nargs='+',
        type=int,
        help='Seasons to fetch for training (e.g., 2022 2023 2024)'
    )
    
    parser.add_argument(
        '--export',
        action='store_true',
        help='Export predictions to CSV'
    )
    
    parser.add_argument(
        '--load-models',
        action='store_true',
        help='Load pre-trained models instead of training'
    )
    
    args = parser.parse_args()
    
    # Initialize system
    system = F1PredictionSystem()
    
    if args.mode == 'interactive':
        system.interactive_mode()
    
    elif args.mode == 'train':
        system.display.show_banner()
        
        # Fetch and process
        if not system.fetch_and_process_data(seasons=args.seasons):
            return
        
        # Train
        system.train_models(save_models=True)
    
    elif args.mode == 'predict':
        system.display.show_banner()
        
        # Load or train models
        if args.load_models:
            if not system.load_models():
                return
        else:
            if not system.fetch_and_process_data(seasons=args.seasons):
                return
            if not system.train_models():
                return
        
        # Make prediction
        if args.year and args.race:
            system.predict_race(args.year, args.race, export=args.export)
        else:
            system.predict_next_race(export=args.export)


if __name__ == "__main__":
    main()
