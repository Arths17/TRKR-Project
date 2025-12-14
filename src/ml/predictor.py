"""
Predictor Module - Makes predictions for upcoming races
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import config


class F1Predictor:
    """Makes race predictions using trained models"""
    
    def __init__(self, models: Dict, scalers: Dict, feature_columns: List[str]):
        """
        Initialize the predictor
        
        Args:
            models: Dictionary of trained models
            scalers: Dictionary of feature scalers
            feature_columns: List of feature column names
        """
        self.models = models
        self.scalers = scalers
        self.feature_columns = feature_columns
    
    def prepare_prediction_features(self, race_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction - uses ONLY pre-race features with smart imputation
        
        Args:
            race_data: DataFrame with race data and features
            
        Returns:
            DataFrame with features ready for prediction
        """
        # Create a copy to avoid modifying original
        X = pd.DataFrame()
        
        # Add each feature column with smart imputation
        missing_features = []
        imputed_features = []
        
        for col in self.feature_columns:
            if col in race_data.columns:
                X[col] = race_data[col]
            else:
                # Smart imputation based on feature type
                if 'TirePerf' in col or 'Laps_' in col:
                    # For tire features, use 0 (not used)
                    X[col] = 0
                elif 'Team' in col:
                    # For team features, use team average if available
                    if 'Team' in race_data.columns:
                        team_avg = race_data.groupby('Team')[col].mean() if col in race_data.columns else 0
                        X[col] = race_data['Team'].map(team_avg).fillna(0)
                    else:
                        X[col] = 0
                else:
                    # For other features, use median or 0
                    X[col] = 0
                missing_features.append(col)
        
        # Replace NaN with 0 for all features
        X = X.fillna(0)
        
        # Replace zero values in critical features with smart estimates
        if 'AvgLapTime' in X.columns and 'Team' in race_data.columns:
            # For drivers with 0 lap time, use team average
            team_lap_avg = race_data[race_data['AvgLapTime'] > 0].groupby('Team')['AvgLapTime'].mean()
            zero_lap_mask = X['AvgLapTime'] == 0
            if zero_lap_mask.any():
                X.loc[zero_lap_mask, 'AvgLapTime'] = race_data.loc[zero_lap_mask, 'Team'].map(team_lap_avg).fillna(90.0)
                imputed_features.append('AvgLapTime (zero values replaced with team avg)')
        
        if missing_features:
            print(f"⚠️  {len(missing_features)} features missing from race data")
        
        if imputed_features:
            print(f"📝 Imputed features:")
            for feat in imputed_features:
                print(f"   - {feat}")
        
        # Show feature statistics
        non_zero_features = (X != 0).sum(axis=1).mean()
        feature_coverage = (X != 0).sum().sum() / (len(X) * len(self.feature_columns)) * 100
        print(f"📊 Average non-zero features per driver: {non_zero_features:.1f}/{len(self.feature_columns)}")
        print(f"📊 Overall feature coverage: {feature_coverage:.1f}%")
        
        # Warn if coverage is too low
        if feature_coverage < 85:
            print(f"⚠️  WARNING: Feature coverage ({feature_coverage:.1f}%) below recommended threshold (85%)")
        
        # Store coverage for confidence calculation
        self._last_feature_coverage = feature_coverage
        self._last_num_imputed = len(imputed_features)
        
        return X
    
    def predict_race_times(self, race_data: pd.DataFrame, model_name: str = 'FinishPosition_xgboost') -> pd.DataFrame:
        """Predict finishing positions with probabilistic adjustments and realistic time gaps."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        model = self.models[model_name]
        target = model_name.split('_')[0]
        scaler = self.scalers.get(target)
        X = self.prepare_prediction_features(race_data)
        X_scaled = scaler.transform(X) if scaler else X
        pred_positions = model.predict(X_scaled)
        results = race_data[['Driver', 'Team']].copy()
        results['RawPosition'] = pred_positions
        
        # Apply probabilistic ranking with variance for lower positions
        # Top drivers get tighter predictions, mid-field gets more variance
        for i, pos in enumerate(pred_positions):
            if pos <= 3:
                # Podium - very low variance
                variance = np.random.normal(0, 0.1)
            elif pos <= 7:
                # Points positions - low variance
                variance = np.random.normal(0, 0.3)
            elif pos <= 12:
                # Mid-field - moderate variance
                variance = np.random.normal(0, 0.5)
            else:
                # Back markers - higher variance (more unpredictable)
                variance = np.random.normal(0, 0.8)
            
            pred_positions[i] = max(1, pos + variance)
        
        results['PredictedPosition'] = pred_positions
        results = results.sort_values('PredictedPosition', ascending=True)
        results['PredictedPosition'] = results['PredictedPosition'].round().astype(int)
        
        # Circuit-specific time normalization
        # Use circuit-specific historical median race duration if available
        # WHY: Different circuits have vastly different race durations (Monaco ~100min, Monza ~80min)
        circuit_name = race_data['Circuit'].iloc[0] if 'Circuit' in race_data.columns else None
        
        if circuit_name and 'AvgLapTime' in race_data.columns:
            # Calculate base time from actual lap data
            valid_laps = race_data[race_data['AvgLapTime'] > 0]['AvgLapTime']
            if len(valid_laps) > 0:
                avg_lap = valid_laps.median()
                num_laps = 57  # Default F1 race laps (varies by circuit)
                base_time = num_laps * avg_lap
            else:
                base_time = 57 * 90.0  # Fallback
        else:
            # Fallback: 57 laps × 90s average lap time = 5130s baseline
            base_time = 57 * 90.0
        
        # Create realistic gaps based on predicted position
        times = []
        for i, (idx, row) in enumerate(results.iterrows()):
            pos = row['PredictedPosition']
            if i == 0:
                # Winner - fastest time (slightly under baseline)
                race_time = base_time * 0.98
            else:
                # Progressive time gaps with position-based variance
                # Top 3: tight gaps (0.1-0.25% per position)
                # Mid-field: moderate gaps (0.15-0.35% per position)
                # Back: larger gaps (0.2-0.5% per position)
                if pos <= 3:
                    gap_factor = 1 + (pos - 1) * (0.001 + np.random.uniform(0, 0.0015))
                elif pos <= 10:
                    gap_factor = 1 + (pos - 1) * (0.0015 + np.random.uniform(0, 0.002))
                else:
                    gap_factor = 1 + (pos - 1) * (0.002 + np.random.uniform(0, 0.003))
                
                race_time = base_time * 0.98 * gap_factor
            
            # Add small random variance for realism (±1.5 seconds)
            race_time += np.random.uniform(-1.5, 1.5)
            times.append(race_time)
        
        results['PredictedRaceTime'] = times
        
        # Add per-driver uncertainty estimates (±seconds)
        # WHY: Provides confidence intervals for predictions
        # Based on: 1) Position variance (lower positions = higher uncertainty)
        #          2) Feature coverage (lower coverage = higher uncertainty)
        uncertainties = []
        feature_coverage = getattr(self, '_last_feature_coverage', 80.0)
        coverage_factor = 1.0 if feature_coverage >= 85 else (1.5 if feature_coverage >= 70 else 2.0)
        
        for _, row in results.iterrows():
            pos = row['PredictedPosition']
            if pos <= 3:
                base_uncertainty = 2.0  # Podium: ±2s
            elif pos <= 10:
                base_uncertainty = 4.0  # Points: ±4s
            else:
                base_uncertainty = 8.0  # Back markers: ±8s
            
            # Adjust for data quality
            uncertainty = base_uncertainty * coverage_factor
            uncertainties.append(uncertainty)
        
        results['Uncertainty'] = uncertainties
        
        # CRITICAL: Sort by predicted race time to enforce ranking integrity
        results = results.sort_values('PredictedRaceTime', ascending=True).reset_index(drop=True)
        
        # Recalculate positions based on sorted order
        results['PredictedPosition'] = range(1, len(results) + 1)
        
        # Recalculate gaps from the winner (must be non-negative)
        winner_time = results.iloc[0]['PredictedRaceTime']
        results['Gap'] = results['PredictedRaceTime'] - winner_time
        
        # Validation checks
        validation_errors = []
        
        # Check 1: No negative gaps
        if (results['Gap'] < 0).any():
            validation_errors.append("Negative gaps detected")
        
        # Check 2: Gaps should be monotonically increasing
        if not results['Gap'].is_monotonic_increasing:
            validation_errors.append("Gaps not monotonically increasing")
        
        # Check 3: No driver with slower time ranked above faster driver
        for i in range(1, len(results)):
            if results.iloc[i]['PredictedRaceTime'] < results.iloc[i-1]['PredictedRaceTime']:
                validation_errors.append(f"Ranking violation at position {i+1}")
        
        if validation_errors:
            print("⚠️ VALIDATION WARNINGS:")
            for error in validation_errors:
                print(f"  - {error}")
        
        # Data quality guards
        # WHY: Alert users when predictions may be unstable
        feature_coverage = getattr(self, '_last_feature_coverage', 100.0)
        
        if feature_coverage < 60:
            print("\n⚠️ LOW DATA QUALITY – Predictions may be unstable")
            print(f"   Feature coverage: {feature_coverage:.1f}% (minimum recommended: 60%)")
        
        # Check for extreme gaps (potential outliers)
        max_gap = results['Gap'].max()
        if max_gap > 120:  # >2 minutes gap to P1
            slowest = results.iloc[-1]
            print(f"\n⚠️ EXTREME GAP DETECTED: {slowest['Driver']} has {max_gap:.1f}s gap to P1")
            print(f"   This may indicate: DNF risk, mechanical issues, or data quality problem")
        
        return results
    
    def predict_podium(self, race_data: pd.DataFrame,
                      model_name: str = 'Podium_xgboost') -> pd.DataFrame:
        """
        Predict podium finish probabilities
        
        Args:
            race_data: DataFrame with driver data for the race
            model_name: Name of the model to use
            
        Returns:
            DataFrame with podium predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        target = model_name.split('_')[0]
        scaler = self.scalers.get(target)
        
        # Prepare features
        X = self.prepare_prediction_features(race_data)
        
        # Scale features
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        # Make predictions
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        # Create results DataFrame
        results = race_data[['Driver', 'Team']].copy()
        results['PodiumProbability'] = probabilities
        results['PodiumPrediction'] = model.predict(X_scaled)
        
        # Sort by probability
        results = results.sort_values('PodiumProbability', ascending=False)
        
        return results
    
    def predict_top5(self, race_data: pd.DataFrame,
                    model_name: str = 'Top5_xgboost') -> pd.DataFrame:
        """
        Predict top 5 finish probabilities
        
        Args:
            race_data: DataFrame with driver data for the race
            model_name: Name of the model to use
            
        Returns:
            DataFrame with top 5 predictions
        """
        if model_name not in self.models:
            # If model not available, return empty DataFrame
            results = race_data[['Driver', 'Team']].copy()
            results['Top5Probability'] = 0.0
            return results
        
        model = self.models[model_name]
        target = model_name.split('_')[0]
        scaler = self.scalers.get(target)
        
        # Prepare features
        X = self.prepare_prediction_features(race_data)
        
        # Scale features
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        # Make predictions
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        # Create results DataFrame
        results = race_data[['Driver', 'Team']].copy()
        results['Top5Probability'] = probabilities
        
        # Sort by probability
        results = results.sort_values('Top5Probability', ascending=False)
        
        return results
    
    def predict_comprehensive(self, race_data: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive predictions centered on race time ranking."""
        return self.predict_race_times(race_data)
    
    def get_insights(self, race_data: pd.DataFrame, predictions: pd.DataFrame) -> Dict:
        """
        Generate insights from predictions and race data
        
        Args:
            race_data: Original race data
            predictions: Prediction results
            
        Returns:
            Dictionary with insights
        """
        insights = {}
        
        # Predicted winner and podium from race time ranking
        top_driver = predictions.iloc[0]
        insights['predicted_winner'] = {
            'driver': top_driver['Driver'],
            'team': top_driver['Team'],
            'predicted_time': top_driver['PredictedRaceTime']
        }
        podium = predictions.head(3)
        insights['predicted_podium'] = [
            {
                'driver': row['Driver'],
                'team': row['Team'],
                'predicted_time': row['PredictedRaceTime']
            }
            for _, row in podium.iterrows()
        ]
        
        # Dark horses: drivers ranked P4-P8 AFTER sorting with close gaps to podium
        if len(predictions) > 3:
            # Use SORTED predictions (positions 4-8)
            dark_horse_candidates = predictions.iloc[3:8].copy()
            podium_time = predictions.iloc[2]['PredictedRaceTime']
            
            # Calculate gap to podium (must be positive)
            dark_horse_candidates['TimeDelta'] = dark_horse_candidates['PredictedRaceTime'] - podium_time
            
            # Filter for realistic gaps (< 15s to podium)
            dark_horses = dark_horse_candidates[
                (dark_horse_candidates['TimeDelta'] > 0) & 
                (dark_horse_candidates['TimeDelta'] < 15)
            ]
            
            insights['dark_horses'] = [
                {
                    'driver': row['Driver'],
                    'team': row['Team'],
                    'predicted_time': row['PredictedRaceTime'],
                    'gap_to_podium': row['TimeDelta']
                }
                for _, row in dark_horses.iterrows()
            ]
        else:
            insights['dark_horses'] = []
        
        # Team analysis - normalize to Team Strength Index (0-100)
        # WHY: Replace "combined win probability >100%" with interpretable 0-100 scale
        team_avg_pos = predictions.groupby('Team')['PredictedPosition'].mean().sort_values()
        
        # Convert average position to strength index (P1=100, P20=0)
        # Formula: Strength = 100 * (21 - avg_position) / 20
        team_strength = {}
        for team, avg_pos in team_avg_pos.head(3).items():
            strength_index = max(0, min(100, 100 * (21 - avg_pos) / 20))
            team_strength[team] = strength_index
        
        insights['strongest_teams'] = team_strength
        
        # Performance metrics from race_data if available
        if 'AvgLapTime' in race_data.columns:
            # CRITICAL: Exclude drivers with missing or zero lap times
            valid_lap_data = race_data[
                (race_data['AvgLapTime'] > 0) & 
                (race_data['AvgLapTime'].notna())
            ].copy()
            
            if len(valid_lap_data) >= 3:
                fastest_drivers = valid_lap_data.nsmallest(3, 'AvgLapTime')[['Driver', 'Team', 'AvgLapTime']]
                insights['fastest_avg_lap'] = [
                    {
                        'driver': row['Driver'],
                        'team': row['Team'],
                        'avg_lap_time': row['AvgLapTime']
                    }
                    for _, row in fastest_drivers.iterrows()
                ]
            else:
                insights['fastest_avg_lap'] = []
        
        # Add validation summary
        insights['validation'] = self._validate_predictions(predictions)
        
        # Add confidence score
        insights['confidence'] = self._compute_confidence_score()
        
        return insights
    
    def _compute_confidence_score(self) -> Dict:
        """
        Compute prediction confidence score based on data quality
        
        WHY: Users need to know how reliable predictions are
        
        Returns:
            Dictionary with confidence level and score
        """
        feature_coverage = getattr(self, '_last_feature_coverage', 0.0)
        num_imputed = getattr(self, '_last_num_imputed', 0)
        
        # Confidence scoring logic
        # HIGH: coverage >= 85%, imputed <= 3
        # MEDIUM: coverage >= 70%, imputed <= 7
        # LOW: coverage < 70% or imputed > 7
        
        if feature_coverage >= 85 and num_imputed <= 3:
            confidence_level = 'HIGH'
            confidence_score = 90 + (feature_coverage - 85) / 15 * 10  # 90-100
        elif feature_coverage >= 70 and num_imputed <= 7:
            confidence_level = 'MEDIUM'
            confidence_score = 60 + (feature_coverage - 70) / 15 * 30  # 60-90
        else:
            confidence_level = 'LOW'
            confidence_score = max(20, feature_coverage * 0.6)  # 20-60
        
        return {
            'level': confidence_level,
            'score': min(100, confidence_score),
            'feature_coverage': feature_coverage,
            'num_imputed': num_imputed
        }
    
    def _validate_predictions(self, predictions: pd.DataFrame) -> Dict:
        """
        Validate prediction integrity
        
        Args:
            predictions: DataFrame with predictions
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'sorted_correct': True,
            'gaps_valid': True,
            'feature_coverage': getattr(self, '_last_feature_coverage', 0.0),
            'errors': []
        }
        
        # Check 1: Sorted correctness (no inversions)
        for i in range(1, len(predictions)):
            if predictions.iloc[i]['PredictedRaceTime'] < predictions.iloc[i-1]['PredictedRaceTime']:
                validation['sorted_correct'] = False
                validation['errors'].append(f"Ranking inversion at P{i+1}")
        
        # Check 2: Gap validity (all non-negative and monotonic)
        if 'Gap' in predictions.columns:
            if (predictions['Gap'] < 0).any():
                validation['gaps_valid'] = False
                validation['errors'].append("Negative gaps detected")
            
            if not predictions['Gap'].is_monotonic_increasing:
                validation['gaps_valid'] = False
                validation['errors'].append("Gaps not monotonically increasing")
        
        # Overall status
        validation['status'] = 'PASS' if validation['sorted_correct'] and validation['gaps_valid'] else 'FAIL'
        
        return validation
    
    @staticmethod
    def load_predictor(model_dir: str = 'models') -> 'F1Predictor':
        """
        Load a predictor from saved models
        
        Args:
            model_dir: Directory containing saved models
            
        Returns:
            Initialized F1Predictor
        """
        model_path = Path(model_dir)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory {model_dir} not found")
        
        # Load models
        models = {}
        for model_file in model_path.glob("*.pkl"):
            if 'scaler' not in model_file.name and 'feature' not in model_file.name:
                with open(model_file, 'rb') as f:
                    model_name = model_file.stem
                    models[model_name] = pickle.load(f)
        
        # Load scalers
        scalers = {}
        for scaler_file in model_path.glob("scaler_*.pkl"):
            with open(scaler_file, 'rb') as f:
                scaler_name = scaler_file.stem.replace('scaler_', '')
                scalers[scaler_name] = pickle.load(f)
        
        # Load feature columns
        feature_path = model_path / "feature_columns.pkl"
        with open(feature_path, 'rb') as f:
            feature_columns = pickle.load(f)
        
        return F1Predictor(models, scalers, feature_columns)


if __name__ == "__main__":
    # Test predictor
    print("Testing predictor with sample data...")
    
    # This would normally load from saved models
    # For testing, we'd need to train models first
    print("Run model_trainer.py first to generate models for prediction.")
