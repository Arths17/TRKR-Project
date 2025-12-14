"""
Model Trainer Module - Trains ML models for F1 race predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

import config


class F1ModelTrainer:
    """Trains and evaluates ML models for F1 predictions (classification and regression)"""
    
    def __init__(self, data: pd.DataFrame, feature_columns: List[str]):
        """
        Initialize the model trainer
        
        Args:
            data: Processed DataFrame with features and targets
            feature_columns: List of feature column names
        """
        self.data = data
        self.feature_columns = feature_columns
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        
    def prepare_data(self, target: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepare train/test split
        
        Args:
            target: Target variable name
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print(f"\nPreparing data for target: {target}")
        
        # Filter out rows with missing target
        data = self.data[self.data[target].notna()].copy()
        
        X = data[self.feature_columns].copy()
        y = data[target]
        
        # Handle any remaining NaN in features
        X = X.fillna(0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            shuffle=True
        )
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        # For regression target, show mean value
        try:
            print(f"Target mean: {float(y.mean()):.3f}")
        except Exception:
            pass
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      target: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler
        
        Args:
            X_train: Training features
            X_test: Test features
            target: Target name (for storing scaler)
            
        Returns:
            Scaled X_train, X_test
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[target] = scaler
        
        return X_train_scaled, X_test_scaled
    
    def train_xgboost_regressor(self, X_train: np.ndarray, y_train: pd.Series,
                                X_test: np.ndarray, y_test: pd.Series) -> xgb.XGBRegressor:
        """
        Train XGBoost regressor to predict race time.
        """
        print("\nTraining XGBoost regressor...")
        model = xgb.XGBRegressor(
            n_estimators=config.N_ESTIMATORS,
            learning_rate=config.LEARNING_RATE,
            max_depth=config.MAX_DEPTH,
            random_state=config.RANDOM_STATE,
            objective='reg:squaredlogerror'
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        return model
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: pd.Series,
                      X_test: np.ndarray, y_test: pd.Series) -> lgb.LGBMClassifier:
        """
        Train LightGBM classifier
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Trained LightGBM model
        """
        print("\nTraining LightGBM model...")
        
        # Calculate scale_pos_weight for imbalanced classes
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1
        
        model = lgb.LGBMClassifier(
            n_estimators=config.N_ESTIMATORS,
            learning_rate=config.LEARNING_RATE,
            max_depth=config.MAX_DEPTH,
            random_state=config.RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
            verbose=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='logloss'
        )
        
        return model
    
    def train_random_forest_regressor(self, X_train: np.ndarray, y_train: pd.Series) -> RandomForestRegressor:
        """Train Random Forest regressor (optional baseline)."""
        print("\nTraining Random Forest regressor...")
        model = RandomForestRegressor(
            n_estimators=config.N_ESTIMATORS,
            max_depth=config.MAX_DEPTH,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model
    
    def evaluate_regressor(self, model, X_test: np.ndarray, y_test: pd.Series, model_name: str) -> Dict:
        """Evaluate regression with MAE and feature importance."""
        print(f"\n=== Evaluating {model_name} ===")
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"MAE: {mae:.4f} seconds")
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_imp = pd.DataFrame({'feature': self.feature_columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
            print("\nTop 10 Important Features:")
            print(feature_imp.head(10))
            self.feature_importance[model_name] = feature_imp
        return {'mae': mae, 'predictions': y_pred}
    
    def train_position_model(self, target: str = 'FinishPosition') -> Dict:
        """
        Train XGBoost model to predict race time using ONLY pre-race features.
        Uses cross-validation for robust evaluation.
        
        Args:
            target: Target variable to predict
            
        Returns:
            Dictionary with model and evaluation metrics
        """
        print(f"\n{'='*50}\nTraining XGBoost regressor for target: {target}\n{'='*50}")
        X_train, X_test, y_train, y_test = self.prepare_data(target)
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test, target)
        model = self.train_xgboost_regressor(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Cross-validation on training set
        print("\nPerforming 5-fold cross-validation...")
        cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                   cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
        print(f"Cross-validation MAE: {cv_mae:.2f}s (±{cv_std:.2f}s)")
        
        eval_results = self.evaluate_regressor(model, X_test_scaled, y_test, f"{target}_xgboost")
        eval_results['cv_mae'] = cv_mae
        eval_results['cv_std'] = cv_std
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        self.feature_importance[target] = feature_importance
        
        model_key = f"{target}_xgboost"
        self.models[model_key] = model
        return {'model': model, 'scaler': self.scalers[target], 'evaluation': eval_results, 'features': self.feature_columns}
    
    
    def create_regression_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create finishing position target for prediction.
        Position is much better correlated with pre-race features than absolute race time.
        """
        df = df.copy()
        
        print("Creating regression target 'FinishPosition'...")
        
        # Use finishing position as target (already in data)
        if 'Position' in df.columns:
            df['FinishPosition'] = pd.to_numeric(df['Position'], errors='coerce')
            
            # Remove DNFs and invalid positions
            df = df[df['FinishPosition'].notna() & (df['FinishPosition'] > 0)]
            
            valid_count = len(df)
            print(f"  Valid records: {valid_count}")
            print(f"  Position range: {df['FinishPosition'].min():.0f} to {df['FinishPosition'].max():.0f}")
            print(f"  Average position: {df['FinishPosition'].mean():.2f}")
        else:
            print("  Error: Missing Position column")
            df['FinishPosition'] = np.nan
        
        return df
    
    def validate_on_season(self, season: int, target: str = 'FinishPosition') -> Dict:
        """
        Validate model on a specific season to check prediction accuracy
        
        Args:
            season: Year to validate on
            target: Target variable
            
        Returns:
            Dictionary with validation metrics
        """
        print(f"\n{'='*60}")
        print(f"Validating Model on {season} Season")
        print(f"{'='*60}")
        
        # Filter data for validation season
        val_data = self.data[self.data['Year'] == season].copy()
        
        if val_data.empty or target not in val_data.columns:
            print(f"No data available for season {season}")
            return {}
        
        # Remove rows with missing target
        val_data = val_data[val_data[target].notna()]
        
        if len(val_data) < 10:
            print(f"Insufficient data for validation ({len(val_data)} samples)")
            return {}
        
        # Prepare features
        X_val = val_data[self.feature_columns].copy().fillna(0)
        y_val = val_data[target]
        
        # Get model and scaler
        model_name = f"{target}_xgboost"
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return {}
        
        model = self.models[model_name]
        scaler = self.scalers.get(target)
        
        # Scale and predict
        X_val_scaled = scaler.transform(X_val) if scaler else X_val
        predictions = model.predict(X_val_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_val, predictions)
        
        # Position-specific metrics
        correct_within_1 = np.sum(np.abs(predictions - y_val) <= 1) / len(y_val)
        correct_within_2 = np.sum(np.abs(predictions - y_val) <= 2) / len(y_val)
        correct_within_3 = np.sum(np.abs(predictions - y_val) <= 3) / len(y_val)
        
        results = {
            'season': season,
            'samples': len(val_data),
            'mae': mae,
            'within_1_pos': correct_within_1 * 100,
            'within_2_pos': correct_within_2 * 100,
            'within_3_pos': correct_within_3 * 100
        }
        
        print(f"\nValidation Results for {season}:")
        print(f"  Samples: {results['samples']}")
        print(f"  MAE: {results['mae']:.2f} positions")
        print(f"  Within ±1 position: {results['within_1_pos']:.1f}%")
        print(f"  Within ±2 positions: {results['within_2_pos']:.1f}%")
        print(f"  Within ±3 positions: {results['within_3_pos']:.1f}%")
        
        return results
    
    def analyze_feature_importance(self, target: str = 'FinishPosition', top_n: int = 15) -> pd.DataFrame:
        """
        Detailed feature importance analysis with recommendations
        
        Args:
            target: Target variable
            top_n: Number of top features to analyze
            
        Returns:
            DataFrame with feature importance analysis
        """
        model_name = f"{target}_xgboost"
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return pd.DataFrame()
        
        model = self.models[model_name]
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calculate cumulative importance
        importance_df['cumulative'] = importance_df['importance'].cumsum()
        importance_df['percentage'] = importance_df['importance'] * 100
        
        print(f"\n{'='*60}")
        print(f"Feature Importance Analysis (Top {top_n})")
        print(f"{'='*60}")
        print(f"\n{'Feature':<30} {'Importance':>12} {'Cumulative':>12}")
        print("-" * 60)
        
        for idx, row in importance_df.head(top_n).iterrows():
            print(f"{row['feature']:<30} {row['percentage']:>11.2f}% {row['cumulative']*100:>11.1f}%")
        
        # Identify low-importance features for potential removal
        low_importance = importance_df[importance_df['importance'] < 0.01]
        if not low_importance.empty:
            print(f"\n⚠️  {len(low_importance)} features with <1% importance (consider removing):")
            print(f"  {', '.join(low_importance['feature'].head(10).tolist())}")
        
        return importance_df
    
    def save_models(self, directory: str = 'models'):
        """
        Save trained models to disk
        
        Args:
            directory: Directory to save models
        """
        model_dir = Path(directory)
        model_dir.mkdir(exist_ok=True)
        
        print(f"\nSaving models to {model_dir}...")
        
        for model_name, model in self.models.items():
            model_path = model_dir / f"{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {model_name}")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_path = model_dir / f"scaler_{scaler_name}.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        
        # Save feature columns
        feature_path = model_dir / "feature_columns.pkl"
        with open(feature_path, 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        print("All models saved successfully!")
    
    def load_models(self, directory: str = 'models'):
        """
        Load trained models from disk
        
        Args:
            directory: Directory containing saved models
        """
        model_dir = Path(directory)
        
        if not model_dir.exists():
            print(f"Model directory {model_dir} does not exist.")
            return
        
        print(f"\nLoading models from {model_dir}...")
        
        # Load models
        for model_file in model_dir.glob("*.pkl"):
            if 'scaler' not in model_file.name and 'feature' not in model_file.name:
                with open(model_file, 'rb') as f:
                    model_name = model_file.stem
                    self.models[model_name] = pickle.load(f)
                    print(f"Loaded {model_name}")
        
        # Load scalers
        for scaler_file in model_dir.glob("scaler_*.pkl"):
            with open(scaler_file, 'rb') as f:
                scaler_name = scaler_file.stem.replace('scaler_', '')
                self.scalers[scaler_name] = pickle.load(f)
        
        # Load feature columns
        feature_path = model_dir / "feature_columns.pkl"
        if feature_path.exists():
            with open(feature_path, 'rb') as f:
                self.feature_columns = pickle.load(f)
        
        print("All models loaded successfully!")


if __name__ == "__main__":
    # Test the trainer
    from data_fetcher import F1DataFetcher
    from data_processor import F1DataProcessor
    
    print("Fetching data...")
    fetcher = F1DataFetcher(seasons=[2024])
    raw_data = fetcher.fetch_historical_data()
    
    print("\nProcessing data...")
    processor = F1DataProcessor(raw_data)
    processed_data = processor.process()
    
    feature_cols = processor.get_feature_columns()
    
    print("\nTraining models...")
    trainer = F1ModelTrainer(processed_data, feature_cols)
    results = trainer.train_all_targets(['Won', 'Podium'])
    
    print("\nTraining completed!")
