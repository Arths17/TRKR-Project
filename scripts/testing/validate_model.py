#!/usr/bin/env python3
"""
Model Validation Script
Validates F1 prediction model on recent seasons and analyzes feature importance
"""

from data_fetcher import F1DataFetcher
from data_processor import F1DataProcessor
from model_trainer import F1ModelTrainer
import pickle

def main():
    print("=" * 70)
    print("F1 PREDICTION MODEL VALIDATION")
    print("=" * 70)
    
    # Load trained model
    print("\n📊 Loading trained model...")
    with open('models/FinishPosition_xgboost.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/scaler_FinishPosition.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('models/feature_columns.pkl', 'rb') as f:
        features = pickle.load(f)
    
    print(f"✅ Model loaded with {len(features)} features")
    
    # Fetch training data
    print("\n📥 Fetching validation data...")
    fetcher = F1DataFetcher(seasons=[2022, 2023, 2024])
    raw_data = fetcher.fetch_historical_data()
    
    processor = F1DataProcessor(raw_data)
    processed = processor.process()
    
    # Get pre-race features
    feature_cols = processor.get_feature_columns(pre_race_only=True)
    
    # Create regression target
    temp_trainer = F1ModelTrainer(processed, feature_cols)
    processed = temp_trainer.create_regression_target(processed)
    
    # Create trainer with loaded models
    trainer = F1ModelTrainer(processed, feature_cols)
    trainer.models = {'FinishPosition_xgboost': model}
    trainer.scalers = {'FinishPosition': scaler}
    trainer.feature_columns = features
    
    # Validate on each season
    print("\n" + "=" * 70)
    print("SEASON-BY-SEASON VALIDATION")
    print("=" * 70)
    
    for year in [2022, 2023, 2024]:
        trainer.validate_on_season(year, 'FinishPosition')
    
    # Feature importance analysis
    print("\n" + "=" * 70)
    importance_df = trainer.analyze_feature_importance('FinishPosition', top_n=20)
    
    # Save importance analysis
    importance_df.to_csv('predictions/feature_importance.csv', index=False)
    print(f"\n✅ Feature importance saved to predictions/feature_importance.csv")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    # Check if weather features are being used
    weather_features = [f for f in features if 'Temp' in f or 'Rain' in f or 'Humidity' in f]
    if weather_features:
        weather_importance = importance_df[importance_df['feature'].isin(weather_features)]['importance'].sum()
        print(f"🌤️  Weather features: {len(weather_features)} features, {weather_importance*100:.1f}% total importance")
    else:
        print("⚠️  Weather features not available in training data")
    
    # Check tire performance features
    tire_features = [f for f in features if 'TirePerf' in f]
    if tire_features:
        tire_importance = importance_df[importance_df['feature'].isin(tire_features)]['importance'].sum()
        print(f"🏎️  Tire performance: {len(tire_features)} features, {tire_importance*100:.1f}% total importance")
    else:
        print("⚠️  Tire performance features not available")
    
    # Model quality assessment
    top_10_importance = importance_df.head(10)['importance'].sum()
    print(f"\n📈 Top 10 features account for {top_10_importance*100:.1f}% of predictions")
    
    if top_10_importance > 0.95:
        print("✅ Model is well-optimized (top features dominate)")
        print("💡 Recommendation: Consider removing low-importance features for faster training")
    else:
        print("✅ Model uses diverse features (good generalization)")

if __name__ == "__main__":
    main()
