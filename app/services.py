"""
Service layer: prediction lifecycle, persistence, evaluation.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import fastf1
from sqlalchemy.orm import Session

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import settings
from app import models, schemas
from data_fetcher import F1DataFetcher
from data_processor import F1DataProcessor
from predictor import F1Predictor


class PredictionService:
    def __init__(self, db: Session):
        self.db = db
        # Ensure FastF1 cache is enabled and persisted
        fastf1.Cache.enable_cache(str(settings.FASTF1_CACHE_DIR))

    def _get_or_create_race(self, year: int, race_identifier) -> models.Race:
        race = (
            self.db.query(models.Race)
            .filter(models.Race.year == year, models.Race.name == str(race_identifier))
            .first()
        )
        if race:
            return race
        race = models.Race(year=year, round=None, name=str(race_identifier))
        self.db.add(race)
        self.db.commit()
        self.db.refresh(race)
        return race

    def _prediction_exists(self, race_id: int) -> bool:
        return self.db.query(models.Prediction).filter_by(race_id=race_id).count() > 0

    def generate_prediction(self, year: int, race_identifier, freeze_policy: str = None) -> models.Prediction:
        freeze_policy = freeze_policy or settings.PREDICTION_FREEZE_POLICY
        race = self._get_or_create_race(year, race_identifier)
        if self._prediction_exists(race.id):
            return (
                self.db.query(models.Prediction)
                .filter_by(race_id=race.id)
                .order_by(models.Prediction.created_at.desc())
                .first()
            )

        fetcher = F1DataFetcher(seasons=[year])
        race_data_dict = fetcher.fetch_race_with_qualifying(year, race_identifier)
        processor = F1DataProcessor(race_data_dict)
        processed = processor.process()
        predictor = F1Predictor.load_predictor()
        predictions_df = predictor.predict_comprehensive(processed)
        insights = predictor.get_insights(processed, predictions_df)

        confidence = insights.get("confidence", {})

        prediction = models.Prediction(
            race_id=race.id,
            freeze_policy=freeze_policy,
            snapshot_ts=datetime.utcnow(),
            session_type="Q",
            fastf1_version=getattr(fastf1, "__version__", None),
            cache_path=str(settings.FASTF1_CACHE_DIR),
            confidence_level=confidence.get("level"),
            confidence_score=confidence.get("score"),
            feature_coverage=confidence.get("feature_coverage"),
            num_imputed=confidence.get("num_imputed"),
            status="frozen",
        )
        self.db.add(prediction)
        self.db.flush()

        for _, row in predictions_df.iterrows():
            entry = models.PredictionEntry(
                prediction_id=prediction.id,
                driver=row["Driver"],
                team=row.get("Team"),
                predicted_position=int(row["PredictedPosition"]),
                predicted_race_time=float(row["PredictedRaceTime"]),
                gap=float(row["Gap"]),
                uncertainty=float(row.get("Uncertainty", 0)),
            )
            self.db.add(entry)

        self.db.commit()
        self.db.refresh(prediction)
        return prediction

    def store_race_results(self, year: int, race_identifier) -> models.Race:
        race = self._get_or_create_race(year, race_identifier)
        # If results already stored, skip
        if self.db.query(models.RaceResult).filter_by(race_id=race.id).count() > 0:
            return race

        fetcher = F1DataFetcher(seasons=[year])
        race_data_dict = fetcher.fetch_all_race_data(year, race_identifier)
        results_df = race_data_dict.get("results")
        if results_df is None or results_df.empty:
            return race

        for _, row in results_df.iterrows():
            # Convert Timedelta to total seconds if needed
            race_time = row.get("Time")
            if race_time is not None:
                try:
                    race_time = race_time.total_seconds()
                except (AttributeError, TypeError):
                    race_time = float(race_time) if race_time else None
            
            result = models.RaceResult(
                race_id=race.id,
                driver=row.get("Driver"),
                team=row.get("Team"),
                position=int(row.get("Position")) if row.get("Position") is not None else None,
                time=race_time,
                status=row.get("Status"),
                points=float(row.get("Points", 0)) if row.get("Points") is not None else 0,
            )
            self.db.add(result)

        self.db.commit()
        return race

    def evaluate(self, race_id: int) -> Optional[models.EvaluationMetric]:
        prediction = (
            self.db.query(models.Prediction)
            .filter_by(race_id=race_id)
            .order_by(models.Prediction.created_at.desc())
            .first()
        )
        results = self.db.query(models.RaceResult).filter_by(race_id=race_id).all()
        if not prediction or not results:
            return None

        pred_entries = {e.driver: e for e in prediction.entries}
        pos_errors = []
        time_errors = []
        podium_hits = 0
        winner_correct = False

        for res in results:
            pred = pred_entries.get(res.driver)
            if not pred or res.position is None:
                continue
            pos_errors.append(abs(pred.predicted_position - res.position))
            if res.position == 1 and pred.predicted_position == 1:
                winner_correct = True
            if res.position <= 3 and pred.predicted_position <= 3:
                podium_hits += 1
            if res.time is not None and pred.predicted_race_time is not None:
                time_errors.append(abs(pred.predicted_race_time - res.time))

        position_mae = sum(pos_errors) / len(pos_errors) if pos_errors else None
        time_mae = sum(time_errors) / len(time_errors) if time_errors else None
        podium_accuracy = podium_hits / 3.0 if results else None

        metric = models.EvaluationMetric(
            race_id=race_id,
            prediction_id=prediction.id,
            position_mae=position_mae,
            time_mae_seconds=time_mae,
            winner_correct=winner_correct,
            podium_accuracy=podium_accuracy,
            confidence_score=prediction.confidence_score,
            calibration_error=None,
        )
        self.db.add(metric)
        self.db.commit()
        self.db.refresh(metric)
        return metric
