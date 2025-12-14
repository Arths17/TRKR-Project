"""
ORM models for production tracking (SQLite/Postgres compatible).
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean, UniqueConstraint
from sqlalchemy.orm import relationship
from app.database import Base


class Race(Base):
    __tablename__ = "races"

    id = Column(Integer, primary_key=True, index=True)
    year = Column(Integer, nullable=False)
    round = Column(Integer, nullable=True)
    name = Column(String, nullable=False)
    circuit = Column(String, nullable=True)
    event_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    predictions = relationship("Prediction", back_populates="race")
    results = relationship("RaceResult", back_populates="race")

    __table_args__ = (
        UniqueConstraint("year", "round", "name", name="uq_race_identity"),
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    race_id = Column(Integer, ForeignKey("races.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    freeze_policy = Column(String, default="post_qualifying")
    snapshot_ts = Column(DateTime, default=datetime.utcnow)
    session_type = Column(String, default="Q")
    fastf1_version = Column(String, nullable=True)
    cache_path = Column(String, nullable=True)
    confidence_level = Column(String, nullable=True)
    confidence_score = Column(Float, nullable=True)
    feature_coverage = Column(Float, nullable=True)
    num_imputed = Column(Integer, nullable=True)
    status = Column(String, default="frozen")  # frozen | superseded

    race = relationship("Race", back_populates="predictions")
    entries = relationship("PredictionEntry", back_populates="prediction", cascade="all, delete-orphan")
    metrics = relationship("EvaluationMetric", back_populates="prediction")


class PredictionEntry(Base):
    __tablename__ = "prediction_entries"

    id = Column(Integer, primary_key=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)
    driver = Column(String, nullable=False)
    team = Column(String, nullable=True)
    predicted_position = Column(Integer, nullable=False)
    predicted_race_time = Column(Float, nullable=False)
    gap = Column(Float, nullable=False)
    uncertainty = Column(Float, nullable=True)

    prediction = relationship("Prediction", back_populates="entries")


class RaceResult(Base):
    __tablename__ = "race_results"

    id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey("races.id"), nullable=False)
    driver = Column(String, nullable=False)
    team = Column(String, nullable=True)
    position = Column(Integer, nullable=True)
    time = Column(Float, nullable=True)
    status = Column(String, nullable=True)
    points = Column(Float, nullable=True)

    race = relationship("Race", back_populates="results")


class EvaluationMetric(Base):
    __tablename__ = "evaluation_metrics"

    id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey("races.id"), nullable=False)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    position_mae = Column(Float, nullable=True)
    time_mae_seconds = Column(Float, nullable=True)
    winner_correct = Column(Boolean, default=False)
    podium_accuracy = Column(Float, nullable=True)  # 0-1
    confidence_score = Column(Float, nullable=True)
    calibration_error = Column(Float, nullable=True)

    race = relationship("Race")
    prediction = relationship("Prediction", back_populates="metrics")
