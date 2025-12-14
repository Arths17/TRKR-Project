"""
Pydantic schemas for API I/O.
"""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class RaceBase(BaseModel):
    year: int
    round: Optional[int] = None
    name: str
    circuit: Optional[str] = None
    event_date: Optional[datetime] = None


class RaceCreate(RaceBase):
    pass


class RaceOut(RaceBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class PredictionEntryOut(BaseModel):
    driver: str
    team: Optional[str]
    predicted_position: int
    predicted_race_time: float
    gap: float
    uncertainty: Optional[float]

    class Config:
        from_attributes = True


class PredictionOut(BaseModel):
    id: int
    race_id: int
    created_at: datetime
    freeze_policy: str
    snapshot_ts: datetime
    session_type: str
    fastf1_version: Optional[str]
    cache_path: Optional[str]
    confidence_level: Optional[str]
    confidence_score: Optional[float]
    feature_coverage: Optional[float]
    num_imputed: Optional[int]
    entries: List[PredictionEntryOut]

    class Config:
        from_attributes = True


class RaceResultOut(BaseModel):
    driver: str
    team: Optional[str]
    position: Optional[int]
    time: Optional[float]
    status: Optional[str]
    points: Optional[float]

    class Config:
        from_attributes = True


class EvaluationMetricOut(BaseModel):
    position_mae: Optional[float]
    time_mae_seconds: Optional[float]
    winner_correct: bool
    podium_accuracy: Optional[float]
    confidence_score: Optional[float]
    calibration_error: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True
