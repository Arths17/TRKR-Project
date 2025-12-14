"""
FastAPI router definitions.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_session
from app import schemas, models
from app.services import PredictionService

router = APIRouter()


@router.post("/predict", response_model=schemas.PredictionOut)
def predict_race(year: int, race: str, db: Session = Depends(get_session)):
    service = PredictionService(db)
    prediction = service.generate_prediction(year, race)
    return prediction


@router.get("/predictions/{race_id}", response_model=schemas.PredictionOut)
def get_prediction(race_id: int, db: Session = Depends(get_session)):
    prediction = (
        db.query(models.Prediction)
        .filter_by(race_id=race_id)
        .order_by(models.Prediction.created_at.desc())
        .first()
    )
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return prediction


@router.get("/results/{race_id}", response_model=list[schemas.RaceResultOut])
def get_results(race_id: int, db: Session = Depends(get_session)):
    results = db.query(models.RaceResult).filter_by(race_id=race_id).all()
    return results


@router.get("/metrics/{race_id}", response_model=list[schemas.EvaluationMetricOut])
def get_metrics(race_id: int, db: Session = Depends(get_session)):
    metrics = db.query(models.EvaluationMetric).filter_by(race_id=race_id).all()
    return metrics


@router.post("/results/sync/{year}/{race}")
def sync_results(year: int, race: str, db: Session = Depends(get_session)):
    service = PredictionService(db)
    race_obj = service.store_race_results(year, race)
    metric = service.evaluate(race_obj.id)
    return {"race_id": race_obj.id, "evaluation_id": metric.id if metric else None}
