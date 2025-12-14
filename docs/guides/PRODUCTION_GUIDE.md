# F1 Prediction Tracking – Production Guide

## Folder Structure (added)
- `app/`
  - `settings.py` – env-driven config (DB URL, cache dir, freeze policy)
  - `database.py` – SQLAlchemy engine/base/session factory
  - `models.py` – ORM tables: races, predictions, prediction_entries, race_results, evaluation_metrics
  - `schemas.py` – Pydantic response models
  - `services.py` – lifecycle orchestration (freeze predictions, store results, evaluate)
  - `api.py` – FastAPI routes (predict, fetch predictions/results/metrics, sync results)
  - `main.py` – FastAPI app entrypoint + CORS + table bootstrap
- `dashboard/app.py` – Streamlit dashboard (race selector, predictions, results, metrics)
- `Dockerfile` – uvicorn entrypoint, FastF1 cache env
- `.env.example` – environment defaults for DB/cache/freeze policy

## Database Schema (SQLite/Postgres)
- `races`: id, year, round, name, circuit, event_date
- `predictions`: id, race_id, created_at, freeze_policy, snapshot_ts, session_type, fastf1_version, cache_path, confidence_level/score, feature_coverage, num_imputed, status
- `prediction_entries`: id, prediction_id, driver, team, predicted_position, predicted_race_time, gap, uncertainty
- `race_results`: id, race_id, driver, team, position, time, status, points
- `evaluation_metrics`: id, race_id, prediction_id, position_mae, time_mae_seconds, winner_correct, podium_accuracy, confidence_score, calibration_error

## Prediction Lifecycle
1) **Pre-race (freeze):**
   - Endpoint: `POST /predict?year=YYYY&race=RoundOrName`
   - Steps: fetch FastF1 with cache → process features (pre-race only) → predict once → validate (no inversions, no negative gaps, no zero laps, coverage/gap warnings) → store immutable snapshot + FastF1 metadata
2) **Post-race:**
   - Endpoint: `POST /results/sync/{year}/{race}`
   - Steps: fetch official results via FastF1 → persist `race_results` → evaluate latest frozen prediction → store metrics
3) **Read:**
   - `GET /predictions/{race_id}`
   - `GET /results/{race_id}`
   - `GET /metrics/{race_id}`

## FastF1 Integration Rules
- Persistent cache enabled via `FASTF1_CACHE_DIR`
- Snapshot metadata stored on prediction: `fastf1_version`, `cache_path`, `snapshot_ts`, `session_type`
- Predictions created once per race (`status=frozen`), never overwritten; subsequent calls return latest frozen record
- Freeze policy envs: `PREDICTION_FREEZE_POLICY` (`post_qualifying` or `t-minus`), `PREDICTION_FREEZE_TMINUTES` (lead time)

## Metrics & Tracking
- `EvaluationMetric`: position MAE, time MAE (seconds), winner correctness, podium accuracy, confidence score, calibration placeholder
- Metrics stored per race_id, linked to the frozen prediction

## API Layer (FastAPI)
- `POST /predict` → generate & store prediction
- `GET /predictions/{race_id}` → frozen snapshot
- `GET /results/{race_id}` → official results
- `GET /metrics/{race_id}` → evaluation summary
- `POST /results/sync/{year}/{race}` → fetch results + evaluate

## Dashboard (Streamlit)
- Run: `streamlit run dashboard/app.py`
- Features: race selector, frozen predictions table (with gaps/uncertainty), results table, metrics table, confidence/coverage caption

## Deployment
- Build: `docker build -t f1-tracker .`
- Run: `docker run -p 8000:8000 --env-file .env f1-tracker`
- Volume FastF1 cache: `-v $(pwd)/cache:/app/cache`
- Env overrides: `DATABASE_URL`, `FASTF1_CACHE_DIR`, `PREDICTION_FREEZE_POLICY`, `PREDICTION_FREEZE_TMINUTES`, `ALLOWED_ORIGINS`

## Operational Notes
- Keep model logic unchanged; only lifecycle/persistence added
- Predictions validated for ranking integrity, non-negative/monotonic gaps, coverage warnings (<85%), extreme gaps (>120s) surfaced by existing predictor
- Post-race evaluations should be re-run after results sync to populate metrics
