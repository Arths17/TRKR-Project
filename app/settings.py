"""
Runtime settings for production services (FastAPI, jobs).
Reads from environment with safe defaults for local dev.
"""
import os
from pathlib import Path

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./f1prod.db")
DATABASE_ECHO = os.getenv("DATABASE_ECHO", "false").lower() == "true"

# FastF1 cache
FASTF1_CACHE_DIR = Path(os.getenv("FASTF1_CACHE_DIR", "cache"))
FASTF1_CACHE_DIR.mkdir(exist_ok=True)

# Prediction freeze logic
PREDICTION_FREEZE_POLICY = os.getenv("PREDICTION_FREEZE_POLICY", "post_qualifying")
# Optional minutes before race start cutoff (used when policy=="t-minus")
PREDICTION_FREEZE_TMINUTES = int(os.getenv("PREDICTION_FREEZE_TMINUTES", "30"))

# Deployment
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Security / misc
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Metadata
APP_VERSION = os.getenv("APP_VERSION", "2.0.0")
APP_ENV = os.getenv("APP_ENV", "local")
