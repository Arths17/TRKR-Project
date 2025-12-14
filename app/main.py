"""
FastAPI entrypoint for production deployment.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app import settings
from app.database import Base, engine
from app.api import router

# Create tables if not exist (simple bootstrap; swap to migrations in real deploy)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="F1 Prediction Tracking API", version=settings.APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok", "env": settings.APP_ENV}
