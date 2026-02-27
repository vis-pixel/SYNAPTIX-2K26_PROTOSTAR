"""
BioRhythm X — Research-Grade AI Backend
FastAPI Application Entry Point
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    _PROMETHEUS = True
except ImportError:
    _PROMETHEUS = False

from app.config import settings
from app.database import init_db
from app.logging_config import setup_logging
from app.routes import (
    auth,
    vitals,
    steps,
    calories,
    diet,
    predictions,
    anomaly,
    risk,
    datasets,
    websocket,
    water,
    whatsapp,
    field_alerts,
    devices,
    sse,
    live_data,
)

# ─── Logging ────────────────────────────────────────────────────────────────
setup_logging()
logger = logging.getLogger("biorhythm")


# ─── Lifespan ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    logger.info("BioRhythm X starting up...")
    await init_db()
    logger.info("Database initialized")
    logger.info(f"Dataset mode: {settings.DATASET_MODE}")
    wa_provider = getattr(settings, 'WHATSAPP_PROVIDER', 'not_configured')
    logger.info(f"WhatsApp provider: {wa_provider}")
    yield
    logger.info("BioRhythm X shutting down...")


# ─── Application ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="BioRhythm X API",
    description=(
        "Research-grade AI backend for wearable health + nutrition intelligence. "
        "Provides real-time biometric analysis, ML-powered predictions, adaptive diet planning, "
        "per-field ML training (1 dataset per field), WhatsApp alerts, water tracking, "
        "and multi-dataset support including PhysioNet, WESAD, MHEALTH, and Sleep-EDF."
    ),
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ─── Middleware ───────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=500)

# ─── Prometheus Metrics (optional) ───────────────────────────────────────────
if _PROMETHEUS:
    Instrumentator().instrument(app).expose(app)

# ─── Routes ──────────────────────────────────────────────────────────────────
API = "/api"
app.include_router(auth.router, prefix=f"{API}/auth", tags=["Authentication"])
app.include_router(vitals.router, prefix=f"{API}/vitals", tags=["Live Vitals"])
app.include_router(steps.router, prefix=f"{API}/steps", tags=["Step Engine"])
app.include_router(calories.router, prefix=f"{API}/calories", tags=["Calorie Engine"])
app.include_router(diet.router, prefix=f"{API}/diet", tags=["Diet Intelligence"])
app.include_router(predictions.router, prefix=f"{API}/predictions", tags=["Biometric Predictions"])
app.include_router(anomaly.router, prefix=f"{API}/anomaly", tags=["Anomaly Detection"])
app.include_router(risk.router, prefix=f"{API}/risk", tags=["AI Risk Engine"])
app.include_router(datasets.router, prefix=f"{API}/datasets", tags=["Dataset Management"])
app.include_router(websocket.router, tags=["WebSocket"])
# ── New features ──────────────────────────────────────────────────────────────
app.include_router(water.router, prefix=f"{API}/water", tags=["Water Tracker"])
app.include_router(whatsapp.router, prefix=f"{API}/whatsapp", tags=["WhatsApp"])
app.include_router(field_alerts.router, prefix=f"{API}/field-alerts", tags=["Per-Field Alerts"])
app.include_router(devices.router, prefix=f"{API}/devices", tags=["Device Management"])
app.include_router(sse.router, tags=["Real-time SSE"])
app.include_router(live_data.router, tags=["Live Risk Engine"])


# ─── Health Check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "dataset_mode": settings.DATASET_MODE,
    }


@app.get("/", tags=["System"])
async def root():
    return {
        "message": "BioRhythm X API",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }
