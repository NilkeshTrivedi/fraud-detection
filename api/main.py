"""
FastAPI application entry point.
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health, predict, stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload model and DB connection on startup."""
    logger.info("Starting Fraud Detection API...")
    from api.dependencies import get_predictor, get_db_engine
    get_predictor()
    get_db_engine()
    logger.info("API ready ✅")
    yield
    logger.info("Shutting down API...")


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud scoring API powered by XGBoost trained on IEEE-CIS dataset",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(stats.router)