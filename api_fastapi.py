import hmac
import json
import os
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from municipal.rate_limit import limiter

from utils import geo_bucket, stable_hash
from model_inference import ENC_PATH, MODEL_PATH, Predictor, TFIDF_PATH
from reporting_routes import router as reporting_router

log = structlog.get_logger("municipall.api")


def _ml_artifacts_ready() -> bool:
    return TFIDF_PATH.is_file() and MODEL_PATH.is_file() and ENC_PATH.is_file()

try:
    import redis
    REDIS_AVAIL = True
except Exception:
    REDIS_AVAIL = False

predictor = None
rds = None

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    global predictor, rds
    predictor = Predictor() if _ml_artifacts_ready() else None
    if REDIS_AVAIL:
        try:
            rds = redis.Redis(
                host=os.environ.get("REDIS_HOST", "localhost"),
                port=int(os.environ.get("REDIS_PORT", "6379")),
                decode_responses=True,
            )
            rds.ping()
        except Exception:
            rds = None
    yield
    if rds:
        rds.close()

app = FastAPI(title="Municip'All IA API", version="0.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

cors_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3002,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(reporting_router)

_API_KEY = os.environ.get("API_KEY", "").strip()
_IS_PROD = os.environ.get("NODE_ENV", "").strip() == "production"

if not _API_KEY:
    if _IS_PROD:
        log.error("api_key_missing_prod")
    else:
        log.warning("api_key_missing_dev")


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    path = request.url.path
    is_protected = (path.startswith("/reporting") or path.startswith("/predict")) and not path.startswith("/health")
    if is_protected:
        if _API_KEY:
            if not hmac.compare_digest(request.headers.get("X-API-Key", ""), _API_KEY):
                from fastapi.responses import JSONResponse
                return JSONResponse(status_code=401, content={"detail": "X-API-Key manquant ou invalide"})
        elif _IS_PROD:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=401, content={"detail": "API_KEY not configured"})
    return await call_next(request)

class PredictIn(BaseModel):
    description: str = Field(..., max_length=5000)
    lat: float
    lon: float
    hour: int = 12

class PredictOut(BaseModel):
    pred: str
    proba: float
    cache: bool = False

@app.post("/predict", response_model=PredictOut)
@limiter.limit("30/minute")
def predict(request: Request, payload: PredictIn):
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Modèle Random Forest indisponible : lancez le pipeline ML "
                "(python main.py) pour générer artifacts/*.joblib."
            ),
        )
    g = geo_bucket(payload.lat, payload.lon)
    key = stable_hash(f"{payload.description}|{g}|{payload.hour}")
    if rds is not None:
        cached = rds.get(key)
        if cached:
            c = json.loads(cached)
            return PredictOut(pred=c["pred"], proba=c["proba"], cache=True)

    out = predictor.predict(payload.description, g, payload.hour)

    if rds is not None:
        rds.setex(key, 3600, json.dumps(out))

    return PredictOut(**out, cache=False)

@app.get("/health")
def health(request: Request):
    checks: dict[str, Any] = {
        "status": "ok",
        "model_loaded": predictor is not None,
        "redis": "unknown",
        "database": "unknown",
    }
    if rds is not None:
        try:
            rds.ping()
            checks["redis"] = "ok"
        except Exception:
            checks["redis"] = "error"
            checks["status"] = "degraded"
    else:
        checks["redis"] = "not_configured"
    try:
        from municipal.db import get_conninfo
        get_conninfo()
        checks["database"] = "ok"
    except Exception:
        checks["database"] = "error"
        checks["status"] = "degraded"
    return checks
