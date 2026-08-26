######
#   ALKAYA MEHMET
#   EPITECH 2025
#   PROJET EDP
#####


import json
import os
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from utils import geo_bucket, stable_hash
from model_inference import ENC_PATH, MODEL_PATH, Predictor, TFIDF_PATH
from reporting_routes import router as reporting_router

limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])


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

app = FastAPI(title="Municip'All IA API", version="0.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(reporting_router)

_API_KEY = os.environ.get("API_KEY", "").strip()


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if _API_KEY and request.url.path.startswith("/reporting"):
        if request.headers.get("X-API-Key") != _API_KEY:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=401, content={"detail": "X-API-Key manquant ou invalide"})
    return await call_next(request)

class PredictIn(BaseModel):
    description: str
    lat: float
    lon: float
    hour: int = 12

class PredictOut(BaseModel):
    pred: str
    proba: float
    cache: bool = False

@app.post("/predict", response_model=PredictOut)
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
    return {"status": "ok", "model_loaded": predictor is not None}
