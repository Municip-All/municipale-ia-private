######
#   ALKAYA MEHMET
#   EPITECH 2025
#   PROJET EDP
#####

"""
FastAPI serving IA predictions with Swagger.
Endpoints:
- POST /predict
- GET /health
- (stub) GET /stats
"""
from fastapi import FastAPI
from pydantic import BaseModel
from utils import geo_bucket, stable_hash
from model_inference import Predictor

try:
    import redis
    REDIS_AVAIL = True
except Exception:
    REDIS_AVAIL = False

app = FastAPI(title="Municip'All IA API", version="0.1.0")
predictor = None
rds = None

class PredictIn(BaseModel):
    description: str
    lat: float
    lon: float
    hour: int = 12

class PredictOut(BaseModel):
    pred: str
    proba: float
    cache: bool = False

@app.on_event("startup")
def _startup():
    global predictor, rds
    predictor = Predictor()
    if REDIS_AVAIL:
        try:
            rds = redis.Redis(host="localhost", port=6379, decode_responses=True)
            rds.ping()
        except Exception:
            rds = None

@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    g = geo_bucket(payload.lat, payload.lon, 500)
    key = stable_hash(f"{payload.description}|{g}|{payload.hour}")
    if rds is not None:
        cached = rds.get(key)
        if cached:
            import json
            c = json.loads(cached)
            return PredictOut(pred=c["pred"], proba=c["proba"], cache=True)

    out = predictor.predict(payload.description, g, payload.hour)

    if rds is not None:
        import json
        rds.setex(key, 3600, json.dumps(out))

    return PredictOut(**out, cache=False)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}

@app.get("/stats")
def stats():
    # stub for demo
    return {"info": "aggregate stats would be computed here"}

