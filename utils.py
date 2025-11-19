######
#   ALKAYA MEHMET
#   EPITECH 2025
#   PROJET EDP
#####

import hashlib
import json
from pathlib import Path

def geo_bucket(lat: float, lon: float, size_m: int = 500) -> str:
    """
    Very simple spatial bucketing by rounding lat/lon to ~grid cell.
    1 degree lat ≈ 111_000 m. We approximate for demo purposes.
    """
    if lat is None or lon is None:
        return "NA"
    cell = size_m / 111_000.0
    lat_b = round(lat / cell) * cell
    lon_b = round(lon / cell) * cell
    return f"{lat_b:.5f}_{lon_b:.5f}"

def stable_hash(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]

def save_json(path: str, data: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
