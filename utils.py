import hashlib


def geo_bucket(lat: float, lon: float, size_m: int = 500) -> str:
    if lat is None or lon is None:
        return "NA"
    cell = size_m / 111_000.0
    lat_b = round(lat / cell) * cell
    lon_b = round(lon / cell) * cell
    return f"{lat_b:.5f}_{lon_b:.5f}"

def stable_hash(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]
