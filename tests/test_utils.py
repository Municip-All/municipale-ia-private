from utils import geo_bucket, stable_hash


def test_geo_bucket_returns_string():
    assert isinstance(geo_bucket(49.0, 2.0), str)


def test_geo_bucket_none_returns_na():
    assert geo_bucket(None, 2.0) == "NA"
    assert geo_bucket(49.0, None) == "NA"


def test_geo_bucket_deterministic():
    assert geo_bucket(49.265, 2.435) == geo_bucket(49.265, 2.435)


def test_stable_hash_deterministic():
    assert stable_hash("test") == stable_hash("test")


def test_stable_hash_empty():
    assert isinstance(stable_hash(""), str)
    assert len(stable_hash("")) == 16


def test_stable_hash_none_safe():
    assert isinstance(stable_hash(None), str)
