from __future__ import annotations

from unittest.mock import patch

import numpy as np

import municipal.router as rt


def test_confidence_equals_raw_cosine_no_inflation():
    rt._cached = None
    try:
        _assert_raw_sim_confidence()
    finally:
        rt._cached = None


def _assert_raw_sim_confidence():
    raw_sim_value = 0.47

    def fake_emb(texts: list[str], normalize: bool = True) -> np.ndarray:
        k = len(texts)
        if k == len(rt.ANCHORS):
            m = np.zeros((k, 384), dtype=np.float64)
            for i in range(k):
                m[i, i] = 1.0
            return m
        assert k == 1
        anchor_idx = 2
        row = np.zeros((1, 384), dtype=np.float64)
        row[0, anchor_idx] = raw_sim_value
        return row

    with patch.object(rt, "embed_texts", side_effect=fake_emb):
        out = rt.smart_route("test routing query")

    assert out["confidence"] == raw_sim_value
    assert out["confidence"] != (raw_sim_value + 1) / 2
    assert 0.0 <= out["confidence"] <= 1.0
