"""Seuil de confiance du routeur : texte sans ancrage → catégorie Autre."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

import municipal.router as rt



def test_smart_route_autre_when_max_cosine_below_threshold() -> None:
    rt._cached = None
    try:
        _run_low_sim_case()
    finally:
        rt._cached = None


def _run_low_sim_case() -> None:
    def fake_emb(texts: list[str], normalize: bool = True) -> np.ndarray:
        k = len(texts)
        if k == len(rt.ANCHORS):
            m = np.zeros((k, 384), dtype=np.float64)
            for i in range(k):
                m[i, i] = 1.0
            return m
        assert k == 1
        row = np.zeros((1, 384), dtype=np.float64)
        row[0, 200] = 1.0
        return row

    with patch.object(rt, "embed_texts", side_effect=fake_emb):
        out = rt.smart_route("qwrty plop zzzz")
    assert out["category"] == "Autre"
    assert out["municipal_service"] == "Secrétariat général"
