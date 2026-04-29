"""Routage catégorie / service : exige le téléchargement du modèle d’embedding."""

from __future__ import annotations

import pytest

from municipal.router import smart_route


@pytest.mark.slow
def test_smart_route_voirie_nid_de_poule() -> None:
    out = smart_route("Nid de poule dangereux pour les cyclistes rue des Tilleuls")
    assert out["category"] == "Voirie"
    assert "techniques" in (out["municipal_service"] or "").lower()


@pytest.mark.slow
def test_smart_route_eclairage() -> None:
    out = smart_route("Lampadaire cassé, rue sombre la nuit")
    assert out["category"] == "Éclairage public"
