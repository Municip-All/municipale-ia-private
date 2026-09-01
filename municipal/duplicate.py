from __future__ import annotations

import json
from typing import Any

from municipal.config import DUPLICATE_SIMILARITY_THRESHOLD
from municipal.db import find_nearest_report_by_embedding


def duplicate_finder(
    embedding: list[float],
    exclude_report_id: int | None = None,
    threshold: float | None = None,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    t = float(threshold if threshold is not None else DUPLICATE_SIMILARITY_THRESHOLD)
    return find_nearest_report_by_embedding(embedding, exclude_report_id, t, tenant_id=tenant_id)


def duplicate_finder_json(
    embedding: list[float],
    exclude_report_id: int | None = None,
    threshold: float | None = None,
    tenant_id: str | None = None,
) -> str:
    return json.dumps(
        duplicate_finder(embedding, exclude_report_id, threshold, tenant_id=tenant_id),
        ensure_ascii=False,
    )
