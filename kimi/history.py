"""Helpers for comparing and trending historic runs."""

from typing import List, Tuple

from kimi.models import SSSResult
from kimi.storage import find_result_by_run_id, list_results, load_result


def load_all_results() -> List[SSSResult]:
    return [load_result(p) for p in list_results()]


def diff_results(a: SSSResult, b: SSSResult) -> List[Tuple[int, float, float, float]]:
    """Return list of (segment_index, sss_a, sss_b, delta)."""
    max_len = max(len(a.segments), len(b.segments))
    rows = []
    for i in range(max_len):
        sa = a.segments[i].sss if i < len(a.segments) else None
        sb = b.segments[i].sss if i < len(b.segments) else None
        if sa is not None and sb is not None:
            rows.append((i, sa, sb, sb - sa))
        elif sa is not None:
            rows.append((i, sa, 0.0, -sa))
        else:
            rows.append((i, 0.0, sb, sb))
    return rows


def get_result(run_id: str) -> SSSResult:
    path = find_result_by_run_id(run_id)
    return load_result(path)
