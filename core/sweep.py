"""
Concurrency sweep: runs the same prompt at multiple concurrency levels
to build a throughput-vs-latency curve. This is the core of Phase 1's
"stress test" view — shows where each backend saturates.
"""
from __future__ import annotations
from dataclasses import dataclass
from backends.base import BaseBackend, GenerateParams
from core.concurrent_runner import run_concurrent
from core.aggregator import summarize
from core.models import RequestResult, RunSummary


@dataclass
class SweepPoint:
    concurrency: int
    summary: RunSummary
    raw: list[RequestResult]


def run_concurrency_sweep(
    backend: BaseBackend,
    params: GenerateParams,
    quant: str,
    concurrency_levels: list[int],
    requests_per_level: int = 20,
    progress_callback=None,
) -> list[SweepPoint]:
    """
    For each concurrency level, fire `requests_per_level` requests and aggregate.
    Returns a list of SweepPoints — one per concurrency level.

    progress_callback(level_idx, total_levels) is called after each level completes.
    """
    points: list[SweepPoint] = []

    for i, c in enumerate(concurrency_levels):
        results = run_concurrent(
            backend=backend,
            params=params,
            quant=quant,
            concurrency=c,
            total_requests=requests_per_level,
        )
        summary = summarize(results)
        points.append(SweepPoint(concurrency=c, summary=summary, raw=results))

        if progress_callback:
            progress_callback(i + 1, len(concurrency_levels))

    return points
