"""Aggregate a list of RequestResults into a RunSummary."""
from __future__ import annotations
import numpy as np
from core.models import RequestResult, RunSummary


def summarize(results: list[RequestResult]) -> RunSummary:
    if not results:
        raise ValueError("No results to summarize.")

    r = results[0]
    errors = [x for x in results if x.error is not None]
    ok = [x for x in results if x.error is None]

    def p(arr, q):
        return float(np.percentile(arr, q)) if arr else 0.0

    ttfts = [x.ttft for x in ok]
    total_lats = [x.total_time for x in ok]
    output_tps_vals = [x.output_tps for x in ok]
    total_tps_vals = [x.total_tps for x in ok]
    prefills = [x.prefill_time for x in ok]
    decodes = [x.decode_time for x in ok]
    mems = [x.memory_used_mb for x in ok]

    return RunSummary(
        backend=r.backend,
        model=r.model,
        quant=r.quant,
        n_requests=len(results),
        ttft_p50=p(ttfts, 50),
        ttft_p95=p(ttfts, 95),
        ttft_p99=p(ttfts, 99),
        total_latency_p50=p(total_lats, 50),
        total_latency_p95=p(total_lats, 95),
        total_latency_p99=p(total_lats, 99),
        mean_output_tps=float(np.mean(output_tps_vals)) if output_tps_vals else 0.0,
        mean_total_tps=float(np.mean(total_tps_vals)) if total_tps_vals else 0.0,
        mean_prefill_time=float(np.mean(prefills)) if prefills else 0.0,
        mean_decode_time=float(np.mean(decodes)) if decodes else 0.0,
        peak_memory_mb=max(mems) if mems else 0.0,
        error_rate=len(errors) / len(results),
    )
