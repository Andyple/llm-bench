"""
Concurrent request runner.
Fires `concurrency` requests simultaneously against a backend,
returns all RequestResults with accurate wall-clock timing.
"""
from __future__ import annotations
import concurrent.futures
from backends.base import BaseBackend, GenerateParams
from core.models import RequestResult
from core.runner import run_single_request


def run_concurrent(
    backend: BaseBackend,
    params: GenerateParams,
    quant: str,
    concurrency: int,
    total_requests: int,
) -> list[RequestResult]:
    """
    Run `total_requests` total requests with `concurrency` in-flight at once.
    Returns results in completion order.
    """
    results: list[RequestResult] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(run_single_request, backend, params, quant, concurrency)
            for _ in range(total_requests)
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                # Create a minimal error result
                from core.models import RequestResult
                import time
                results.append(RequestResult(
                    backend=backend.name,
                    model=backend.model,
                    quant=quant,
                    prompt_tokens=0,
                    output_tokens=0,
                    ttft=0.0,
                    prefill_time=0.0,
                    decode_time=0.0,
                    total_time=0.0,
                    output_tps=0.0,
                    total_tps=0.0,
                    memory_used_mb=0.0,
                    memory_total_mb=0.0,
                    temperature=params.temperature,
                    max_tokens=params.max_tokens,
                    concurrency=concurrency,
                    error=str(e),
                    timestamp=time.time(),
                ))

    return results
