"""
Core runner: executes a single inference request against a backend,
collecting TTFT, prefill time, decode time, TPS, and memory.
"""
from __future__ import annotations
import time
from typing import Optional
from backends.base import BaseBackend, GenerateParams
from core.models import RequestResult
from core.memory import sample_memory


def run_single_request(
    backend: BaseBackend,
    params: GenerateParams,
    quant: str = "unknown",
    concurrency: int = 1,
) -> RequestResult:
    """
    Run one request and return a fully populated RequestResult.

    Timing model:
    - t0: request sent
    - t_first: first token received → TTFT = t_first - t0
    - t_end: last token received
    - prefill_time ≈ TTFT (the time before any decode output)
    - decode_time = t_end - t_first
    - total_time = t_end - t0
    """
    error: Optional[str] = None
    tokens: list[str] = []
    t0 = t_first = t_end = 0.0

    try:
        t0 = time.perf_counter()
        for i, token in enumerate(backend.generate_stream(params)):
            now = time.perf_counter()
            if i == 0:
                t_first = now
            tokens.append(token)
        t_end = time.perf_counter()
    except Exception as e:
        error = str(e)
        t_end = time.perf_counter()
        if t_first == 0.0:
            t_first = t_end

    mem_used, mem_total = sample_memory()

    ttft = t_first - t0
    prefill_time = ttft
    decode_time = max(t_end - t_first, 0.0)
    total_time = t_end - t0

    # Approximate token counts
    output_tokens = len(tokens)
    # Rough prompt token estimate: words * 1.3
    prompt_tokens = max(1, int(len(params.prompt.split()) * 1.3))

    output_tps = output_tokens / decode_time if decode_time > 0 else 0.0
    total_tps = (prompt_tokens + output_tokens) / total_time if total_time > 0 else 0.0

    return RequestResult(
        backend=backend.name,
        model=backend.model,
        quant=quant,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft=ttft,
        prefill_time=prefill_time,
        decode_time=decode_time,
        total_time=total_time,
        output_tps=output_tps,
        total_tps=total_tps,
        memory_used_mb=mem_used,
        memory_total_mb=mem_total,
        temperature=params.temperature,
        max_tokens=params.max_tokens,
        concurrency=concurrency,
        error=error,
    )
