from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import time


@dataclass
class RequestResult:
    """Result for a single inference request."""
    backend: str
    model: str
    quant: str
    prompt_tokens: int
    output_tokens: int

    # Timing (seconds)
    ttft: float                     # Time to first token
    prefill_time: float             # Time to complete prefill phase
    decode_time: float              # Time for all decode steps after first token
    total_time: float               # Wall-clock end-to-end time

    # Throughput
    output_tps: float               # Output tokens/sec
    total_tps: float                # (prompt + output) tokens/sec

    # Memory (MB) — sampled at end of request
    memory_used_mb: float
    memory_total_mb: float

    # Config snapshot
    temperature: float
    max_tokens: int
    concurrency: int                # How many concurrent requests were running

    # Optional
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Result for a task benchmark sample (OCR, etc.)."""
    backend: str
    model: str
    quant: str
    benchmark_name: str
    sample_id: str

    prediction: str
    reference: str
    score: float                    # Benchmark-specific scalar (e.g. CER, accuracy)
    score_label: str                # e.g. "CER", "Accuracy"

    latency_s: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunSummary:
    """Aggregated summary across all RequestResults for a backend+model+quant combo."""
    backend: str
    model: str
    quant: str
    n_requests: int

    # Latency percentiles (seconds)
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    total_latency_p50: float
    total_latency_p95: float
    total_latency_p99: float

    # Throughput
    mean_output_tps: float
    mean_total_tps: float

    # Prefill / decode breakdown
    mean_prefill_time: float
    mean_decode_time: float

    # Memory
    peak_memory_mb: float

    # Error rate
    error_rate: float

    def to_dict(self) -> dict:
        return asdict(self)
