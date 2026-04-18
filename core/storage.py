"""Persist and load benchmark results as Parquet files."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from core.models import RequestResult, BenchmarkResult, RunSummary

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def save_request_results(results: list[RequestResult], run_id: str) -> Path:
    path = RESULTS_DIR / f"{run_id}_requests.parquet"
    df = pd.DataFrame([r.to_dict() for r in results])
    df.to_parquet(path, index=False)
    return path


def save_benchmark_results(results: list[BenchmarkResult], run_id: str) -> Path:
    path = RESULTS_DIR / f"{run_id}_benchmark.parquet"
    df = pd.DataFrame([r.to_dict() for r in results])
    df.to_parquet(path, index=False)
    return path


def save_summary(summary: RunSummary, run_id: str) -> Path:
    path = RESULTS_DIR / f"{run_id}_summary.json"
    path.write_text(json.dumps(summary.to_dict(), indent=2))
    return path


def load_all_request_results() -> pd.DataFrame:
    files = sorted(RESULTS_DIR.glob("*_requests.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_all_benchmark_results() -> pd.DataFrame:
    files = sorted(RESULTS_DIR.glob("*_benchmark.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def list_run_ids() -> list[str]:
    return sorted({
        f.stem.replace("_requests", "").replace("_benchmark", "").replace("_summary", "")
        for f in RESULTS_DIR.glob("*.parquet")
    })
