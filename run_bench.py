#!/usr/bin/env python3
"""
CLI runner — headless benchmark execution for CI/CD pipelines.

Usage:
  python run_bench.py --backend vllm --model qwen2.5-7b-instruct --quant fp16
  python run_bench.py --backend ollama --model llama3.1:8b --quant q4_k_m --n 30
  python run_bench.py --backend vllm --model qwen2.5-7b-instruct --sweep
  python run_bench.py --compare results/run_abc_requests.parquet results/run_xyz_requests.parquet
"""
import argparse
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backends import BACKEND_REGISTRY
from backends.base import GenerateParams
from core.runner import run_single_request
from core.aggregator import summarize
from core.storage import save_request_results, save_summary
from core.sweep import run_concurrency_sweep


PROMPTS = {
    "short":  "Explain what a transformer model is in one sentence.",
    "medium": "Explain the difference between supervised and unsupervised learning with examples.",
    "long":   "Write a detailed technical explanation of how attention mechanisms work in transformer models.",
}


def parse_args():
    p = argparse.ArgumentParser(description="llm-bench CLI")
    p.add_argument("--backend",  required=True, choices=list(BACKEND_REGISTRY.keys()))
    p.add_argument("--model",    required=True)
    p.add_argument("--quant",    default="unknown")
    p.add_argument("--host",     default="localhost")
    p.add_argument("--port",     type=int, default=None,
                   help="Defaults: vllm=8000, ollama=11434, llama.cpp=8080")
    p.add_argument("--n",        type=int, default=20, help="Number of requests")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--prompt",   default="medium", choices=list(PROMPTS.keys()) + ["custom"])
    p.add_argument("--prompt-text", default=None, help="Custom prompt string")
    p.add_argument("--sweep",    action="store_true", help="Run concurrency sweep")
    p.add_argument("--sweep-levels", default="1,4,16", help="Comma-separated concurrency levels")
    p.add_argument("--sweep-requests", type=int, default=10)
    p.add_argument("--output-dir", default="results")
    p.add_argument("--baseline", default=None,
                   help="Path to baseline summary JSON for regression check")
    p.add_argument("--tps-threshold", type=float, default=0.9,
                   help="Fail if mean TPS drops below this fraction of baseline (default 0.90)")
    p.add_argument("--latency-threshold", type=float, default=1.2,
                   help="Fail if p95 latency exceeds this multiple of baseline (default 1.20)")
    return p.parse_args()


DEFAULT_PORTS = {"vllm": 8000, "ollama": 11434, "llama.cpp": 8080}


def main():
    args = parse_args()
    port = args.port or DEFAULT_PORTS.get(args.backend, 8000)

    BackendCls = BACKEND_REGISTRY[args.backend]
    backend = BackendCls(host=args.host, port=port, model=args.model)

    print(f"[llm-bench] Health check {args.backend}@{args.host}:{port} ...", end=" ")
    if not backend.health_check():
        print("FAILED")
        sys.exit(1)
    print("OK")

    prompt_text = args.prompt_text if args.prompt_text else PROMPTS[args.prompt]
    params = GenerateParams(
        prompt=prompt_text,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.sweep:
        levels = [int(x.strip()) for x in args.sweep_levels.split(",")]
        print(f"[llm-bench] Concurrency sweep: levels={levels}, {args.sweep_requests} req/level")

        def _progress(done, total):
            print(f"  level {done}/{total} done")

        points = run_concurrency_sweep(
            backend=backend,
            params=params,
            quant=args.quant,
            concurrency_levels=levels,
            requests_per_level=args.sweep_requests,
            progress_callback=_progress,
        )

        sweep_output = []
        for pt in points:
            s = pt.summary
            row = {
                "concurrency": pt.concurrency,
                "mean_output_tps": round(s.mean_output_tps, 2),
                "ttft_p50_ms": round(s.ttft_p50 * 1000, 1),
                "ttft_p95_ms": round(s.ttft_p95 * 1000, 1),
                "total_latency_p95_ms": round(s.total_latency_p95 * 1000, 1),
                "peak_memory_gb": round(s.peak_memory_mb / 1024, 2),
                "error_rate": round(s.error_rate, 3),
            }
            sweep_output.append(row)
            print(f"  c={pt.concurrency:3d}  tps={s.mean_output_tps:7.1f}  "
                  f"ttft_p50={s.ttft_p50*1000:.0f}ms  lat_p95={s.total_latency_p95*1000:.0f}ms")

        sweep_path = output_dir / f"{run_id}_sweep.json"
        sweep_path.write_text(json.dumps(sweep_output, indent=2))
        print(f"[llm-bench] Sweep saved: {sweep_path}")

    else:
        print(f"[llm-bench] Running {args.n} requests (quant={args.quant}) ...")
        results = []
        for i in range(args.n):
            r = run_single_request(backend, params, quant=args.quant)
            results.append(r)
            if (i + 1) % 5 == 0 or (i + 1) == args.n:
                print(f"  {i+1}/{args.n}  tps={r.output_tps:.1f}  ttft={r.ttft*1000:.0f}ms", end="")
                if r.error:
                    print(f"  ERROR: {r.error}", end="")
                print()

        summary = summarize(results)
        req_path = save_request_results(results, run_id)
        sum_path = save_summary(summary, run_id)

        print(f"\n[llm-bench] Results:")
        print(f"  mean output TPS : {summary.mean_output_tps:.2f}")
        print(f"  TTFT   p50/p95  : {summary.ttft_p50*1000:.0f}ms / {summary.ttft_p95*1000:.0f}ms")
        print(f"  Latency p50/p95 : {summary.total_latency_p50*1000:.0f}ms / {summary.total_latency_p95*1000:.0f}ms")
        print(f"  Prefill mean    : {summary.mean_prefill_time*1000:.0f}ms")
        print(f"  Decode mean     : {summary.mean_decode_time*1000:.0f}ms")
        print(f"  Peak memory     : {summary.peak_memory_mb/1024:.2f}GB")
        print(f"  Error rate      : {summary.error_rate:.1%}")
        print(f"  Saved: {req_path.name}, {sum_path.name}")

        # ── Regression check against baseline ──────────────────────────────
        if args.baseline:
            baseline_path = Path(args.baseline)
            if not baseline_path.exists():
                print(f"[llm-bench] WARNING: baseline not found at {baseline_path}")
                sys.exit(0)

            baseline = json.loads(baseline_path.read_text())
            passed = True

            baseline_tps = baseline.get("mean_output_tps", 0)
            if baseline_tps > 0:
                tps_ratio = summary.mean_output_tps / baseline_tps
                tps_ok = tps_ratio >= args.tps_threshold
                print(f"\n[regression] TPS: {summary.mean_output_tps:.1f} vs baseline {baseline_tps:.1f} "
                      f"(ratio={tps_ratio:.2f}, threshold≥{args.tps_threshold}) → {'PASS ✓' if tps_ok else 'FAIL ✗'}")
                if not tps_ok:
                    passed = False

            baseline_p95 = baseline.get("total_latency_p95", 0)
            if baseline_p95 > 0:
                lat_ratio = summary.total_latency_p95 / baseline_p95
                lat_ok = lat_ratio <= args.latency_threshold
                print(f"[regression] p95 latency: {summary.total_latency_p95:.3f}s vs baseline {baseline_p95:.3f}s "
                      f"(ratio={lat_ratio:.2f}, threshold≤{args.latency_threshold}) → {'PASS ✓' if lat_ok else 'FAIL ✗'}")
                if not lat_ok:
                    passed = False

            if not passed:
                print("[regression] ✗ REGRESSION DETECTED — exiting with code 1")
                sys.exit(1)
            else:
                print("[regression] ✓ All checks passed")


if __name__ == "__main__":
    main()
