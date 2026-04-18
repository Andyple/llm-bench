# llm-bench

A local LLM inference benchmark suite with a Streamlit dashboard. Measures throughput, prefill/decode latency, TTFT, memory, and pluggable task benchmarks (OCR, etc.) across vLLM, Ollama, and llama.cpp backends.

## Structure

```
llm-bench/
├── backends/           # Backend client wrappers (vLLM, Ollama, llama.cpp)
├── benchmarks/         # Pluggable task benchmarks (OCR, etc.)
├── core/               # Metrics collection, runner, result models
├── ui/                 # Streamlit app + components
├── config/             # YAML configs for models, backends, defaults
├── results/            # Output Parquet/JSON files
└── app.py              # Streamlit entrypoint
```

## Setup

```bash
uv sync
streamlit run app.py
```

## Backends

Each backend must be running before benchmarking:
- **vLLM**: `vllm serve <model> --port 8000`
- **Ollama**: `ollama serve` (default port 11434)
- **llama.cpp**: `./llama-server -m <model> --port 8080`

## Adding a Benchmark

Create a new file in `benchmarks/` implementing the `BaseBenchmark` interface:

```python
from benchmarks.base import BaseBenchmark

class MyBenchmark(BaseBenchmark):
    name = "my_benchmark"

    def load_dataset(self): ...
    def run_single(self, client, sample): ...
    def score(self, prediction, reference): ...
```

Then register it in `benchmarks/__init__.py`.
