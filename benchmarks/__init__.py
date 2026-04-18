from benchmarks.ocr_benchmark import OCRBenchmark

BENCHMARK_REGISTRY: dict[str, type] = {
    "ocr": OCRBenchmark,
    # Add more here:
    # "my_benchmark": MyBenchmark,
}

__all__ = ["BENCHMARK_REGISTRY", "OCRBenchmark"]
