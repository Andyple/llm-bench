"""
Abstract base class for pluggable task benchmarks (OCR, QA, etc.).
Implement this to add a new benchmark type.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from backends.base import BaseBackend
from core.models import BenchmarkResult


class BaseBenchmark(ABC):
    name: str = "base"
    score_label: str = "score"

    @abstractmethod
    def load_dataset(self) -> list[dict]:
        """
        Return a list of samples. Each sample is a dict with at least:
          - 'id': str
          - 'input': the data passed to the model (text, image path, etc.)
          - 'reference': the ground truth output
        """
        ...

    @abstractmethod
    def build_prompt(self, sample: dict) -> str:
        """Convert a sample into a prompt string for the model."""
        ...

    @abstractmethod
    def score_sample(self, prediction: str, reference: str) -> float:
        """Return a scalar score for one sample (higher = better, range depends on metric)."""
        ...

    def run(
        self,
        backend: BaseBackend,
        quant: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        sample_limit: int | None = None,
    ) -> list[BenchmarkResult]:
        import time
        from backends.base import GenerateParams

        dataset = self.load_dataset()
        if sample_limit:
            dataset = dataset[:sample_limit]

        results = []
        for sample in dataset:
            prompt = self.build_prompt(sample)
            params = GenerateParams(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            t0 = time.perf_counter()
            tokens = list(backend.generate_stream(params))
            latency = time.perf_counter() - t0
            prediction = "".join(tokens).strip()
            score = self.score_sample(prediction, sample["reference"])
            results.append(BenchmarkResult(
                backend=backend.name,
                model=backend.model,
                quant=quant,
                benchmark_name=self.name,
                sample_id=str(sample["id"]),
                prediction=prediction,
                reference=sample["reference"],
                score=score,
                score_label=self.score_label,
                latency_s=latency,
            ))
        return results
