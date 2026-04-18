"""vLLM backend — uses the OpenAI-compatible /v1/completions streaming endpoint."""
from __future__ import annotations
import json
from typing import Generator
import httpx
from backends.base import BaseBackend, GenerateParams


class VLLMBackend(BaseBackend):
    name = "vllm"

    def health_check(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/health", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        try:
            r = httpx.get(f"{self.base_url}/v1/models", timeout=5)
            data = r.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []

    def generate_stream(self, params: GenerateParams) -> Generator[str, None, None]:
        payload = {
            "model": self.model,
            "prompt": params.prompt,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "stream": True,
        }
        if params.stop:
            payload["stop"] = params.stop

        with httpx.stream(
            "POST",
            f"{self.base_url}/v1/completions",
            json=payload,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    token = chunk["choices"][0].get("text", "")
                    if token:
                        yield token
                except (json.JSONDecodeError, KeyError):
                    continue
