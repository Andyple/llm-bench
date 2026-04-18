"""llama.cpp server backend — uses /completion with stream=true."""
from __future__ import annotations
import json
from typing import Generator
import httpx
from backends.base import BaseBackend, GenerateParams


class LlamaCppBackend(BaseBackend):
    name = "llama.cpp"

    def health_check(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/health", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        # llama.cpp server serves a single model; return it from /props if available
        try:
            r = httpx.get(f"{self.base_url}/props", timeout=5)
            data = r.json()
            model_name = data.get("model_path", self.model) or self.model
            return [model_name]
        except Exception:
            return [self.model]

    def generate_stream(self, params: GenerateParams) -> Generator[str, None, None]:
        payload = {
            "prompt": params.prompt,
            "n_predict": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "stream": True,
        }
        if params.stop:
            payload["stop"] = params.stop

        with httpx.stream(
            "POST",
            f"{self.base_url}/completion",
            json=payload,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                try:
                    chunk = json.loads(data_str)
                    token = chunk.get("content", "")
                    if token:
                        yield token
                    if chunk.get("stop"):
                        break
                except json.JSONDecodeError:
                    continue
