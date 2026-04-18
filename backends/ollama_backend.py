"""Ollama backend — uses /api/generate with stream=true."""
from __future__ import annotations
import json
from typing import Generator
import httpx
from backends.base import BaseBackend, GenerateParams


class OllamaBackend(BaseBackend):
    name = "ollama"

    def health_check(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def generate_stream(self, params: GenerateParams) -> Generator[str, None, None]:
        payload = {
            "model": self.model,
            "prompt": params.prompt,
            "stream": True,
            "options": {
                "temperature": params.temperature,
                "top_p": params.top_p,
                "num_predict": params.max_tokens,
            },
        }
        if params.stop:
            payload["options"]["stop"] = params.stop

        with httpx.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
