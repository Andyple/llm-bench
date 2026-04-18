"""
Abstract base class for all inference backends.
Each backend must implement `generate_stream` which yields token strings
and exposes timing hooks used by the runner.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generator, Optional
from dataclasses import dataclass


@dataclass
class GenerateParams:
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    stop: Optional[list[str]] = None


class BaseBackend(ABC):
    name: str = "base"

    def __init__(self, host: str, port: int, model: str):
        self.host = host
        self.port = port
        self.model = model

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the backend is reachable and ready."""
        ...

    @abstractmethod
    def generate_stream(self, params: GenerateParams) -> Generator[str, None, None]:
        """
        Yield decoded token strings one at a time.
        Must yield at minimum one token before blocking.
        """
        ...

    @abstractmethod
    def list_models(self) -> list[str]:
        """Return available model identifiers."""
        ...
