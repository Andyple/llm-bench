from backends.vllm_backend import VLLMBackend
from backends.ollama_backend import OllamaBackend
from backends.llamacpp_backend import LlamaCppBackend

BACKEND_REGISTRY = {
    "vllm": VLLMBackend,
    "ollama": OllamaBackend,
    "llama.cpp": LlamaCppBackend,
}

__all__ = ["VLLMBackend", "OllamaBackend", "LlamaCppBackend", "BACKEND_REGISTRY"]
