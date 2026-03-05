"""
Centralized configuration loading from environment variables.
"""
import os
from pathlib import Path


def get_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def get_env_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


def get_env_float(key: str, default: float = 0.0) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


# ─── Pipeline defaults from environment ──────────────────────────────────────
WHISPER_MODEL       = get_env("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE      = get_env("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE     = get_env("WHISPER_COMPUTE_TYPE", "float16")

OLLAMA_HOST         = get_env("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL        = get_env("OLLAMA_MODEL", "qwen2.5:32b")
LLM_LANGUAGE        = get_env("LLM_LANGUAGE", "fr")

OCR_ENABLED         = get_env_bool("OCR_ENABLED", True)
OCR_LANGUAGES       = get_env("OCR_LANGUAGES", "fr,en").split(",")
OCR_FPS_SAMPLE      = get_env_float("OCR_FPS_SAMPLE", 1.0)
OCR_MIN_CONFIDENCE  = get_env_float("OCR_MIN_CONFIDENCE", 0.5)

OUTPUT_DIR          = get_env("OUTPUT_DIR", "./outputs")
UPLOAD_DIR          = get_env("UPLOAD_DIR", "./uploads")
REDIS_URL           = get_env("REDIS_URL", "redis://localhost:6379/0")
