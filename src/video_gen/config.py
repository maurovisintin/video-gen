"""Platform detection, device/dtype selection, and path management."""

from __future__ import annotations

import os
import platform
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
ASSETS_DIR = PROJECT_ROOT / "assets"
REFERENCE_VOICES_DIR = ASSETS_DIR / "reference_voices"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Device detection ───────────────────────────────────────────────────────

def get_device() -> str:
    """Return the best available torch device string."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device: str | None = None) -> torch.dtype:
    """Return the appropriate dtype for the device.

    CUDA supports bfloat16 for faster inference.
    MPS and CPU require float32 for Wan2.1 compatibility.
    """
    if device is None:
        device = get_device()
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def get_video_dtype(device: str | None = None) -> torch.dtype:
    """Return dtype specifically for the video generation pipeline."""
    return get_dtype(device)


def get_vae_dtype(device: str | None = None) -> torch.dtype:
    """VAE always needs float32 on MPS."""
    if device is None:
        device = get_device()
    if device == "mps":
        return torch.float32
    return get_dtype(device)


# ── Environment helpers ────────────────────────────────────────────────────

def setup_environment() -> None:
    """Set environment variables needed for platform compatibility."""
    if get_device() == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_anthropic_api_key() -> str:
    """Return the Anthropic API key from the environment."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. "
            "Copy .env.example to .env and add your key."
        )
    return key


# ── Platform info ──────────────────────────────────────────────────────────

def platform_summary() -> dict[str, str]:
    """Return a dict of platform info for debugging."""
    device = get_device()
    return {
        "os": platform.system(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
        "dtype": str(get_dtype(device)),
    }
