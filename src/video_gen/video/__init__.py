"""Video generation module."""

from .base import VideoGenerator
from .wan import WanVideoGenerator

__all__ = ["VideoGenerator", "WanVideoGenerator"]
