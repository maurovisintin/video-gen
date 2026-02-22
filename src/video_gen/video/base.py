"""Abstract base class for video generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class VideoGenerator(ABC):
    """Base class for AI video generation."""

    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory. Call once before generating clips."""
        ...

    @abstractmethod
    def generate_clip(
        self,
        prompt: str,
        negative_prompt: str,
        output_path: Path,
        num_frames: int = 81,
        fps: int = 15,
    ) -> Path:
        """Generate a video clip from a text prompt.

        Args:
            prompt: Visual description of the scene.
            negative_prompt: What to avoid in generation.
            output_path: Where to save the MP4 clip.
            num_frames: Number of frames to generate (default 81 ≈ 5.4s at 15fps).
            fps: Frames per second for the output video.

        Returns:
            Path to the generated clip.
        """
        ...

    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model from memory."""
        ...
