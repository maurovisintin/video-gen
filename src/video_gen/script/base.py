"""Abstract base class for script generators."""

from __future__ import annotations

from abc import ABC, abstractmethod

from video_gen.models.script import VideoScript


class ScriptGenerator(ABC):
    """Base class for video script generation."""

    @abstractmethod
    def generate(self, topic: str) -> VideoScript:
        """Generate a video script for the given topic."""
        ...
