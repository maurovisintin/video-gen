"""Abstract base class for TTS engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TTSResult:
    """Result from TTS synthesis."""

    audio_path: Path
    duration_seconds: float


class TTSEngine(ABC):
    """Base class for text-to-speech synthesis."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: Path,
        reference_audio: Path | None = None,
    ) -> TTSResult:
        """Synthesize speech from text.

        Args:
            text: The text to speak.
            output_path: Where to save the WAV file.
            reference_audio: Optional reference audio for voice cloning.

        Returns:
            TTSResult with the audio path and actual duration.
        """
        ...
