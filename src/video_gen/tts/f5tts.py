"""F5-TTS implementation for narration synthesis."""

from __future__ import annotations

import wave
from pathlib import Path

from .base import TTSEngine, TTSResult


class F5TTSEngine(TTSEngine):
    """Text-to-speech using F5-TTS with optional voice cloning."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._model = None

    def _ensure_model(self) -> None:
        """Lazy-load the F5-TTS model on first use."""
        if self._model is not None:
            return

        from f5_tts.api import F5TTS

        self._model = F5TTS(device=self.device)

    def synthesize(
        self,
        text: str,
        output_path: Path,
        reference_audio: Path | None = None,
    ) -> TTSResult:
        """Synthesize speech from text using F5-TTS.

        Args:
            text: The text to speak.
            output_path: Where to save the WAV file.
            reference_audio: Optional reference audio for voice cloning.
                If not provided, uses F5-TTS default voice.

        Returns:
            TTSResult with audio path and actual duration.
        """
        self._ensure_model()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ref_audio = str(reference_audio) if reference_audio else None
        # F5-TTS needs reference text for voice cloning; empty string
        # triggers automatic transcription of reference audio
        ref_text = ""

        wav_path, _spectrogram, _ = self._model.infer(
            ref_file=ref_audio,
            ref_text=ref_text,
            gen_text=text,
            file_wave=str(output_path),
        )

        duration = _get_wav_duration(Path(wav_path))

        return TTSResult(audio_path=Path(wav_path), duration_seconds=duration)


def _get_wav_duration(path: Path) -> float:
    """Get the duration of a WAV file in seconds."""
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / rate
