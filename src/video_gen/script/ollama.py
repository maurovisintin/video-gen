"""Script generation using a local Ollama model."""

from __future__ import annotations

import ollama
from pydantic import ValidationError

from video_gen.models.script import VideoScript

from .base import ScriptGenerator
from .prompts import SYSTEM_PROMPT, user_prompt

MAX_RETRIES = 3


class OllamaScriptGenerator(ScriptGenerator):
    """Generate video scripts using a local Ollama model."""

    def __init__(self, model: str = "qwen2.5:7b") -> None:
        self.model = model
        self.client = ollama.Client()

    def _call_ollama(self, topic: str) -> str:
        """Send the chat request and return raw content."""
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt(topic)},
                ],
                format=VideoScript.model_json_schema(),
            )
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                raise RuntimeError(
                    f"Model '{self.model}' not found. "
                    f"Run: ollama pull {self.model}"
                ) from exc
            raise
        except ConnectionError as exc:
            raise RuntimeError(
                "Cannot connect to Ollama. "
                "Make sure it is running: ollama serve"
            ) from exc
        return response.message.content

    def generate(self, topic: str) -> VideoScript:
        """Generate a VideoScript for the given topic.

        Retries up to MAX_RETRIES times if the model produces invalid output.
        """
        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            content = self._call_ollama(topic)
            try:
                return VideoScript.model_validate_json(content)
            except ValidationError as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    continue

        raise RuntimeError(
            f"Ollama failed after {MAX_RETRIES} attempts. "
            f"Try a larger model.\nLast error: {last_error}"
        ) from last_error
