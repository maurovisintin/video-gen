"""Script generation using Claude API with Structured Outputs."""

from __future__ import annotations

import anthropic

from video_gen.config import get_anthropic_api_key
from video_gen.models.script import VideoScript

from .base import ScriptGenerator
from .prompts import SYSTEM_PROMPT, user_prompt


class ClaudeScriptGenerator(ScriptGenerator):
    """Generate video scripts using Claude Structured Outputs."""

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.client = anthropic.Anthropic(api_key=get_anthropic_api_key())
        self.model = model

    def generate(self, topic: str) -> VideoScript:
        """Generate a VideoScript for the given topic."""
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt(topic),
                }
            ],
            output_format=VideoScript,
        )

        return response.parsed_output
