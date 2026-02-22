"""Script generation using Claude API with Structured Outputs."""

from __future__ import annotations

import anthropic

from video_gen.config import get_anthropic_api_key
from video_gen.models.script import VideoScript

from .base import ScriptGenerator

SYSTEM_PROMPT = """\
You are an expert short-form video scriptwriter for TikTok.

Your job is to create engaging, fast-paced video scripts that:
- Hook the viewer in the first 2 seconds
- Deliver information in punchy, conversational narration
- Use vivid visual descriptions optimized for AI video generation
- Keep total video length between 15-60 seconds (3-12 scenes, 2-6s each)

Video prompt guidelines (for Wan2.1 text-to-video model):
- Describe concrete, filmable visuals—not abstract concepts
- Include camera movement (slow zoom, tracking shot, pan, static)
- Specify lighting (golden hour, dramatic shadows, bright studio)
- Mention subject position and motion
- Use a consistent visual style across all scenes
- Avoid text overlays, UI elements, or watermarks in descriptions
- Keep each prompt self-contained (the model generates each scene independently)

Narration guidelines:
- Short, punchy sentences (each scene's narration should take 2-6 seconds to speak)
- Conversational tone, as if talking to a friend
- Start with a hook: a question, surprising fact, or bold statement
- End with a call to action or memorable closing line
"""


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
                    "content": (
                        f"Create a TikTok video script about: {topic}\n\n"
                        "Make it engaging, informative, and visually striking. "
                        "Aim for 20-40 seconds total."
                    ),
                }
            ],
            output_format=VideoScript,
        )

        return response.parsed_output
