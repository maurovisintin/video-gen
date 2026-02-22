"""Pydantic models for video scripts."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Scene(BaseModel):
    """A single scene in the video script."""

    scene_number: int = Field(ge=1, description="Scene number (1-indexed)")
    video_prompt: str = Field(
        min_length=10,
        description=(
            "Detailed visual description for the video generation model. "
            "Should describe camera angle, subject, motion, lighting, and style."
        ),
    )
    narration_text: str = Field(
        min_length=1,
        description="Text to be spoken as narration during this scene.",
    )
    duration_seconds: float = Field(
        ge=2.0,
        le=6.0,
        description="Target duration of this scene in seconds (2-6s).",
    )


class VideoScript(BaseModel):
    """Complete script for a short-form video."""

    title: str = Field(description="Short, catchy title for the video.")
    topic: str = Field(description="The topic the video is about.")
    scenes: list[Scene] = Field(
        min_length=3,
        max_length=12,
        description="Ordered list of scenes.",
    )
    style_notes: str = Field(
        description=(
            "Visual style guidance applied to all scenes. "
            "E.g. 'cinematic, vibrant colors, smooth camera movements'."
        ),
    )
    negative_prompt: str = Field(
        default="blurry, low quality, distorted, watermark, text overlay",
        description="Negative prompt applied to all video generation.",
    )

    @model_validator(mode="after")
    def validate_total_duration(self) -> VideoScript:
        total = sum(s.duration_seconds for s in self.scenes)
        if total < 15:
            raise ValueError(
                f"Total duration is {total:.1f}s, minimum is 15s."
            )
        if total > 60:
            raise ValueError(
                f"Total duration is {total:.1f}s, maximum is 60s."
            )
        return self

    @property
    def total_duration(self) -> float:
        return sum(s.duration_seconds for s in self.scenes)
