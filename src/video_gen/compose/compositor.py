"""Video composition: stitch clips, overlay audio, upscale, and export."""

from __future__ import annotations

from pathlib import Path

from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
    concatenate_videoclips,
    VideoFileClip,
)


# TikTok output specs
TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920


class Compositor:
    """Stitch video clips and narration audio into a final TikTok video."""

    def __init__(
        self,
        crossfade_duration: float = 0.3,
        output_fps: int = 30,
    ) -> None:
        self.crossfade_duration = crossfade_duration
        self.output_fps = output_fps

    def compose(
        self,
        clips: list[ClipWithAudio],
        output_path: Path,
    ) -> Path:
        """Compose clips and audio into a final video.

        Args:
            clips: List of ClipWithAudio (video path + optional audio path
                   + target duration).
            output_path: Where to save the final MP4.

        Returns:
            Path to the final video.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        video_segments = []
        for clip_info in clips:
            segment = self._prepare_segment(clip_info)
            video_segments.append(segment)

        if self.crossfade_duration > 0 and len(video_segments) > 1:
            final = concatenate_videoclips(
                video_segments,
                method="compose",
                padding=-self.crossfade_duration,
            )
        else:
            final = concatenate_videoclips(video_segments, method="compose")

        final.write_videofile(
            str(output_path),
            fps=self.output_fps,
            codec="libx264",
            audio_codec="aac",
            bitrate="8000k",
            preset="medium",
            logger=None,
        )

        # Clean up
        for seg in video_segments:
            seg.close()
        final.close()

        return output_path

    def _prepare_segment(self, clip_info: ClipWithAudio) -> VideoFileClip:
        """Load, resize, and attach audio to a single clip."""
        video = VideoFileClip(str(clip_info.video_path))

        # Upscale to 1080x1920 (TikTok resolution)
        if video.w != TARGET_WIDTH or video.h != TARGET_HEIGHT:
            video = video.resized((TARGET_WIDTH, TARGET_HEIGHT))

        # Adjust video duration to match narration if audio is provided
        if clip_info.audio_path and clip_info.audio_path.exists():
            audio = AudioFileClip(str(clip_info.audio_path))

            # If video is shorter than audio, slow it down to match
            if video.duration < audio.duration:
                speed_factor = video.duration / audio.duration
                video = video.with_speed_scaled(speed_factor)
            # If video is much longer than audio, trim it
            elif video.duration > audio.duration + 0.5:
                video = video.subclipped(0, audio.duration + 0.3)

            video = video.with_audio(audio)
        elif clip_info.target_duration:
            # No audio but a target duration—trim or loop
            if video.duration > clip_info.target_duration:
                video = video.subclipped(0, clip_info.target_duration)

        return video


class ClipWithAudio:
    """Container for a video clip path paired with its narration audio."""

    def __init__(
        self,
        video_path: Path,
        audio_path: Path | None = None,
        target_duration: float | None = None,
    ) -> None:
        self.video_path = video_path
        self.audio_path = audio_path
        self.target_duration = target_duration
