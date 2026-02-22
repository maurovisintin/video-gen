"""Pipeline orchestrator: topic -> final video."""

from __future__ import annotations

import json
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from video_gen.compose.compositor import ClipWithAudio, Compositor
from video_gen.config import OUTPUT_DIR, REFERENCE_VOICES_DIR
from video_gen.models.script import VideoScript
from video_gen.script.claude import ClaudeScriptGenerator
from video_gen.tts.f5tts import F5TTSEngine
from video_gen.video.wan import WanVideoGenerator

console = Console()


def generate_script(topic: str) -> VideoScript:
    """Generate a video script for the given topic."""
    console.print(Panel(f"[bold]Topic:[/bold] {topic}", title="Script Generation"))

    generator = ClaudeScriptGenerator()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating script with Claude...", total=None)
        script = generator.generate(topic)
        progress.update(task, completed=True)

    console.print(f"\n[green]Script generated:[/green] {script.title}")
    console.print(f"  Scenes: {len(script.scenes)}")
    console.print(f"  Total duration: {script.total_duration:.1f}s")

    for scene in script.scenes:
        console.print(
            f"\n  [bold]Scene {scene.scene_number}[/bold] "
            f"({scene.duration_seconds}s)"
        )
        console.print(f"    Narration: {scene.narration_text}")
        console.print(f"    Visual: {scene.video_prompt[:80]}...")

    return script


def run_pipeline(
    topic: str,
    output_dir: Path | None = None,
    reference_voice: Path | None = None,
    skip_video: bool = False,
) -> Path:
    """Run the full pipeline: topic -> final video.

    Args:
        topic: The video topic.
        output_dir: Output directory (defaults to output/<timestamp>/).
        reference_voice: Path to reference audio for voice cloning.
        skip_video: If True, skip video generation (for testing TTS + composition).

    Returns:
        Path to the final video.
    """
    # Set up output directory
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = OUTPUT_DIR / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect reference voice if not provided
    if reference_voice is None:
        voices = list(REFERENCE_VOICES_DIR.glob("*.wav"))
        if voices:
            reference_voice = voices[0]
            console.print(f"Using reference voice: {reference_voice.name}")

    # ── Stage 1: Script Generation ──────────────────────────────────────
    script = generate_script(topic)

    # Save script to disk
    script_path = output_dir / "script.json"
    script_path.write_text(json.dumps(script.model_dump(), indent=2))
    console.print(f"\nScript saved to: {script_path}")

    # ── Stage 2: TTS Narration ──────────────────────────────────────────
    console.print(Panel("Generating narration audio", title="TTS"))

    tts = F5TTSEngine(device="cpu")  # Keep GPU free for video
    tts_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for scene in script.scenes:
            task = progress.add_task(
                f"Scene {scene.scene_number}: TTS...", total=None
            )
            audio_path = output_dir / f"scene_{scene.scene_number:02d}_audio.wav"

            result = tts.synthesize(
                text=scene.narration_text,
                output_path=audio_path,
                reference_audio=reference_voice,
            )
            tts_results.append(result)
            progress.update(
                task,
                description=(
                    f"Scene {scene.scene_number}: "
                    f"{result.duration_seconds:.1f}s audio"
                ),
                completed=True,
            )

    # ── Stage 3: Video Generation ───────────────────────────────────────
    clip_paths: list[Path] = []

    if not skip_video:
        console.print(Panel("Generating video clips", title="Video Generation"))

        video_gen = WanVideoGenerator()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            for i, scene in enumerate(script.scenes):
                task = progress.add_task(
                    f"Scene {scene.scene_number}: Generating video...",
                    total=None,
                )
                clip_path = output_dir / f"scene_{scene.scene_number:02d}_clip.mp4"

                # Prepend style notes to each scene prompt
                full_prompt = f"{script.style_notes}. {scene.video_prompt}"

                try:
                    video_gen.generate_clip(
                        prompt=full_prompt,
                        negative_prompt=script.negative_prompt,
                        output_path=clip_path,
                    )
                    clip_paths.append(clip_path)
                    progress.update(
                        task,
                        description=f"Scene {scene.scene_number}: Done",
                        completed=True,
                    )
                except Exception as e:
                    console.print(
                        f"[red]Scene {scene.scene_number} failed: {e}[/red]"
                    )
                    # Retry once
                    try:
                        console.print("  Retrying...")
                        video_gen.generate_clip(
                            prompt=full_prompt,
                            negative_prompt=script.negative_prompt,
                            output_path=clip_path,
                        )
                        clip_paths.append(clip_path)
                        progress.update(task, completed=True)
                    except Exception as e2:
                        console.print(
                            f"[red]  Retry failed: {e2}. Skipping scene.[/red]"
                        )

        video_gen.unload_model()
    else:
        console.print("[yellow]Skipping video generation (--skip-video)[/yellow]")

    # ── Stage 4: Composition ────────────────────────────────────────────
    if clip_paths:
        console.print(Panel("Composing final video", title="Composition"))

        clips_with_audio = []
        for i, clip_path in enumerate(clip_paths):
            audio_path = tts_results[i].audio_path if i < len(tts_results) else None
            target_dur = (
                tts_results[i].duration_seconds if i < len(tts_results) else None
            )
            clips_with_audio.append(
                ClipWithAudio(
                    video_path=clip_path,
                    audio_path=audio_path,
                    target_duration=target_dur,
                )
            )

        compositor = Compositor()
        final_path = output_dir / "final.mp4"

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Compositing final video...", total=None)
            compositor.compose(clips_with_audio, final_path)
            progress.update(task, completed=True)

        console.print(f"\n[bold green]Final video:[/bold green] {final_path}")
        return final_path
    else:
        console.print("[yellow]No clips generated. Skipping composition.[/yellow]")
        return output_dir / "script.json"


def compose_from_directory(project_dir: Path) -> Path:
    """Compose a final video from an existing project directory.

    Expects the directory to contain:
    - script.json
    - scene_*_clip.mp4 files
    - scene_*_audio.wav files
    """
    project_dir = Path(project_dir)

    script_path = project_dir / "script.json"
    if not script_path.exists():
        raise FileNotFoundError(f"No script.json found in {project_dir}")

    script = VideoScript.model_validate_json(script_path.read_text())

    clips = sorted(project_dir.glob("scene_*_clip.mp4"))
    audios = sorted(project_dir.glob("scene_*_audio.wav"))

    if not clips:
        raise FileNotFoundError(f"No video clips found in {project_dir}")

    clips_with_audio = []
    for i, clip_path in enumerate(clips):
        audio_path = audios[i] if i < len(audios) else None
        clips_with_audio.append(
            ClipWithAudio(
                video_path=clip_path,
                audio_path=audio_path,
                target_duration=script.scenes[i].duration_seconds
                if i < len(script.scenes)
                else None,
            )
        )

    compositor = Compositor()
    final_path = project_dir / "final.mp4"
    compositor.compose(clips_with_audio, final_path)

    console.print(f"\n[bold green]Final video:[/bold green] {final_path}")
    return final_path
