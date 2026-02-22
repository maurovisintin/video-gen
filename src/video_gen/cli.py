"""CLI entry point using Typer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from video_gen.config import platform_summary, setup_environment

app = typer.Typer(
    name="video-gen",
    help="AI TikTok video generation pipeline.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main() -> None:
    """AI TikTok video generation pipeline."""
    setup_environment()


@app.command()
def create(
    topic: str = typer.Argument(help="Topic for the video."),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory."
    ),
    reference_voice: Optional[Path] = typer.Option(
        None, "--voice", "-v", help="Reference audio for voice cloning."
    ),
    skip_video: bool = typer.Option(
        False, "--skip-video", help="Skip video generation (test TTS only)."
    ),
    engine: str = typer.Option(
        "ollama", "--engine", "-e", help="Script generation engine (ollama or claude)."
    ),
) -> None:
    """Generate a complete TikTok video from a topic."""
    from video_gen.pipeline import run_pipeline

    console.print(
        Panel(
            "[bold]AI TikTok Video Generator[/bold]",
            subtitle="Full Pipeline",
        )
    )

    result = run_pipeline(
        topic=topic,
        output_dir=output_dir,
        reference_voice=reference_voice,
        skip_video=skip_video,
        engine=engine,
    )
    console.print(f"\nOutput: {result}")


@app.command()
def script(
    topic: str = typer.Argument(help="Topic for the video script."),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save script JSON to file."
    ),
    engine: str = typer.Option(
        "ollama", "--engine", "-e", help="Script generation engine (ollama or claude)."
    ),
) -> None:
    """Generate a video script only (no audio or video)."""
    from video_gen.pipeline import generate_script

    video_script = generate_script(topic, engine=engine)

    script_json = json.dumps(video_script.model_dump(), indent=2)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(script_json)
        console.print(f"\nScript saved to: {output}")
    else:
        console.print(f"\n{script_json}")


@app.command()
def compose(
    project_dir: Path = typer.Argument(
        help="Directory containing clips and audio to compose."
    ),
) -> None:
    """Compose a final video from existing clips and audio."""
    from video_gen.pipeline import compose_from_directory

    compose_from_directory(project_dir)


@app.command()
def info() -> None:
    """Show platform and device information."""
    summary = platform_summary()
    console.print(Panel("[bold]Platform Info[/bold]"))
    for key, value in summary.items():
        console.print(f"  {key}: {value}")
