"""Wan2.1 1.3B video generation implementation."""

from __future__ import annotations

from pathlib import Path

import torch
from diffusers.utils import export_to_video

from video_gen.config import get_device, get_dtype, get_vae_dtype, setup_environment

from .base import VideoGenerator


class WanVideoGenerator(VideoGenerator):
    """Generate video clips using Wan2.1 T2V 1.3B via Diffusers."""

    MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    # Native 9:16 resolution for the 1.3B model
    WIDTH = 480
    HEIGHT = 832

    def __init__(self) -> None:
        self.device = get_device()
        self.dtype = get_dtype(self.device)
        self.vae_dtype = get_vae_dtype(self.device)
        self._pipe = None
        setup_environment()

    def load_model(self) -> None:
        """Load Wan2.1 pipeline into memory."""
        if self._pipe is not None:
            return

        from diffusers import AutoencoderKLWan, WanPipeline

        # Ensure the download timeout is applied even if huggingface_hub
        # was imported before our env var was set.
        import huggingface_hub.constants
        huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT = 300

        # Load VAE with appropriate dtype (float32 on MPS)
        vae = AutoencoderKLWan.from_pretrained(
            self.MODEL_ID,
            subfolder="vae",
            torch_dtype=self.vae_dtype,
        )

        self._pipe = WanPipeline.from_pretrained(
            self.MODEL_ID,
            vae=vae,
            torch_dtype=self.dtype,
        )

        if self.device == "mps":
            # On MPS: move pipeline to MPS but keep T5 encoder on CPU
            # to avoid MPS memory issues with the text encoder.
            # Do NOT call enable_model_cpu_offload() — it resets .to("mps")
            # by calling .to("cpu") internally, and its accelerate hooks
            # are not reliable on MPS.
            self._pipe.to("mps")
            self._pipe.text_encoder.to("cpu")
        else:
            self._pipe.to(self.device)
            # enable_model_cpu_offload only works reliably on CUDA
            if self.device == "cuda":
                try:
                    self._pipe.enable_model_cpu_offload()
                except Exception:
                    pass

    def generate_clip(
        self,
        prompt: str,
        negative_prompt: str,
        output_path: Path,
        num_frames: int = 81,
        fps: int = 15,
    ) -> Path:
        """Generate a video clip from a text prompt.

        Args:
            prompt: Visual description of the scene.
            negative_prompt: What to avoid in generation.
            output_path: Where to save the MP4 clip.
            num_frames: Number of frames (default 81 ≈ 5.4s at 15fps).
            fps: Frames per second.

        Returns:
            Path to the generated MP4 clip.
        """
        if self._pipe is None:
            self.load_model()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=self.WIDTH,
            height=self.HEIGHT,
            num_frames=num_frames,
            num_inference_steps=30,
            guidance_scale=5.0,
        )

        frames = output.frames[0]
        export_to_video(frames, str(output_path), fps=fps)

        return output_path

    def unload_model(self) -> None:
        """Unload the model and free memory."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
