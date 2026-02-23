# WSL2 + CUDA Setup Guide

Run **video-gen** on Windows with full GPU acceleration via WSL2.

Tested with: RTX 4070 Ti Super (16 GB VRAM), Ubuntu 24.04 on WSL2.

---

## Prerequisites (Windows side)

- **Windows 11** (or Windows 10 21H2+) with WSL2 enabled
- **Latest NVIDIA GPU driver** installed on Windows (do NOT install a driver inside WSL — it's shared automatically)
- **WSL2 with Ubuntu:**
  ```powershell
  wsl --install -d Ubuntu-24.04
  ```

---

## Setup (inside WSL)

### 1. Verify CUDA passthrough

```bash
nvidia-smi
```

You should see your GPU listed (e.g. `NVIDIA GeForce RTX 4070 Ti SUPER`). If this doesn't work, see [Troubleshooting](#troubleshooting) below.

### 2. Install system dependencies

```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv ffmpeg git
```

### 3. Clone and create a virtual environment

```bash
git clone <repo-url> ~/video-gen && cd ~/video-gen
python3.11 -m venv .venv
source .venv/bin/activate
```

### 4. Install PyTorch with CUDA (before the project)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 5. Install the project

```bash
pip install -e .
```

### 6. Create `.env`

```bash
cp .env.example .env
# Edit .env and add your keys:
#   ANTHROPIC_API_KEY=sk-...   (required for Claude engine)
#   HF_TOKEN=hf_...            (required for model downloads)
```

### 7. (Optional) Install Ollama

Only needed if you want to use the default `ollama` script engine instead of Claude.

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:7b
# ollama serve runs automatically as a systemd service
```

### 8. Verify the setup

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# → True NVIDIA GeForce RTX 4070 Ti SUPER

video-gen info
# → device: cuda, dtype: torch.bfloat16
```

### 9. First run

Wan2.1 model weights (~3.5 GB) are downloaded automatically on the first run.

```bash
video-gen create "your topic"
```

---

## Performance notes

- **bfloat16 on CUDA** — faster inference and lower VRAM usage than float32
- `enable_model_cpu_offload()` works reliably on CUDA
- Expect **~8–15x faster** per-scene video generation compared to an M2 Mac (MPS)
- 16 GB VRAM is plenty for the 1.3B model

---

## Tips

- **Keep the project in the WSL filesystem** (`~/video-gen`), not on the Windows mount (`/mnt/c/...`) — I/O is significantly faster
- **Access output from Windows** at `\\wsl$\Ubuntu\home\<user>\video-gen\output\`
- Use `video-gen create "topic" -e claude` to use the Claude script engine, or omit `-e` for the default Ollama engine

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `nvidia-smi` not found or shows no GPU | Update the NVIDIA driver **on Windows** (not inside WSL), then run `wsl --shutdown` in PowerShell and reopen your terminal |
| CUDA out of memory | Close other GPU-using apps; the 1.3B model should fit comfortably in 16 GB |
| Slow file I/O | Move the project to `~/` inside WSL instead of `/mnt/c/` |
| `ffmpeg` not found | Run `sudo apt install -y ffmpeg` |
