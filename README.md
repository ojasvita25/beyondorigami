---
title: BeyondOrigami
emoji: 🎨
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.36.0"
app_file: app.py
pinned: false
license: mit
short_description: Origami x AI reimaginings powered by SD 1.5 LCM
---

# BeyondOrigami

Beyond Origami explores Origami x AI.  
Upload an image of your fold and prompt to reimagine your Origami.

<p align="center">
  <img src="demo.gif" alt="BeyondOrigami Demo" width="720" />
</p>

## What's the tech?

Beyond Origami supports two operational backends:

- **Hugging Face Spaces Mode (`app.py`)**: Uses PyTorch + `diffusers` (`AutoPipelineForImage2Image` + `LCMScheduler`) accelerated by **Hugging Face ZeroGPU** (`@spaces.GPU`).
- **Local Mode (`main.py`)**: Connects to a local ComfyUI instance served through a FastAPI backend.

### Model Specs
- **Model**: DreamShaper8_LCM (Stable Diffusion 1.5, LCM variant)
- **LoRA**: Lorena_Style / LCM LoRA
- **Sampler**: LCM, 2-4 steps, denoise 0.4
- **Resolution**: 512x512

The browser-based frontend sends a base64 image + text prompt to the FastAPI server (`/get-image`), which streams the reimagined image back in real time.

---

## How to run?

### 1. Deploying to Hugging Face Spaces (ZeroGPU)

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces) with **Gradio** SDK and **ZeroGPU** hardware.
2. Push this repository to your Hugging Face Space:

```bash
git remote add hf https://huggingface.co/spaces/<your-username>/beyondorigami
git add .
git commit -m "Deploy to HF Spaces"
git push hf main
```

Your Space will automatically build and launch the interactive web interface!

---

### 2. Running Locally (with ComfyUI)

**Prerequisites**
- Python 3.10+
- ComfyUI running on `127.0.0.1:8188`
- `DreamShaper8_LCM.safetensors` in ComfyUI's `models/checkpoints/`
- `Lorena_Style.safetensors` in ComfyUI's `models/loras/`
- ComfyUI custom nodes: `SaveImageWebsocket`, `ETN_LoadImageBase64`

**Setup**
```bash
git clone https://github.com/<your-username>/beyondorigami.git
cd beyondorigami
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Run Backend**
```bash
uvicorn main:app --reload --port 8000
```

Open `index.html` directly in your browser or navigate to `http://127.0.0.1:8000`.

---

## Features & What to expect

1. **Upload & Reimagine**: Click `+` to upload an image of your origami fold (or drag and drop), enter a prompt, and click `→`.
2. **Instant Visual Feedback**: The reimagined result displays as a full-screen dynamic background.
3. **Surprise Dice**: After generating 6+ variations, click *Surprise Dice* to create and download a high-resolution 2x3 collage of your past origami creations.
4. **History Management**: Easily clear local session history using the *Clear History* button.

---

## Performance

Benchmarked with `benchmark.py` using 2 demo images, 3 runs each.

| Metric | Value |
|--------|-------|
| Avg. inference time | 2.70s |
| Median inference time | 2.31s |
| Cold start (first run) | 4-24s |

First generation is slower because the model loads into GPU memory. Subsequent generations take ~2 seconds.

---

## License

This project is open source. Feel free to use and modify.
