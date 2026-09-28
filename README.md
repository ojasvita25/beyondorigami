# BeyondOrigami

Beyond Origami explores Origami x AI.
Upload an image of your fold and prompt to reimagine your Origami.

<p align="center">
  <img src="demo.gif" alt="BeyondOrigami Demo" width="720" />
</p>

## What's the tech?

ComfyUI workflow served through a FastAPI backend, with a simple browser-based frontend.

- **Model**: DreamShaper8_LCM (Stable Diffusion 1.5, LCM variant)
- **LoRA**: Lorena_Style
- **Sampler**: LCM, 2 steps, denoise 0.4
- **Resolution**: 512x512

The frontend sends a base64 image + text prompt to the FastAPI server, which queues a ComfyUI workflow over WebSocket and streams the result back.

## How to run?

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

**Run**

```bash
uvicorn main:app --reload --port 8000
```

Then open `index.html` in your browser.

## What to expect?

1. Click + to upload an image of your origami fold (or drag and drop)
2. Type a prompt describing how you want it reimagined
3. Click the arrow to generate
4. The result appears as the fullscreen background in about 2 seconds

You can also use Surprise Dice after 6+ generations to create a 2x3 collage of past outputs.

## Performance

Benchmarked with `benchmark.py` using 2 demo images, 3 runs each.

| Metric | Value |
|--------|-------|
| Avg. inference time | 2.70s |
| Median inference time | 2.31s |
| Cold start (first run) | 4-24s |

First generation is slower because ComfyUI loads the model into memory. After that, each generation takes about 2 seconds.

## License

This project is open source. Feel free to use and modify.
