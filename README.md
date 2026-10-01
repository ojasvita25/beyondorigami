---
title: BeyondOrigami
emoji: 🎨
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
short_description: Origami x AI reimaginings powered by SD 1.5 LCM on ZeroGPU
---

# 🎨 BeyondOrigami

**Beyond Origami** explores Origami x AI.  
Upload an image of your fold (or choose from built-in sample origami pictures) and enter a prompt to reimagine your origami artwork into stunning AI creations!

<p align="center">
  <img src="demo.gif" alt="BeyondOrigami Demo" width="720" />
</p>

---

## ⚡ Tech Stack & Architecture

- **Framework**: Gradio + PyTorch + Diffusers
- **Inference Pipeline**: `AutoPipelineForImage2Image` with `LCMScheduler`
- **Acceleration**: **Hugging Face ZeroGPU** (`@spaces.GPU`) & LCM LoRA for sub-second, real-time image-to-image synthesis
- **Model**: DreamShaper 8 (SD 1.5) + LCM LoRA (`latent-consistency/lcm-lora-sdv1-5`)
- **Custom LoRA**: Auto-detects and applies `Lorena_Style.safetensors` when uploaded to the Space

---

## 🚀 Running on Hugging Face Spaces

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces) with **Gradio** SDK and **ZeroGPU** hardware.
2. Push this repository to your Hugging Face Space:

```bash
git remote add hf https://huggingface.co/spaces/<your-username>/beyondorigami
git add .
git commit -m "Deploy BeyondOrigami to HF Spaces"
git push hf main
```

Your Space will automatically build, set up the GPU runtime, and launch the interactive web interface!

---

## ✨ Features

1. **Upload or Select Sample Origami**: Pick from built-in fold presets (Butterfly, Spider, Crane, Dragon, Flower) or upload your own photo.
2. **Prompt-driven Reimaginings**: Transform paper folds into glowing crystal dragons, bioluminescent butterflies, stardust roses, and more.
3. **Advanced Controls**: Fine-tune inference steps (LCM), denoise strength, and CFG scale using the advanced settings drawer.

---

## 📜 License

This project is open source under the MIT License.
