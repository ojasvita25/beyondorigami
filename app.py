import os
import torch
import spaces
from PIL import Image
from diffusers import AutoPipelineForImage2Image, LCMScheduler
import gradio as gr

# Load SD 1.5 DreamShaper 8 Pipeline at startup
MODEL_ID = "Lykon/dreamshaper-8"
pipe = AutoPipelineForImage2Image.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# Load LCM LoRA for fast inference
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

# Auto-detect and load custom Lorena_Style LoRA if uploaded to the space
CUSTOM_LORA = "Lorena_Style.safetensors"
if os.path.exists(CUSTOM_LORA):
    try:
        pipe.load_lora_weights(".", weight_name=CUSTOM_LORA, adapter_name="lorena")
        print(f"Loaded custom LoRA: {CUSTOM_LORA}")
    except Exception as e:
        print(f"Failed to load {CUSTOM_LORA}: {e}")


# HF ZeroGPU function allocation
@spaces.GPU
def generate_image(input_image: Image.Image, prompt: str, steps: int = 8, cfg: float = 1.5, strength: float = 0.45):
    if input_image is None or not prompt or not prompt.strip():
        return None

    # 1. Capture original dimensions & aspect ratio
    orig_w, orig_h = input_image.size

    # 2. Prevent 0 actual steps calculation in diffusers (steps * strength >= 1)
    actual_steps = int(steps * strength)
    if actual_steps < 1:
        steps = int(1.0 / max(strength, 0.05)) + 1

    # 3. Resize to 512x512 for pipeline inference
    input_img = input_image.convert("RGB").resize((512, 512), Image.LANCZOS)
    pipe.to("cuda")
    output = pipe(
        prompt=prompt,
        negative_prompt="bad anatomy, extra fingers, watermark, blurred, low quality, distortion, noise",
        image=input_img,
        num_inference_steps=int(steps),
        guidance_scale=float(cfg),
        strength=float(strength),
    ).images[0]

    # 4. Resize output back to the original input aspect ratio & size
    output = output.resize((orig_w, orig_h), Image.LANCZOS)
    return output


# Clean & simple Gradio UI for Hugging Face Space
with gr.Blocks(title="Beyond Origami") as demo:
    gr.Markdown("# 🎨 Beyond Origami\nUpload an image of your fold or pick a sample origami image below, then enter a prompt to reimagine your Origami.")

    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="pil", label="Upload or Pick Fold Image")
            prompt_in = gr.Textbox(
                label="Prompt",
                placeholder="e.g. bioluminescent crystal wings, masterpiece, highly detailed",
                lines=2
            )

            with gr.Accordion("⚙️ Quality & Advanced Settings", open=False):
                steps_slider = gr.Slider(minimum=4, maximum=20, value=8, step=1, label="Inference Steps (LCM)")
                strength_slider = gr.Slider(minimum=0.10, maximum=0.90, value=0.45, step=0.05, label="Denoise Strength")
                cfg_slider = gr.Slider(minimum=1.0, maximum=4.0, value=1.5, step=0.1, label="CFG Scale")

            btn = gr.Button("✨ Reimagine Fold", variant="primary")

        with gr.Column():
            img_out = gr.Image(type="pil", label="Reimagined Origami")

    btn.click(
        fn=generate_image,
        inputs=[img_in, prompt_in, steps_slider, cfg_slider, strength_slider],
        outputs=img_out,
        api_name=False
    )

    gr.Examples(
        examples=[
            ["static/examples/butterfly.jpg", "futuristic"],
            ["static/examples/spider.jpg", "cat"],
        ],
        inputs=[img_in, prompt_in],
        outputs=img_out,
        fn=generate_image,
        cache_examples=False,
        label="💡 Sample Origami Folds & Prompts (Click to try)"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
