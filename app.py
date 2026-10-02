import os
import random
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
def generate_image(input_image: Image.Image, prompt: str, steps: int = 8, cfg: float = 1.5, strength: float = 0.45, seed: int = 42, randomize_seed: bool = True):
    if input_image is None or not prompt or not prompt.strip():
        return None

    # 1. Capture original dimensions & aspect ratio
    orig_w, orig_h = input_image.size

    # 2. Prevent 0 actual steps calculation in diffusers (steps * strength >= 1)
    actual_steps = int(steps * strength)
    if actual_steps < 1:
        steps = int(1.0 / max(strength, 0.05)) + 1

    # 3. Calculate aspect-ratio-preserving dimensions (max dimension 512, divisible by 8)
    scale = 512.0 / max(orig_w, orig_h)
    target_w = max(64, int(round((orig_w * scale) / 8.0)) * 8)
    target_h = max(64, int(round((orig_h * scale) / 8.0)) * 8)

    input_img = input_image.convert("RGB").resize((target_w, target_h), Image.LANCZOS)

    # 4. Handle seed generation
    if randomize_seed or seed is None:
        seed = random.randint(0, 2147483647)
    else:
        seed = int(seed)

    generator = torch.Generator(device="cuda").manual_seed(seed)

    pipe.to("cuda")
    output = pipe(
        prompt=prompt,
        negative_prompt="bad anatomy, extra fingers, watermark, blurred, low quality, distortion, noise",
        image=input_img,
        num_inference_steps=int(steps),
        guidance_scale=float(cfg),
        strength=float(strength),
        generator=generator,
    ).images[0]

    # 5. Resize output back to the original input aspect ratio & size
    output = output.resize((orig_w, orig_h), Image.LANCZOS)
    return output


# Clean & simple Gradio UI for Hugging Face Space
with gr.Blocks(title="Beyond Origami") as demo:
    gr.Markdown("# Beyond Origami\nUpload an image of your fold or pick a sample origami image below, then enter a prompt to reimagine your Origami.")

    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="pil", label="Upload or Pick Fold Image")
            prompt_in = gr.Textbox(
                label="Prompt",
                placeholder="e.g. bioluminescent crystal wings, masterpiece, highly detailed",
                lines=2
            )

            with gr.Accordion("⚙️ Quality & Advanced Settings", open=False):
                steps_slider = gr.Slider(
                    minimum=4, maximum=20, value=8, step=1,
                    label="Inference Steps (LCM)",
                    info="Higher values refine image details and overall quality (8-12 recommended)."
                )
                strength_slider = gr.Slider(
                    minimum=0.10, maximum=0.90, value=0.45, step=0.05,
                    label="Denoise Strength",
                    info="Higher values allow more creative freedom away from fold image; lower values preserve original fold shape strictly."
                )
                cfg_slider = gr.Slider(
                    minimum=1.0, maximum=4.0, value=1.5, step=0.1,
                    label="CFG Scale",
                    info="Higher values force the AI to follow your text prompt more strictly."
                )
                with gr.Row():
                    seed_number = gr.Number(
                        value=42, label="Seed", precision=0,
                        info="Seed number controlling output image variations. Same seed = similar output."
                    )
                    randomize_seed = gr.Checkbox(
                        value=True, label="Randomize Seed",
                        info="Generates a new random seed for every image generation."
                    )

            btn = gr.Button("✨ Reimagine Fold", variant="primary")

        with gr.Column():
            img_out = gr.Image(type="pil", label="Reimagined Origami")

    btn.click(
        fn=generate_image,
        inputs=[img_in, prompt_in, steps_slider, cfg_slider, strength_slider, seed_number, randomize_seed],
        outputs=img_out,
        api_name=False
    )

    gr.Examples(
        examples=[
            ["static/examples/butterfly.jpg", "bioluminescent crystal wings, masterpiece, highly detailed"],
            ["static/examples/spider.jpg", "cyberpunk robotic arachnid with glowing red neon lights"],
        ],
        inputs=[img_in, prompt_in],
        outputs=img_out,
        fn=generate_image,
        cache_examples=False,
        label="💡 Sample Origami Folds & Prompts (Click to try)"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
