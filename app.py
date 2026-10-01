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

# Load LCM LoRA for fast 2-4 step inference
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")


# HF ZeroGPU function allocation
@spaces.GPU
def generate_image(input_image: Image.Image, prompt: str):
    if input_image is None or not prompt or not prompt.strip():
        return None

    input_img = input_image.convert("RGB").resize((512, 512))
    pipe.to("cuda")
    output = pipe(
        prompt=prompt,
        negative_prompt="bad anatomy, extra fingers, watermark, (worst quality, low quality:1.4)",
        image=input_img,
        num_inference_steps=4,
        guidance_scale=1.5,
        strength=0.4,  # Denoise strength matching original workflow
    ).images[0]
    return output


# Clean & simple Gradio UI for Hugging Face Space
with gr.Blocks(title="Beyond Origami") as demo:
    gr.Markdown("# 🎨 Beyond Origami\nUpload an image of your fold and enter a prompt to reimagine your Origami.")

    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="pil", label="Upload Fold Image")
            prompt_in = gr.Textbox(
                label="Prompt",
                placeholder="e.g. bioluminescent crystal wings, masterpiece, highly detailed",
                lines=2
            )
            btn = gr.Button("✨ Reimagine Fold", variant="primary")

        with gr.Column():
            img_out = gr.Image(type="pil", label="Reimagined Origami")

    btn.click(
        fn=generate_image,
        inputs=[img_in, prompt_in],
        outputs=img_out,
    )

if __name__ == "__main__":
    demo.launch()
