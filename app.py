import io
import base64
import torch
import spaces
from PIL import Image
from fastapi import Request, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
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
def generate_image_gpu(input_image: Image.Image, prompt: str):
    pipe.to("cuda")
    output = pipe(
        prompt=prompt,
        negative_prompt="bad anatomy, extra fingers, watermark, (worst quality, low quality:1.4)",
        image=input_image,
        num_inference_steps=4,
        guidance_scale=1.5,
        strength=0.4,  # Denoise strength matching original workflow
    ).images[0]
    return output


# Create Gradio Blocks UI (compatible with ZeroGPU startup detection & Gradio 5 schema)
with gr.Blocks(title="Beyond Origami") as demo:
    gr.Markdown("# Beyond Origami\nOrigami x AI reimaginings powered by SD 1.5 LCM on Free ZeroGPU.")
    with gr.Row():
        img_in = gr.Image(type="pil", label="Upload Fold")
        prompt_in = gr.Textbox(label="Prompt", placeholder="Describe how to reimagine your fold...")
    btn = gr.Button("Reimagine", variant="primary")
    img_out = gr.Image(type="pil", label="Reimagined Origami")

    btn.click(
        fn=generate_image_gpu,
        inputs=[img_in, prompt_in],
        outputs=img_out,
        api_name="generate"
    )

# Enable CORS on demo.app
demo.app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@demo.app.post("/get-image")
async def get_image(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        image_str = data.get("image", "")

        # Decode base64 input image
        image_bytes = base64.b64decode(image_str)
        input_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_img = input_img.resize((512, 512))

        # Run inference on ZeroGPU
        result_img = generate_image_gpu(input_img, prompt)

        # Return image as PNG stream
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        print("Server error:", e)
        raise HTTPException(status_code=500, detail=str(e))


demo.app.mount("/static", StaticFiles(directory="static"), name="static")

@demo.app.get("/")
def read_root():
    return FileResponse("index.html")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
