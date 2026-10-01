import io
import base64
import torch
import spaces
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from diffusers import AutoPipelineForImage2Image, LCMScheduler
import gradio as gr

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


class PromptRequest(BaseModel):
    prompt: str
    image: str  # base64 string


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


@app.post("/get-image")
async def get_image(request: PromptRequest):
    try:
        # Decode base64 input image
        image_bytes = base64.b64decode(request.image)
        input_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_img = input_img.resize((512, 512))

        # Run inference on ZeroGPU
        result_img = generate_image_gpu(input_img, request.prompt)

        # Return image as PNG stream
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        print("Server error:", e)
        raise HTTPException(status_code=500, detail=str(e))


# Serve static files & index.html
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("index.html")


demo = gr.Blocks()
app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
