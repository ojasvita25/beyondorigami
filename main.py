from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import io
import comfyuiservice
from pydantic import BaseModel

class PromptRequest(BaseModel):
    prompt: str
    image: str  # base64 string

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; restrict to specific domains as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/get-image")
async def get_image(request: PromptRequest):
    try:
        # Pass base64 image directly if your function handles base64
        result_bytes = comfyuiservice.fetch_image_from_comfy(request.prompt, request.image)

        return StreamingResponse(io.BytesIO(result_bytes), media_type="image/png")
    except Exception as e:
        print("Server error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/hello")
def read_hello():
    return {"message": "Hello World!"}