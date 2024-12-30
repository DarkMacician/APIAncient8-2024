import io
import base64
from fastapi import HTTPException
from pydantic import BaseModel
from diffusers import DiffusionPipeline
import torch
from Text2Image.model import model_path, app

print("Loading model from local path...")
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)

pipe.to("cuda")

class GenerateRequest(BaseModel):
    prompt: str = "An astronaut riding a green horse"  # Default prompt

@app.post("/generate")
async def generate_image(request: GenerateRequest):
    prompt = request.prompt

    try:
        # Generate the image
        image = pipe(prompt=prompt).images[0]

        # Save the image to an in-memory buffer
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        # Encode the image in Base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Return the Base64 string in the response
        return {"message": "Image generated successfully", "image_base64": image_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")