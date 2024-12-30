from fastapi import FastAPI
from diffusers import DiffusionPipeline
import torch
import os

app = FastAPI()

# Define model path
model_path = "/content/drive/MyDrive/API/stable-diffusion-xl-base-1.0"

# Load or download the model
if not os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
    print("Model not found locally. Downloading...")
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    pipe.save_pretrained(model_path)