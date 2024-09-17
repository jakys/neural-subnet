import os
import io
import torch
import clip
import uvicorn
import time
import argparse
from pydantic import BaseModel
from PIL import Image 
import cv2
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, RasterizationSettings,
    HardPhongShader, PointLights, PerspectiveCameras
)
from pytorch3d.renderer import look_at_view_transform
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import FileResponse
from transformers import CLIPProcessor, CLIPModel
from models import ValidateRequest, ValidateResponse
from rendering import render, load_image

app = FastAPI()

# Load the cuda & CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
except Exception as e:
    print("load model error")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8094)
    args, extras = parser.parse_known_args()
    return args, extras

def resize_image(image, target_size=(256, 256)):
    """Resize an image to the target size."""
    return image.resize(target_size, Image.Resampling.LANCZOS)

def pil_to_cv(image):
    """Convert PIL Image to OpenCV format and resize."""
    image = resize_image(image)  # Resize image to ensure dimensions match
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def calculate_image_entropy(image):
    """Calculate the entropy of an image."""
    image = resize_image(image)  # Resize image to ensure dimensions match
    histogram = np.histogram(image, bins=256, range=[0, 256])[0]
    histogram_normalized = histogram / histogram.sum()
    histogram_normalized = histogram_normalized[histogram_normalized > 0]  # Remove zeros
    entropy = -np.sum(histogram_normalized * np.log2(histogram_normalized))
    return entropy

def compute_clip_similarity(image1, image2):
    global model
    with torch.no_grad():
        image1_features = model.encode_image(image1)
        image2_features = model.encode_image(image2)
        similarity = torch.nn.functional.cosine_similarity(image1_features, image2_features).item()
    return similarity

def compute_clip_similarity_prompt(text, image_path):
    global clip_model, processor
    print("successfully imported")
    # Preprocess the inputs
    image = Image.open(image_path)  # Change to your image path        
    image_inputs = processor(images=image, return_tensors="pt")
    text_inputs = processor(text=text, return_tensors="pt", truncation=True)
    
    # Get the embeddings
    image_embeddings = clip_model.get_image_features(**image_inputs)
    text_embeddings = clip_model.get_text_features(**text_inputs)
    
    # Normalize the embeddings to unit length
    image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    with torch.no_grad():
        similarity = torch.nn.functional.cosine_similarity(text_embeddings, image_embeddings).item()
    return similarity

args, _ = get_args()

# Set up the directory and file paths
DATA_DIR = './results'

device = "cuda" if torch.cuda.is_available() else "cpu"

@app.post("/validate/")
async def validate(data: ValidateRequest) -> ValidateResponse:
    prompt = data.prompt
    uid = data.uid
    start_time = time.time()
    try:
        rendered_images, before_images = await render(prompt_image=prompt, id=uid)
        print(f"render time: {time.time() - start_time}")
        preview_image_path = os.path.join(DATA_DIR, f"{uid}/preview.png")

        # Load all rendered images
        
        preview_image = load_image(preview_image_path)
        print(f"load model time: {time.time() - start_time}")

        # Function to compute similarity using CLIP
        
        
        S0 = compute_clip_similarity_prompt(prompt, preview_image_path)

        print(f"similarity: {S0}")
        
        if S0 < 0.25:
            return ValidateResponse(
                score=0
            )

        Si = [compute_clip_similarity(preview_image, img) for img in rendered_images]
        print(f"similarities: {Si}")
        
        print(f"S calc time: {time.time() - start_time}")
        
        
        if rendered_images and before_images:
            # Q0 = calculate_image_entropy(Image.open(preview_image_path))
            Qi = [calculate_image_entropy(Image.open(before_image)) for before_image in before_images]
        else:
            # Q0 = 0  # No comparison available, set to 0 or an appropriate value indicating no data
            Qi = []
            
        # print(f"Q0: {Q0}")
        print(f"Qi: {Qi}")
        
        print(f"Qi time: {time.time() - start_time}")
        

        S_geo = np.exp(np.log(Si).mean())
        Q_geo = np.exp(np.log(Qi).mean())

        # Total Similarity Score (Stotal)
        S_total = S0 * 0.3 + S_geo * 0.5 + Q_geo * 0.2

        print(S_total)

        return ValidateResponse(
            score=S_total
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Uncomment this section to run the FastAPI app locally
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=args.port)
