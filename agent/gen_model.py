from diffusers import StableDiffusionPipeline
import torch
from peft import PeftModel  
from io import BytesIO
import cloudinary
import cloudinary.uploader
from PIL import Image
from dotenv import load_dotenv
import os
load_dotenv()

def gen_image(prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Load base model ===
    base_model = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        #torch_dtype=torch.float16
    ).to(device)

    # === Load LoRA weights ===
    lora_path = "../lora/lora_sd_output/pytorch_lora_weights.safetensors"
    base_model.load_lora_weights(lora_path)

    # === Generate image ===
    image = base_model(prompt).images[0]  

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)


    cloudinary.config( 
        cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"), 
        api_key = os.getenv("CLOUDINARY_API_KEY"), 
        api_secret = os.getenv("CLOUDINARY_API_SECRET"), 
        secure=True
    )

    upload_result = cloudinary.uploader.upload(
        buffer,
        folder="lora_outputs",       
        format="png"
    )

    print("Image uploaded to Cloudinary!")
    return upload_result['secure_url']

