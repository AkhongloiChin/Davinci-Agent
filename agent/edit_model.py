from diffusers import StableDiffusionInstructPix2PixPipeline
import torch
import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url
from PIL import Image
import requests
from io import BytesIO
from dotenv import load_dotenv
import os
load_dotenv()


def edit_image(cloudinary_url, prompt):
    response = requests.get(cloudinary_url)
    if response.status_code == 200 and "image" in response.headers["Content-Type"]:
        input_image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        raise Exception("Failed to download image from Cloudinary.")

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        #torch_dtype=torch.float16,
        safety_checker=None
    )#.to("cuda")

    output = pipe(prompt, image=input_image)
    generated_image = output.images[0]  

    buffer = BytesIO()
    generated_image.save(buffer, format="PNG")
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

    print("Image uploaded successfully!")
    return upload_result['secure_url']
