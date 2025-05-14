import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline

# Init models
device = "cuda" if torch.cuda.is_available() else "cpu"

gen_pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
).to(device)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

with open('image_prompts.txt', 'r') as f:
    PROMPTS = [line.strip() for line in f.readlines() if line.strip()]

IMAGES_PER_PROMPT = 3
SAVE_DIR = "clip_feedback_dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

records = []

def compute_clip_score(prompt: str, image: Image.Image):
    inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    similarity = torch.cosine_similarity(outputs.image_embeds, outputs.text_embeds)
    return similarity.item()

# Main loop
for prompt in tqdm(PROMPTS, desc="Generating dataset"):
    for i in range(IMAGES_PER_PROMPT):
        result = gen_pipe(prompt)
        image = result.images[0]

        clip_score = compute_clip_score(prompt, image)

        img_name = f"{prompt[:30].replace(' ', '_')}_{i}.png"
        img_path = os.path.join(SAVE_DIR, img_name)
        image.save(img_path)

        records.append({
            "prompt": prompt,
            "image_path": img_path,
            "clip_score": clip_score
        })

# Save CSV metadata
df = pd.DataFrame(records)
df.to_csv(os.path.join(SAVE_DIR, "metadata.csv"), index=False)
print(f"Saved dataset to: {SAVE_DIR}")
