import torch
from PIL import Image
from diffusers import DiffusionPipeline
import os
import subprocess
from sentence_transformers import SentenceTransformer
import os

# Expand the tilde to the full path
conda_env_python = os.path.expanduser('~/anaconda3/envs/lavis/bin/python')

# 1. Load a pretrained Sentence Transformer model
cos_model = SentenceTransformer("all-MiniLM-L6-v2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipeline_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
tmpdir = 'sd-model-finetuned-lora/checkpoint-6000'
diffuser = DiffusionPipeline.from_pretrained(pipeline_path)
diffuser.load_lora_weights(tmpdir)
diffuser.to(device)

with open('imagenet_classes.txt', 'r') as f:
    lines = f.read().split('\n')

for i, line in enumerate(lines):
    if i < 15: continue
    category = line.split(',')[0]
    os.makedirs(f'generated_images/{category}', exist_ok=True)
    os.makedirs(f'generated_images/{category}/correct', exist_ok=True)
    os.makedirs(f'generated_images/{category}/incorrect', exist_ok=True)
    prompt = f'An underwater view of {category}'
    for i in range(1):
        for attempt in range(4):
            image_pred = diffuser(prompt, num_inference_steps=30).images[0] #, generator = torch.Generator(device='cuda')
            image_pred.save('tmp.jpg')
            caption = subprocess.run([conda_env_python, '/home/human/bassel/underwater/diffusers/blip_infer.py'], capture_output=True, text=True).stdout[:-1]
            embeddings = cos_model.encode([prompt, caption])
            similarity = cos_model.similarity(embeddings[0], embeddings[1]).squeeze().item()
            print(f"Prompt: {prompt}. Caption: {caption}. Sim: {similarity}")
            if similarity > 0.7:
                image_pred.save(f'generated_images/{category}/correct/{category}_{i}_attempt{attempt}.jpg')
                break
            else:
                image_pred.save(f'generated_images/{category}/incorrect/{category}_{i}_attempt{attempt}.jpg')
    

