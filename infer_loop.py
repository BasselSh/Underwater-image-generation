import torch
from PIL import Image
from diffusers import DiffusionPipeline
import os
import subprocess
from sentence_transformers import SentenceTransformer
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Process an image file.")
    parser.add_argument('prompt_file', type=str, help='Path to the prompt file to be processed')
    parser.add_argument('lavis_env', type=str, help='Path to the lavis environment')
    parser.add_argument('--sd-model', type=str, default='stable-diffusion-v1-5/stable-diffusion-v1-5', help='Path to the model file to be processed')
    parser.add_argument('--lora_path', type=str, default='sd-model-finetuned-lora/checkpoint-6000', help='Path to the lora file to be processed')
    parser.add_argument('--num_images', type=int, default=10, help='Number of images to be generated')
    parser.add_argument('--num_attempts', type=int, default=4, help='Number of attempts to generate an image')
    return parser.parse_args()

def inference_loop(classes, num_images, num_attempts, cos_model, diffuser, lavis_env):
     os.makedirs('generated_images', exist_ok=True)
     for class_idx, line in enumerate(classes):
        category = line.split(',')[0]
        os.makedirs(f'generated_images/{category}', exist_ok=True)
        os.makedirs(f'generated_images/{category}/correct', exist_ok=True)
        os.makedirs(f'generated_images/{category}/incorrect', exist_ok=True)
        prompt = f'An underwater view of {category}'
        for j in range(num_images):
            for attempt in range(num_attempts):
                image_pred = diffuser(prompt, num_inference_steps=30).images[0] #, generator = torch.Generator(device='cuda')
                image_pred.save('tmp.jpg')
                caption = subprocess.run([lavis_env, 'blip_infer.py'], capture_output=True, text=True).stdout[:-1]
                embeddings = cos_model.encode([prompt, caption])
                similarity = cos_model.similarity(embeddings[0], embeddings[1]).squeeze().item()
                print(f"Prompt: {prompt}. Caption: {caption}. Sim: {similarity}")
                if similarity > 0.7:
                    image_pred.save(f'generated_images/{category}/correct/{category}_{j}_attempt{attempt}.jpg')
                    break
                else:
                    image_pred.save(f'generated_images/{category}/incorrect/{category}_{j}_attempt{attempt}.jpg')

def main():
    args = parse_args()
    prompt_file = args.prompt_file
    num_images = args.num_images
    num_attempts = args.num_attempts
    sd_model = args.sd_model
    lora_dir = args.lora_path
    lavis_env = os.path.expanduser(args.lavis_env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cos_model = SentenceTransformer("all-MiniLM-L6-v2")
    diffuser = DiffusionPipeline.from_pretrained(sd_model)
    diffuser.load_lora_weights(lora_dir)
    diffuser.to(device)
    cos_model.to(device)
    with open(prompt_file, 'r') as f:
        lines = f.read().split('\n')

    inference_loop(lines, num_images, num_attempts, cos_model, diffuser, lavis_env)
        

if __name__ == "__main__":
    main()
