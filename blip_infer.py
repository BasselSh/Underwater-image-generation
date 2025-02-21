from lavis.models import load_model_and_preprocess
from PIL import Image
import argparse
import os
parser = argparse.ArgumentParser(description="Process an image file.")
    
# Add the argument for the image path
parser.add_argument('--image_path', type=str, default='tmp.jpg', help='Path to the image file to be processed')

# Parse the arguments
args = parser.parse_args()
device='cuda'
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

image_in_path = args.image_path
image_in = Image.open(image_in_path)
image_pred_embed = vis_processors["eval"](image_in).unsqueeze(0).to(device)
caption = model.generate({"image": image_pred_embed})[0]
print(caption)