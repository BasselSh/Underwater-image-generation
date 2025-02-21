from lavis.models import load_model_and_preprocess
from PIL import Image
import argparse
import os
import pandas as pd
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image file.")
    parser.add_argument('--data-path', type=str, default='underwater_images/train', help='Path to the image file to be processed')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

    image_in_path = args.data_path
    pairs = pd.DataFrame(columns=['file_name', 'text'])
    for img_name in os.listdir(image_in_path):
        image_in = Image.open(os.path.join(image_in_path, img_name))
        image_pred_embed = vis_processors["eval"](image_in).unsqueeze(0).to(device)
        caption = model.generate({"image": image_pred_embed})[0]
        pairs.loc[len(pairs)] = [img_name, caption]
    pairs.to_csv(os.path.join(image_in_path, 'underwater_captions.csv'), index=False)