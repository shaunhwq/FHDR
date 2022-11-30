import os
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import torchvision.transforms as transforms
from model import FHDR
import torchvision.transforms.functional as tvf

import util


def get_model_input(image: np.array, device: str) -> torch.tensor:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = image[:256, :256, :]
    image = image.astype(np.float32) / 255.
    image = torch.from_numpy(np.ascontiguousarray(np.transpose(image, (2, 0, 1)))).float()
    image = tvf.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    image = torch.stack([image])
    image = image.to(device)
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--iter_num", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, f"FHDR-iter-{args.iter_num}.ckpt")
    assert os.path.exists(checkpoint_path), "Unable to find pretrained weights"

    model = FHDR(iteration_count=args.iter_num)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(args.device)
    model.eval()

    image_paths = [os.path.join(args.input_dir, img_path) for img_path in os.listdir(args.input_dir) if img_path[0] != "."]

    with torch.no_grad():
        for img_path in tqdm(image_paths, total=len(image_paths), desc=f"Running FHDR [iter-{args.iter_num}]..."):
            in_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            model_input = get_model_input(in_image, args.device)

            output = model(model_input)
            output = output[-1]

            new_name = os.path.splitext(os.path.basename(img_path))[0] + ".hdr"
            output_path = os.path.join(args.output_dir, new_name)
            # https://github.com/mukulkhanna/FHDR/issues/7
            util.save_hdr_image(img_tensor=output, batch=0, path=output_path)
