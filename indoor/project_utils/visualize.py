import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms
from tqdm import tqdm
import cv2
import numpy as np

from indoor.config import color_mapping, id_mapping
from indoor.data.dataset import IndoorDataset
from indoor.data.utils import collate
from indoor.model.utils import load_full_model


def create_parser():
    parser = argparse.ArgumentParser(description="Train Network")

    # Arguments for stats and models saving
    parser.add_argument("--split_path", type=str, help="Path to split to visualize")
    parser.add_argument("--model_weights", type=str, help="Path to model weights")
    parser.add_argument("--save_path", type=str, default="visualizations", help="Folder to save images")
    return parser


def draw_rectangles(image: np.ndarray, annotation: dict) -> None:
    for i in range(len(annotation["boxes"])):
        box = np.array(annotation["boxes"][i]).astype(np.uint16)
        label = annotation["labels"][i].item()
        label = id_mapping[label]
        color = color_mapping[label]
        score = round(annotation['scores'][i].item(), 2)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 3)
        cv2.putText(image, f"{label}: {score}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def visualize(args):
    """
    Saves visualization of predictions
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dataset = IndoorDataset(args.split_path)
    val_loader = DataLoader(
        val_dataset,
        num_workers=0,
        batch_size=1,
        collate_fn=collate,
    )
    net = load_full_model(model_weights=args.model_weights)
    net.to(device)
    net.eval()

    os.makedirs(args.save_path, exist_ok=True)

    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
            images, targets = data
            image_to_draw = images[0].permute(1, 2, 0).numpy() * 255
            images = list(image.to(device) for image in images)
            predict = net(images)[0]
            predict = {k: v.cpu() for k, v in predict.items()}
            ids = nms(predict["boxes"], predict["scores"], 0.5)
            predict["boxes"] = predict["boxes"][ids]
            predict["labels"] = predict["labels"][ids]
            predict["scores"] = predict["scores"][ids]
            # Using hardcoded thr for all classes just to visualize. Better find specific thr for each class
            above_thr_idx = torch.where(predict["scores"] > 0.65)
            predict["boxes"] = predict["boxes"][above_thr_idx]
            predict["labels"] = predict["labels"][above_thr_idx]
            predict["scores"] = predict["scores"][above_thr_idx]
            draw_rectangles(image_to_draw, predict)
            cv2.imwrite(f"{args.save_path}/{i}.png", image_to_draw)


if __name__ == "__main__":
    args = create_parser().parse_args()
    visualize(args)
