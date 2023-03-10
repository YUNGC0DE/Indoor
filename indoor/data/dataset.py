import json

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

from indoor.config import label_mapping


class IndoorDataset(Dataset):
    def __init__(self, split, transform=None):
        self.split = pd.read_csv(split)
        self.transform = transform

    def __getitem__(self, idx):

        row = self.split.iloc[idx]
        image_path = row["image"]
        image = cv2.imread(image_path)

        annotation_path = row["annotation"]
        with open(annotation_path, 'r') as ann_bin:
            annotations = json.load(ann_bin)

        annotations = annotations.get("annotations")
        if annotations is None:
            target = {"boxes": torch.zeros((0, 4), dtype=torch.float32),
                      "labels": torch.zeros(0, dtype=torch.int64),
                      "area": torch.zeros(0, dtype=torch.int64),
                      "iscrowd": torch.zeros(0, dtype=torch.int64)}
        else:
            boxes = []
            labels = []

            for ann in annotations:
                box = ann["box"]
                label = ann["label"]
                boxes.append(box)
                labels.append(label_mapping[label])

            if self.transform is not None:
                try:
                    transformed = self.transform(image=image, bboxes=boxes, labels=labels)
                    image = transformed['image']
                    boxes = transformed['bboxes']
                    labels = transformed["labels"]
                except ValueError as _:
                    print("Skipping augmentation")

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            target = {"boxes": boxes, "labels": labels, "area": area, "iscrowd": iscrowd}

        image = torch.tensor(image) / 255
        image = image.permute(2, 0, 1)

        return image, target

    def __len__(self):
        return len(self.split)
