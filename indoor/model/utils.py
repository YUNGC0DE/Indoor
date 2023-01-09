import os
import random
from typing import Optional

import numpy as np
import torch
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN

from indoor.model.backbone import ResNet, Bottleneck
from indoor.config import num_classes, resize


def load_backbone() -> ResNet:
    """
    Load backbone
    :return: model instance
    """
    resnest50 = ResNet(Bottleneck, [3, 4, 6, 3], radix=2, groups=1,
                       bottleneck_width=64, stem_width=32, avg_down=True,
                       avd=True, avd_first=False)

    return resnest50


def load_full_model(model_weights: Optional[str] = None) -> FasterRCNN:
    backbone = load_backbone()
    return_layers = {
        "layer1": "0",
        "layer2": "1",
        "layer3": "2",
        "layer4": "3",
    }

    in_channels_list = [256, 512, 1024, 2048]
    out_channels = 256
    backbone = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)

    model = FasterRCNN(backbone, num_classes+1, min_size=resize, max_size=resize)

    if model_weights is not None:
        model_dict = torch.load(model_weights, map_location="cpu")["net_state"]
        model.load_state_dict(model_dict)

    return model


def fix_seeds(random_state: int = 17) -> None:
    """
    FIX model random sid
    :param random_state:
    :return: None
    """
    random.seed(random_state)
    os.environ["PYTHONHASHSEED"] = str(random_state)
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_):
    # https://pytorch.org/docs/stable/data.html#data-loading-randomness
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
