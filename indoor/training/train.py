import os
import argparse

import torch
from timm.optim import Lookahead
from torch.optim import RAdam
from clearml import Task

from indoor.config import model_dir
from indoor.data.scheduler import CustomScheduler
from indoor.data.utils import create_loaders
from indoor.model.utils import load_full_model, fix_seeds
from indoor.training.train_loop import train_net


def create_parser():
    parser = argparse.ArgumentParser(description="Train Network")

    # Arguments for stats and models saving
    parser.add_argument("--project_name", type=str, default="indoor", help="project name for ClearML server")
    parser.add_argument("--task_name", type=str, help="experiment name for ClearML server")
    parser.add_argument(
        "--output_uri",
        type=str,
        default=None,
        help="AWS path for saving models",
    )

    # Arguments for data loading and preparation
    parser.add_argument(
        "--train_file",
        type=str,
        help="path to a train set markup file",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        help="path to a val set markup file",
    )
    parser.add_argument("--random_state", type=int, default=17, help="random state for random generators")

    # training params
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help="number of gradient accumulation steps",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="num workers for data loader")
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--log_freq", type=int, default=50, help="frequency of logging training loss")
    parser.add_argument("--lr", type=float, default=0.0005, help="optimizer learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="L2 regularization")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD optimizer")
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="patience for RediceLROnPlateau LR-scheduler",
    )
    parser.add_argument(
        "--gamma_factor",
        type=float,
        default=0.5,
        help="gamma factor for RediceLROnPlateau LR-scheduler",
    )

    return parser


def train(args):
    task = Task.init(
        project_name=args.project_name,
        task_name=args.task_name,
        output_uri=None,
    )
    task.set_initial_iteration(0)
    fix_seeds(args.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = create_loaders(args)
    net = load_full_model()
    net.to(device)

    model_path = os.path.join(model_dir, args.task_name)
    os.makedirs(model_path, exist_ok=True)

    base_optimizer = RAdam(net.parameters(), lr=args.lr)
    optimizer = Lookahead(base_optimizer)
    lr_scheduler = CustomScheduler(optimizer, mode="max", factor=args.gamma_factor, patience=args.patience)
    logger = task.get_logger().current_logger()
    train_net(net, train_loader, val_loader, optimizer, lr_scheduler, device, logger, model_path, args)


if __name__ == "__main__":
    args = create_parser().parse_args()
    train(args)
