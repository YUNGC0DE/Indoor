from torch.utils.data import DataLoader

from indoor.data.dataset import IndoorDataset
from indoor.model.utils import seed_worker


def collate(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


def create_loaders(args):
    train_dataset = IndoorDataset(args.train_file)
    val_dataset = IndoorDataset(args.val_file)

    train_loader = DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate
    )

    val_loader = DataLoader(
        val_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        worker_init_fn=seed_worker,
        shuffle=False,
        collate_fn=collate
    )

    return train_loader, val_loader
