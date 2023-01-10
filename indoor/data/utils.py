from torch.utils.data import DataLoader
import albumentations as A
import cv2

from indoor.data.dataset import IndoorDataset
from indoor.model.utils import seed_worker


def collate(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


def create_loaders(args):

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(
            p=0.2,
            limit=10,
            interpolation=cv2.INTER_CUBIC,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
        ),
        A.RandomRotate90(p=0.24),
        #  pixel transforms
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.05, 0.05), p=0.1),
        A.Sharpen(alpha=(0.1, 0.5), lightness=(0.5, 1.0), p=0.1),
        A.OneOf(
            [
                A.Blur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
                A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.5), p=1.0),
            ],
            p=0.1,
        ),
        #  geometric transforms
        A.ElasticTransform(
            alpha=2,
            alpha_affine=16,
            sigma=16,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            interpolation=cv2.INTER_CUBIC,
            p=0.1,
        ),
        A.GridDistortion(
            num_steps=10,
            distort_limit=0.15,
            interpolation=cv2.INTER_CUBIC,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.1,
        ),
        A.OpticalDistortion(interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.1),
    ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.8, label_fields=["labels"]))

    train_dataset = IndoorDataset(args.train_file, transform)
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
