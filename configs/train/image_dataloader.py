import os
from typing import Callable, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision import datasets

import ignite.distributed as idist


def get_train_val_loaders(
    root_path: str,
    train_transforms: Callable,
    val_transforms: Callable,
    batch_size: int = 16,
    num_workers: int = 8,
    val_batch_size: Optional[int] = None,
    limit_train_num_samples: Optional[int] = None,
    limit_val_num_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_ds = datasets.ImageFolder(os.path.join(root_path,'train'), train_transforms)
    val_ds = datasets.ImageFolder(os.path.join(root_path,'val'), val_transforms)

    if limit_train_num_samples is not None:
        np.random.seed(limit_train_num_samples)
        train_indices = np.random.permutation(len(train_ds))[:limit_train_num_samples]
        train_ds = Subset(train_ds, train_indices)

    if limit_val_num_samples is not None:
        np.random.seed(limit_val_num_samples)
        val_indices = np.random.permutation(len(val_ds))[:limit_val_num_samples]
        val_ds = Subset(val_ds, val_indices)

    # random samples for evaluation on training dataset
    if len(val_ds) < len(train_ds):
        np.random.seed(len(val_ds))
        train_eval_indices = np.random.permutation(len(train_ds))[: len(val_ds)]
        train_eval_ds = Subset(train_ds, train_eval_indices)
    else:
        train_eval_ds = train_ds

    train_loader = idist.auto_dataloader(
        train_ds, shuffle=True, batch_size=batch_size, num_workers=num_workers, drop_last=True,
    )

    val_batch_size = batch_size * 4 if val_batch_size is None else val_batch_size
    val_loader = idist.auto_dataloader(
        val_ds, shuffle=False, batch_size=val_batch_size, num_workers=num_workers, drop_last=False,
    )

    train_eval_loader = idist.auto_dataloader(
        train_eval_ds, shuffle=False, batch_size=val_batch_size, num_workers=num_workers, drop_last=False,
    )

    return train_loader, val_loader, train_eval_loader