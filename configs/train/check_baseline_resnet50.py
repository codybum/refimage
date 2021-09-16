# Basic training configuration
import os
from functools import partial

import albumentations as A
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from albumentations.pytorch import ToTensorV2 as ToTensor
from dataflow.dataloaders import get_train_val_loaders
from dataflow.transforms import denormalize
from torchvision import datasets
from torchvision.models.resnet import resnet50

import ignite.distributed as idist

from configs.train.image_dataloader import

def get_train_test_datasets(path, train_transforms, val_transform):

    train_ds = datasets.ImageFolder(os.path.join(path, 'train'),train_transforms)
    test_ds = datasets.ImageFolder(os.path.join(path, 'val'), val_transforms)

    return train_ds, test_ds


# ##############################
# Global configs
# ##############################

seed = 19
device = "cuda"
debug = False

# config to measure time passed to prepare batches and report measured time before the training
benchmark_dataflow = True
benchmark_dataflow_num_iters = 100

fp16_opt_level = "O2"
val_interval = 2
start_by_validation = True

train_crop_size = 224
val_crop_size = 320

batch_size = 64 * idist.get_world_size()  # total batch size
num_workers = 10


# ##############################
# Setup Dataflow
# ##############################

assert "DATASET_PATH" in os.environ
data_path = os.environ["DATASET_PATH"]

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = A.Compose(
    [
        A.Resize(224),
        A.Normalize(mean=mean, std=std),
        ToTensor(),
    ]
)

val_transforms = A.Compose(
    [
        A.Resize(224),
        A.Normalize(mean=mean, std=std),
        ToTensor(),
    ]
)

train_loader, val_loader, train_eval_loader = get_train_val_loaders(
    data_path,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    batch_size=batch_size,
    num_workers=num_workers,
    val_batch_size=batch_size,
)

# Image denormalization function to plot predictions with images
img_denormalize = partial(denormalize, mean=mean, std=std)

# ##############################
# Setup Model
# ##############################

model = resnet50(pretrained=False)


# ##############################
# Setup Solver
# ##############################

num_epochs = 1

criterion = nn.CrossEntropyLoss()

le = len(train_loader)

base_lr = 0.1 * (batch_size / 256.0)
optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
lr_scheduler = lrs.MultiStepLR(optimizer, milestones=[30 * le, 60 * le, 90 * le, 100 * le], gamma=0.1)
