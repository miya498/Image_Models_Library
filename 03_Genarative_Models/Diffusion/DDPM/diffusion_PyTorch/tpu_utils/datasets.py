import os
import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


def pack(image, label):
    """
    PyTorchでのデータ形式に変換
    """
    label = torch.tensor(label, dtype=torch.int32)
    return {"image": image, "label": label}


class SimpleDataset(Dataset):
    DATASET_NAMES = ("cifar10", "celebahq256")

    def __init__(self, name, data_dir):
        assert name in self.DATASET_NAMES, f"Unknown dataset: {name}"
        self.name = name
        self.data_dir = data_dir
        self.img_size = {"cifar10": 32, "celebahq256": 256}[name]
        self.img_shape = (self.img_size, self.img_size, 3)
        self.num_train_examples, self.num_eval_examples = {
            "cifar10": (50000, 10000),
            "celebahq256": (30000, 0),
        }[name]
        self.eval_split_name = {"cifar10": "test", "celebahq256": None}[name]
        self.num_classes = 1  # unconditional

        # Transform for images
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255),  # Scale to [0, 255]
        ])

        # Load dataset
        if name == "cifar10":
            self.dataset = datasets.CIFAR10(
                root=self.data_dir, train=True, download=True, transform=self.transform
            )
        elif name == "celebahq256":
            # Add CelebA-HQ loading logic here if needed
            raise NotImplementedError("CelebA-HQ dataset loading is not implemented.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return pack(image, label)


class LsunDataset(Dataset):
    def __init__(self, tfr_file, resolution=256, max_images=None):
        """
        TFRecord形式で保存されたLSUNデータセットをPyTorch用に変換
        """
        self.tfr_file = tfr_file
        self.resolution = resolution
        self.max_images = max_images
        self.image_shape = (resolution, resolution, 3)
        self.images = self._load_tfrecord()

    def _load_tfrecord(self):
        """
        TFRecordから画像を読み込む（簡易的に実装）
        """
        # ここではTFRecordの処理を省略しています。
        # 実際に実装する場合、`tfrecord`ライブラリなどを使用して画像を読み込む必要があります。
        raise NotImplementedError("TFRecord loading is not implemented.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.max_images is not None and index >= self.max_images:
            raise IndexError
        image = self.images[index]
        label = 0  # Unconditional dataset
        return pack(image, label)


DATASETS = {
    "cifar10": partial(SimpleDataset, name="cifar10"),
    "celebahq256": partial(SimpleDataset, name="celebahq256"),
    "lsun": LsunDataset,
}


def get_dataset(name, data_dir=None, tfr_file=None):
    """
    指定された名前のデータセットを取得
    """
    if name not in DATASETS:
        raise ValueError(f"Dataset {name} is not available.")

    if name == "lsun":
        assert tfr_file is not None, "LSUN dataset requires a TFRecord file."
        return DATASETS[name](tfr_file=tfr_file)
    else:
        assert data_dir is not None, "Simple datasets require a data directory."
        return DATASETS[name](data_dir=data_dir)


def get_dataloader(dataset, batch_size, shuffle=True, num_workers=4):
    """
    DataLoaderを作成
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
