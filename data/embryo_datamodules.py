from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import pytorch_lightning as pl


class _EmbryoWrapper(Dataset):
    """ImageFolder 래퍼 → 4-튜플 반환"""

    def __init__(self, img_folder: datasets.ImageFolder,
                 num_slots: int = 8, spatial_size: int = 14):
        self.base = img_folder
        self.num_slots = num_slots
        self.spatial_size = spatial_size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        dummy_concept = torch.zeros(self.num_slots, dtype=torch.long)
        dummy_spatial = torch.zeros(self.num_slots,
                                    self.spatial_size,
                                    self.spatial_size)
        return img, dummy_concept, dummy_spatial, label


class EmbryoDatamodule(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int = 8,
                 data_dir: str = "~/embryo_vit224/",
                 num_workers: int = 4,
                 **kwargs):
        super().__init__()
        self.data_dir = Path(data_dir).expanduser()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # ViT-Tiny 224 기준 전처리
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    # Lightning hook
    def prepare_data(self):
        """외부 다운로드 없음"""
        pass

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", None):
            train_folder = datasets.ImageFolder(self.data_dir / "train",
                                                self.train_tf)
            val_folder = datasets.ImageFolder(self.data_dir / "val",
                                              self.test_tf)
            self.train_set = _EmbryoWrapper(train_folder)
            self.val_set = _EmbryoWrapper(val_folder)

        if stage in ("test", "predict", None):
            test_folder = datasets.ImageFolder(self.data_dir / "test",
                                               self.test_tf)
            self.test_set = _EmbryoWrapper(test_folder)

        self.num_classes = 2  # 성공 / 실패

    # DataLoaders ----------------------------------------------------------------
    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    predict_dataloader = test_dataloader