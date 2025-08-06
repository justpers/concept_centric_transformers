"""
Embryo‐dataset fine‑tuning script (classification + *unsupervised* SlotCVIT QSA).

* 개념 슬롯은 **비지도**로만 학습(n_unsup_concepts>0, n_concepts=n_spatial_concepts=0)
* 분류 헤드는 성공/실패(2‑class) 지도 학습(CrossEntropy)
* Loss = CE  +  λ_sparse·Sparsity  +  λ_div·Diversity ‑‑> _unsup_slot_losses 구현 참조

Tested with torch 2.3 / pytorch‑lightning 1.9; modify paths & hyper‑params as needed.
"""

from __future__ import annotations
import argparse, os, glob
from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import sys
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchmetrics
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
torch.set_float32_matmul_precision('high')

# ────────────────────────────────────────────────────────────
# 1. Dataset & DataModule
# ────────────────────────────────────────────────────────────
class EmbryoDataset(Dataset):
    """Folder‑based: <root>/<split>/{failure|success}/*.png|jpg"""
    def __init__(self, root: str, split: str, transform: A.BasicTransform):
        self.samples: List[Tuple[str, int]] = []
        self.transform = transform
        for label_name, label_idx in ("failure", 0), ("success", 1):
            dir_path = os.path.join(root, split, label_name)
            if not os.path.isdir(dir_path):
                continue
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
                self.samples += [(p, label_idx) for p in glob.glob(os.path.join(dir_path, ext))]
        if not self.samples:
            raise RuntimeError(f"No images found in {root}/{split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = A.read_rgb_image(path)  # albumentations util
        image = self.transform(image=image)["image"]
        return image, torch.tensor(label, dtype=torch.long)


def default_transforms(img_size: int = 224, is_train: bool = True):
    base = [
        A.LongestMaxSize(max_size=max(img_size, 256)),
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=0,
            value=0,                       # <─ 추가
        ),
    ]
    if is_train:
        aug = [
            A.RandomCrop(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05,
                               scale_limit=0.05,
                               rotate_limit=15, p=0.5),
        ]
    else:
        aug = [A.CenterCrop(img_size, img_size)]

    tail = [
        A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2(),
    ]
    return A.Compose(base + aug + tail)

class EmbryoDataModule(pl.LightningDataModule):
    def __init__(self, root: str, img_size: int = 224, batch_size: int = 16, num_workers: int = 4):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None):
        self.train_ds = EmbryoDataset(self.hparams.root, "train", default_transforms(self.hparams.img_size, True))
        self.val_ds   = EmbryoDataset(self.hparams.root, "val",   default_transforms(self.hparams.img_size, False))
        self.test_ds  = EmbryoDataset(self.hparams.root, "test",  default_transforms(self.hparams.img_size, False))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=True)


# ────────────────────────────────────────────────────────────
# 2. SlotCVIT ‑ LightningModule wrapper
# ────────────────────────────────────────────────────────────
from ctc.vit import SlotCVITQSA  # 원본 레포 코드 import


def _unsup_slot_losses(attn: torch.Tensor, lambda_sparse: float, lambda_div: float):
    """Re‑implementation (FP32) of sparsity & diversity losses used in QSA paper"""
    if attn is None or (lambda_sparse == 0 and lambda_div == 0):
        zero = torch.tensor(0., device=attn.device if attn is not None else "cpu")
        return zero, zero, zero

    # attn: [B, C, N] (C = n_unsup_concepts)
    if attn.size(1) == 1 and attn.size(2) > 1:  # safety transpose
        attn = attn.transpose(1, 2)

    p = attn.float().clamp_min(1e-8)  # [B,C,N]
    C = p.size(1)

    # (1) sparsity: 평균 슬롯 엔트로피 정규화
    entropy = -(p * p.log()).sum(-1).mean()          # H ∈ [0, ln C]
    sparsity = entropy / torch.log(torch.tensor(float(C), device=p.device))

    # (2) diversity: 슬롯 간 dot‑sim 평균
    avg_sim = torch.einsum("bcn,bcm->bnm", p, p).mean()  # rough similarity measure
    diversity = avg_sim / C

    loss_sparse = lambda_sparse * sparsity
    loss_div    = lambda_div * diversity
    return loss_sparse, loss_div, entropy


class LitSlotCVIT(pl.LightningModule):
    def __init__(self, lr: float = 1e-4, weight_decay: float = 0.05,
                 lambda_sparse: float = 1.0, lambda_div: float = 1.0,
                 n_unsup: int = 4):
        super().__init__()
        self.save_hyperparameters()

        self.model = SlotCVITQSA(
            model_name="vit_tiny_patch16_224",
            num_classes=2,
            n_unsup_concepts=n_unsup,
            n_concepts=0,
            n_spatial_concepts=4,
            num_heads=12,
            attention_dropout=0.1,
            projection_dropout=0.1,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_acc   = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    # ‑‑‑ Lightning hooks ‑‑‑
    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        images, labels = batch
        logits, unsup_attn, *_ = self.model(images) 

        ce_loss = self.criterion(logits, labels)
        sp_loss, div_loss, _ = _unsup_slot_losses(
            unsup_attn, self.hparams.lambda_sparse, self.hparams.lambda_div
        )
        loss = ce_loss + sp_loss + div_loss

        preds = logits.argmax(dim=1)
        acc = self.train_acc(preds, labels) if stage == "train" else self.val_acc(preds, labels)

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        self.log(f"{stage}_sp_loss", sp_loss, prog_bar=True)
        self.log(f"{stage}_div_loss", div_loss, prog_bar=True)

        if stage == "train":
            self.log("loss_ce", ce_loss, prog_bar=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")
    
    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [scheduler]

# ────────────────────────────────────────────────────────────
# 3. CLI
# ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Embryo SlotCVIT unsupervised concept fine‑tune")
    p.add_argument("--root", type=str, default="./transfer", help="Dataset root path")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--n_unsup", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--lambda_sparse", type=float, default=1.0)
    p.add_argument("--lambda_div", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--check_attn", action="store_true", help="CLS 어텐션 한 번만 찍고 종료")
    return p.parse_args()

def debug_hook(module, inputs, outputs):   # <─ 확인용 함수
    _, attn, *_ = outputs
    print("★ unsup_attn shape:", attn.shape)
    module._forward_hooks.clear()  

def quick_check(model, dm):              # <─ 확인용 함수
    args = parse_args()
    dm.setup("val")
    imgs, _ = next(iter(dm.val_dataloader()))
    imgs = imgs.to(model.device)
    model.eval()
    with torch.no_grad():
        _, attn, *_ = model.model(imgs[:1])   # (1, C, N) or (1, N, C)
    if attn.size(1) != args.n_unsup:
        attn = attn.transpose(1, 2)
    print("★ attn distrib:", attn.squeeze().cpu().numpy())

def main():
    args = parse_args()

    dm = EmbryoDataModule(root=args.root, img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers)
    model = LitSlotCVIT(lr=args.lr, weight_decay=args.weight_decay, lambda_sparse=args.lambda_sparse,
                        lambda_div=args.lambda_div, n_unsup=args.n_unsup)
    
    #model.model.register_forward_hook(debug_hook)
    if args.check_attn:
        quick_check(model, dm)
        return
    
    trainer = pl.Trainer(max_epochs=args.epochs, devices=args.gpus, accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         precision=16, log_every_n_steps=20,
                         default_root_dir="./checkpoints",
                         )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    main()
    sys.exit()