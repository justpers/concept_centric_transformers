from __future__ import annotations
import argparse, os, glob, warnings, sys
from dataclasses import dataclass
from typing import List, Tuple, Literal, Dict, Type

warnings.filterwarnings("ignore")
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
pl.seed_everything(42, workers=True)
from ctc.swin import SlotCSwinQSA, SlotCSwinISA, SlotCSwinSA
from ctc.vit  import SlotCVITQSA, SlotCVITISA, SlotCVITSA
import torchmetrics
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import math

# 1. Dataset & DataModule
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

# 모델 설정
@dataclass
class CCTCfg:
    backbone: Literal["swin_tiny", "swin_large", "vit_tiny", "vit_base"]
    slot_type: Literal["qsa", "isa", "sa"] = "qsa"
    model_name: str | None = None
    num_classes: int = 2
    n_unsup: int = 4
    n_concepts: int = 0
    n_spatial: int = 4
    pretrained: bool = True
    attn_drop: float = 0.1
    proj_drop: float = 0.1

_SWIN = {"qsa": SlotCSwinQSA, "isa": SlotCSwinISA, "sa": SlotCSwinSA}
_VIT  = {"qsa": SlotCVITQSA,  "isa": SlotCVITISA,  "sa": SlotCVITSA}

def build_cct(cfg: CCTCfg) -> nn.Module:
    if cfg.backbone.startswith("swin"):
        cls = _SWIN[cfg.slot_type]
        default = {
            "swin_tiny":  "swin_tiny_patch4_window7_224",
            "swin_large": "swin_large_patch4_window7_224.ms_in22k",
        }[cfg.backbone]
    else:
        cls = _VIT[cfg.slot_type]
        default = {
            "vit_tiny": "vit_tiny_patch16_224",
            "vit_base": "vit_base_patch16_224",
        }[cfg.backbone]
    name = cfg.model_name or default
    return cls(
        model_name=name,
        num_classes=cfg.num_classes,
        n_unsup_concepts=cfg.n_unsup,
        n_concepts=cfg.n_concepts,
        n_spatial_concepts=cfg.n_spatial,
        pretrained=cfg.pretrained,
        attention_dropout=cfg.attn_drop,
        projection_dropout=cfg.proj_drop,
    )

def _unsup_slot_losses(attn, lambda_sparse, lambda_div):
    if attn is None or (lambda_sparse == 0 and lambda_div == 0):
        z = attn.new_tensor(0.) if attn is not None else torch.tensor(0.)
        return z, z, z

    # attn -> [B,C,N], 토큰축 정규화
    if attn.size(1) not in (2,3,4,6,8,12,16):  # 대략 체크
        attn = attn.transpose(1,2)
    B, C, N = attn.shape
    eps = 1e-12
    p = attn.float().clamp_min(eps)
    p = p / (p.sum(-1, keepdim=True) + eps)            # [B,C,N]

    # Sparsity: 엔트로피(0=스파스, 1=균일)
    entropy = -(p * (p + eps).log()).sum(-1)           # [B,C]
    sparsity = (entropy / math.log(N)).mean()          # scalar ∈[0,1]

    # Diversity: off-diag만 평균 제곱
    G = (p @ p.transpose(1,2)) / float(N)              # [B,C,C]
    off = G - torch.diag_embed(G.diagonal(dim1=1, dim2=2))
    diversity = (off**2).sum(dim=(1,2)) / (C*(C-1))    # scalar

    return lambda_sparse*sparsity, lambda_div*diversity, sparsity

class LitCCT(pl.LightningModule):
    def __init__(self, cfg: CCTCfg, freeze_stages=0, warmup_epochs=3, lr: float = 1e-4, weight_decay: float = 0.05,
                 lambda_sparse: float = 1.0, lambda_div: float = 1.0,
                 n_unsup: int = 4):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.model = build_cct(cfg)
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

        if freeze_stages:
            self._freeze_encoder(freeze_stages)

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_acc   = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    def _slot_weight(self) -> float:
        # warmup_epochs 만큼 선형 램프업, 이후 코사인 감쇠로 min_w까지
        T1 = int(getattr(self.hparams, "warmup_epochs", 0) or 0)
        min_w = float(getattr(self.hparams, "slot_min_weight", 0.3))  # 필요시 CLI 인자로 노출
        T = int(self.trainer.max_epochs) if self.trainer is not None else (T1 + 1)

        if T1 <= 0:
            return 1.0

        # 1) 램프업: (1/T1, 2/T1, ..., 1.0)
        if self.current_epoch < T1:
            return float(self.current_epoch + 1) / float(T1)

        # 2) 코사인 감쇠: 1.0 → min_w
        remain = max(1, T - T1)
        t = min(self.current_epoch - T1, remain - 1)
        denom = (remain - 1) if remain > 1 else 1
        cos = 0.5 * (1.0 + math.cos(math.pi * t / denom))
        return min_w + (1.0 - min_w) * cos

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        images, labels = batch
        logits, unsup_attn, *_ = self.model(images)

        # 1) CE 손실을 확실히 스칼라로
        ce_loss = self.criterion(logits, labels)
        if isinstance(ce_loss, torch.Tensor) and ce_loss.ndim > 0:
            ce_loss = ce_loss.mean()

        # 2) 슬롯 손실 (이미 스칼라일 확률이 높지만 안전장치)
        sp_loss, div_loss, _ = _unsup_slot_losses(
            unsup_attn, self.hparams.lambda_sparse, self.hparams.lambda_div
        )
        if isinstance(sp_loss, torch.Tensor) and sp_loss.ndim > 0:
            sp_loss = sp_loss.mean()
        if isinstance(div_loss, torch.Tensor) and div_loss.ndim > 0:
            div_loss = div_loss.mean()

        # 3) 웜업/감쇠 스케일(이미 넣어두신 로직/함수 사용)
        scale = self._slot_weight() if stage == "train" else 1.0
        loss = ce_loss + scale * (sp_loss + div_loss)

        # 최종 loss도 스칼라 보장
        if isinstance(loss, torch.Tensor) and loss.ndim > 0:
            loss = loss.mean()

        preds = logits.argmax(dim=1)
        acc = self.train_acc(preds, labels) if stage == "train" else self.val_acc(preds, labels)

        bs = images.size(0)  # 배치 크기 로그 집계에 명시
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=bs)
        self.log(f"{stage}_acc",  acc,  prog_bar=True, batch_size=bs)
        self.log(f"{stage}_sp_loss",  sp_loss, prog_bar=True, batch_size=bs)
        self.log(f"{stage}_div_loss", div_loss, prog_bar=True, batch_size=bs)

        if stage == "train":
            self.log("lambda_scale", scale, prog_bar=True, batch_size=bs)
            self.log("train_sp_loss_scaled", sp_loss * scale, prog_bar=False, batch_size=bs)
            self.log("train_div_loss_scaled", div_loss * scale, prog_bar=False, batch_size=bs)
            self.log("loss_ce", ce_loss, prog_bar=False, batch_size=bs)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")
    
    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        max_ep = self.trainer.max_epochs if self.trainer else 50
        sched = {
            "scheduler": LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.hparams.warmup_epochs,
                max_epochs=max_ep,
            ),
            "interval": "epoch",
            "frequency":1,
        }
        return {"optimizer":optimizer, "lr_scheduler":sched}

    def _freeze_encoder(self, k:int):
        fe = self.model.feature_extractor
        if k >= 1:
            fe.patch_embed.requires_grad_(False)  # swin일 때 -> patch, ViT일 때 blocks
        for i in range(1, k):
            fe.layers[i-1].requires_grad_(False)
        print(f"[Info] froze forse {k} stage")

# 인자 설정

def parse_args():
    p = argparse.ArgumentParser(description="Embryo-CCT unsupervised concept fine‑tune")
    p.add_argument("--root", type=str, default="./embryo", help="Dataset root path")
    p.add_argument("--backbone", choices=["swin_tiny","swin_large","vit_tiny","vit_base"], default="swin_tiny")
    p.add_argument("--slot_type", choices=["qsa","isa","sa"], default="qsa")
    p.add_argument("--model_name")
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
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--freeze_stages", type=int, default=1)
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

    dm = EmbryoDataModule(root=args.root, 
                          img_size=args.img_size, 
                          batch_size=args.batch_size, 
                          num_workers=args.num_workers)
    cfg = CCTCfg(
        backbone=args.backbone,
        slot_type=args.slot_type,
        model_name = args.model_name,
        n_unsup = args.n_unsup,
    )
    model = LitCCT(cfg,
                   lr=args.lr, 
                   weight_decay=args.weight_decay, 
                   lambda_sparse=args.lambda_sparse,
                   lambda_div=args.lambda_div,
                   freeze_stages=args.freeze_stages,
                   warmup_epochs=args.warmup_epochs,
                )
    
    #model.model.register_forward_hook(debug_hook)
    if args.check_attn:
        quick_check(model, dm)
        return
    
    ckpt = ModelCheckpoint(monitor="val_acc", mode="max", save_last=True, save_top_k=1)
    es = EarlyStopping(monitor="val_acc", mode="max", patience=8)
    
    trainer = pl.Trainer(max_epochs=args.epochs, devices=args.gpus, accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         precision=16, log_every_n_steps=20,
                         default_root_dir="./checkpoints",
                         deterministic=True,
                         callbacks=[LearningRateMonitor(), ckpt, es]
                         )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path=ckpt.best_model_path)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()
    sys.exit()