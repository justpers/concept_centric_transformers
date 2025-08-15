from __future__ import annotations
import argparse, os, importlib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix, roc_auc_score, cohen_kappa_score, classification_report
import numpy as np

# -----------------------------
# 파서
# -----------------------------
def build_parser():
    p = argparse.ArgumentParser("CCT confusion-matrix metrics (ACC/Prec/Rec/F1/Kappa + AUC)")

    # 체크포인트 / 모델 빌더
    p.add_argument("--checkpoint", required=True, type=str, help=".ckpt path (Lightning or state_dict)")
    p.add_argument("--model_entry", required=True, type=str,
                   help="모델 빌더 'pkg.mod:func' (예: ctc.cct_builder:build_model)")

    # 데이터/로더
    p.add_argument("--dataset_root", required=True, type=str, help="dataset root (train/val/test 하위)")
    p.add_argument("--split", type=str, default="val", help="평가 split (기본 val)")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--pos_label", type=int, default=1, help="이진 AUC 계산 시 양성 클래스 인덱스")

    # CCT 빌더용(학습과 동일하게 맞추면 state_dict 로딩이 용이)
    p.add_argument("--backbone", choices=["swin_tiny","swin_large","vit_tiny","vit_base"], default="swin_tiny")
    p.add_argument("--slot_type", choices=["qsa","isa","sa"], default="qsa")
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--n_unsup", type=int, default=4)
    p.add_argument("--n_concepts", type=int, default=0)
    p.add_argument("--n_spatial", type=int, default=4)
    p.add_argument("--freeze_stages", type=int, default=1)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--lambda_sparse", type=float, default=1.0)
    p.add_argument("--lambda_div", type=float, default=1.0)

    return p

# -----------------------------
# 데이터로더 (ImageFolder)
# -----------------------------
class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path, _ = self.imgs[index]
        return {"image": img, "label": torch.tensor(label, dtype=torch.long), "names": path}

def get_loader(args):
    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    split_dir = os.path.join(args.dataset_root, args.split)
    ds = ImageFolderWithPaths(split_dir, transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    return loader, ds.classes

# -----------------------------
# CCT 로더 (동적 import)
# -----------------------------
def _dynamic_import(entry: str):
    mod_name, fn_name = entry.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    return fn

def load_cct_model(args, device):
    build_fn = _dynamic_import(args.model_entry)
    model = build_fn(args)  # LitCCT 또는 nn.Module 반환(학습 구현에 맞춤)
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Lightning 형태(state_dict) 또는 커스텀 키 대응
    state = None
    if isinstance(ckpt, dict):
        # 보편 키 후보들
        for k in ["state_dict", "model", "net", "module", "model_state", "model_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break
        # 전체 딕셔너리가 곧 state_dict인 경우
        if state is None and all(isinstance(v, (torch.Tensor, np.ndarray)) for v in ckpt.values()):
            state = ckpt
    if state is None and isinstance(ckpt, torch.nn.Module):
        model = ckpt  # 모델 객체 자체가 저장된 경우
    else:
        # LitCCT가 감싼 구조면 자동으로 내부로 로드됨 (strict=False 권장)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:   print("⚠️ missing keys:", missing[:8], "..." if len(missing)>8 else "")
        if unexpected:print("⚠️ unexpected keys:", unexpected[:8], "..." if len(unexpected)>8 else "")

    model = model.to(device).eval()
    return model

# -----------------------------
# 추론 -> 확률/예측
# -----------------------------
@torch.no_grad()
def predict_all(model, loader, device):
    all_probs, all_preds, all_labels = [], [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        out = model(x)
        # CCT는 (logits, ...) 구조일 수 있음
        logits = out[0] if isinstance(out, (tuple, list)) else out
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_probs.append(probs.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_labels.append(y.detach().cpu())

    probs = torch.cat(all_probs, dim=0).numpy()     # [N,C]
    preds = torch.cat(all_preds, dim=0).numpy()     # [N]
    labels= torch.cat(all_labels,dim=0).numpy()     # [N]
    return probs, preds, labels

# -----------------------------
# 메트릭 계산
# -----------------------------
def compute_metrics(probs: np.ndarray, preds: np.ndarray, labels: np.ndarray,
                    pos_label: int, class_names: list[str] | None = None):
    assert probs.shape[0] == preds.shape[0] == labels.shape[0]
    n_classes = probs.shape[1]

    # Confusion matrix (이진 가정일 때 tn,fp,fn,tp 제공)
    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    metrics = {}

    # Accuracy
    metrics["accuracy"] = float(np.trace(cm) / np.maximum(cm.sum(), 1))

    # Precision/Recall/F1 (이진 전용 간단 계산)
    if n_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        precision = tp / np.maximum(tp + fp, 1)
        recall    = tp / np.maximum(tp + fn, 1)
        f1        = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        metrics.update({
            "precision": float(precision),
            "recall":    float(recall),
            "f1":        float(f1),
        })
        # AUC (양성 클래스 확률)
        try:
            metrics["auc"] = float(roc_auc_score(labels, probs[:, pos_label]))
        except Exception:
            metrics["auc"] = float("nan")
    else:
        # 멀티클래스일 경우 macro F1 / macro AUC(ovr)
        try:
            report = classification_report(labels, preds, output_dict=True, zero_division=0)
            metrics["precision_macro"] = float(report["macro avg"]["precision"])
            metrics["recall_macro"]    = float(report["macro avg"]["recall"])
            metrics["f1_macro"]        = float(report["macro avg"]["f1-score"])
        except Exception:
            metrics["precision_macro"] = metrics["recall_macro"] = metrics["f1_macro"] = float("nan")
        try:
            metrics["auc_ovr_macro"] = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
        except Exception:
            metrics["auc_ovr_macro"] = float("nan")

    # Cohen's Kappa
    try:
        metrics["kappa"] = float(cohen_kappa_score(labels, preds))
    except Exception:
        metrics["kappa"] = float("nan")

    # (선택) 클래스 이름 보여주기
    if class_names:
        metrics["classes"] = class_names

    return metrics, cm

# -----------------------------
# main
# -----------------------------
def main():
    args = build_parser().parse_args()
    device = torch.device(args.device)

    loader, class_names = get_loader(args)
    model = load_cct_model(args, device)

    probs, preds, labels = predict_all(model, loader, device)
    metrics, cm = compute_metrics(probs, preds, labels, pos_label=args.pos_label, class_names=class_names)

    # 출력
    print("== Confusion Matrix ==")
    print(cm)
    print("== Metrics ==")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()