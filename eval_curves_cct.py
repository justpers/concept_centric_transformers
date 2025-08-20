import argparse, os, torch, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve, f1_score, auc, average_precision_score,
    confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
)

from embryo_unsup import LitCCT, EmbryoDataModule, CCTCfg


# --------------------------- 유틸 ---------------------------
def get_logits(output, num_classes: int):
    """LitCCT forward 출력에서 (B,num_classes) 로짓 텐서를 안전하게 추출"""
    if torch.is_tensor(output) and output.dim() == 2 and output.size(1) == num_classes:
        return output
    if isinstance(output, (tuple, list)):
        for t in output:
            if torch.is_tensor(t) and t.dim() == 2 and t.size(1) == num_classes:
                return t
        for t in output:
            if torch.is_tensor(t):
                return t
    raise RuntimeError("로짓 텐서를 찾지 못했습니다. 모델 forward 반환을 확인하세요.")


def ensure_dir(p):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def plot_confmat(cm, class_names=("neg","pos"), title="Confusion Matrix", savepath=None):
    plt.figure(figsize=(4.2, 3.6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names); plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label'); plt.xlabel('Predicted label'); plt.tight_layout()
    if savepath: plt.savefig(savepath, dpi=180, bbox_inches="tight")
    plt.show()


# --------------------------- 메인 ---------------------------
def main():
    p = argparse.ArgumentParser()
    # 데이터/로드
    p.add_argument("--root", type=str, default="./hmc")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["val","test"])
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)

    # 모델(학습과 동일하게!)
    p.add_argument("--backbone", type=str, default="swin_tiny")
    p.add_argument("--slot_type", type=str, default="qsa")
    p.add_argument("--n_unsup", type=int, default=4)
    p.add_argument("--n_concepts", type=int, default=0)
    p.add_argument("--n_spatial", type=int, default=4)
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--freeze_stages", type=int, default=3)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--lambda_sparse", type=float, default=1.0)
    p.add_argument("--lambda_div", type=float, default=1.0)

    # 평가 옵션
    p.add_argument("--pos_index", type=int, default=1, help="양성 클래스 인덱스")
    p.add_argument("--threshold", type=float, default=None,
                   help="직접 임계값 지정(미지정 시 F1 최대 임계값 사용)")
    p.add_argument("--load_threshold", type=str, default=None,
                   help="파일에서 임계값 로드(.txt)")
    p.add_argument("--save_threshold", type=str, default=None,
                   help="계산한 최적 임계값을 파일로 저장(.txt)")
    p.add_argument("--out_dir", type=str, default=None,
                   help="그래프/혼동행렬 이미지를 저장할 디렉토리(미지정 시 화면표시만)")

    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.out_dir)

    # 1) DataModule
    dm = EmbryoDataModule(root=args.root, img_size=args.img_size,
                          batch_size=args.batch_size, num_workers=args.num_workers)
    dm.setup("fit" if args.split == "val" else "test")
    loader = dm.val_dataloader() if args.split == "val" else dm.test_dataloader()

    # 2) 모델 로드
    cfg = CCTCfg(
        backbone    = args.backbone,
        slot_type   = args.slot_type,
        num_classes = args.num_classes,
        n_unsup     = args.n_unsup,
        n_concepts  = args.n_concepts,
        n_spatial   = args.n_spatial,
        pretrained  = True,
        attn_drop   = 0.1,
        proj_drop   = 0.1,
    )
    model = LitCCT.load_from_checkpoint(
        args.ckpt,
        cfg=cfg,
        freeze_stages=args.freeze_stages,
        warmup_epochs=args.warmup_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_sparse=args.lambda_sparse,
        lambda_div=args.lambda_div,
        n_unsup=args.n_unsup,
    )
    model.to(device).eval(); model.freeze()

    # 3) 스코어 수집
    probs, labels = [], []
    with torch.no_grad():
        for imgs, y in loader:
            imgs = imgs.to(device)
            out = model(imgs)  # LightningModule.forward
            logits = get_logits(out, args.num_classes)
            p_pos = torch.softmax(logits, dim=1)[:, args.pos_index]
            probs.append(p_pos.cpu().numpy())
            labels.append(y.numpy())
    probs  = np.concatenate(probs)
    labels = np.concatenate(labels).astype(int)

    # 분포 출력
    n_pos = int(labels.sum())
    print(f"[{args.split}] N={len(labels)} | Pos={n_pos} ({n_pos/len(labels):.2%}) | Neg={len(labels)-n_pos}")

    # 4) ROC/PR
    fpr, tpr, _ = roc_curve(labels, probs, pos_label=1)
    prec, rec, thr_pr = precision_recall_curve(labels, probs, pos_label=1)
    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(labels, probs, pos_label=1)

    # 5) 임계값 결정
    chosen_thr = None
    src = "F1-max"
    if args.load_threshold and os.path.isfile(args.load_threshold):
        with open(args.load_threshold, "r") as f:
            chosen_thr = float(f.read().strip())
        src = f"loaded({args.load_threshold})"
    if args.threshold is not None:
        chosen_thr = float(args.threshold)
        src = "given(--threshold)"
    if chosen_thr is None:
        # F1 최대 임계값
        f1s = [f1_score(labels, (probs >= t).astype(int)) for t in thr_pr]
        best_idx = int(np.argmax(f1s))
        chosen_thr = float(thr_pr[best_idx])
        best_f1 = float(f1s[best_idx])
        if args.save_threshold:
            with open(args.save_threshold, "w") as f:
                f.write(str(chosen_thr))
        print(f"[{args.split}] F1 최대 임계값 t*={chosen_thr:.4f} → F1={best_f1:.3f} (저장: {args.save_threshold})")
    else:
        print(f"[{args.split}] 임계값 t={chosen_thr:.4f} ({src}) 사용")

    # 6) 혼동행렬/지표
    y_pred = (probs >= chosen_thr).astype(int)
    cm = confusion_matrix(labels, y_pred)      # [[TN,FP],[FN,TP]]
    TN, FP, FN, TP = cm.ravel()
    acc  = accuracy_score(labels, y_pred)
    prec_ = precision_score(labels, y_pred, zero_division=0)
    rec_  = recall_score(labels, y_pred)       # sensitivity
    spec  = TN / (TN + FP) if (TN+FP) > 0 else 0.0
    f1    = f1_score(labels, y_pred)

    print(f"[{args.split}] ROC-AUC={roc_auc:.3f} | AP={ap:.3f}")
    print("Confusion Matrix:\n", cm)
    print(f"ACC={acc:.3f} | Precision={prec_:.3f} | Recall/Sensitivity={rec_:.3f} | Specificity={spec:.3f} | F1={f1:.3f}")
    print(classification_report(labels, y_pred, target_names=["neg","pos"], digits=3))

    # 7) 그래프(표시 + 선택 저장)
    # ROC
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC ({args.split})"); plt.legend()

    # PR
    plt.subplot(1,2,2)
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR ({args.split})"); plt.legend()
    plt.tight_layout()
    if args.out_dir:
        plt.savefig(os.path.join(args.out_dir, f"roc_pr_{args.split}.png"), dpi=180, bbox_inches="tight")
    plt.show()

    # F1 vs Threshold (PR 커브에서 나온 thr 사용)
    if len(thr_pr) > 0:
        f1s = [f1_score(labels, (probs >= t).astype(int)) for t in thr_pr]
        plt.figure(figsize=(4,3))
        plt.plot(thr_pr, f1s)
        plt.axvline(chosen_thr, c='r', ls='--', label=f"t={chosen_thr:.3f}")
        plt.xlabel("Threshold"); plt.ylabel("F1"); plt.title(f"F1 vs Threshold ({args.split})")
        plt.legend(); plt.tight_layout()
        if args.out_dir:
            plt.savefig(os.path.join(args.out_dir, f"f1_thr_{args.split}.png"), dpi=180, bbox_inches="tight")
        plt.show()

    # Confusion Matrix plot
    plot_confmat(
        cm, class_names=("neg","pos"),
        title=f"Confusion Matrix ({args.split}) @ t={chosen_thr:.3f}",
        savepath=(os.path.join(args.out_dir, f"confmat_{args.split}.png") if args.out_dir else None),
    )


if __name__ == "__main__":
    main()