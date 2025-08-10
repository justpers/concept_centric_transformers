"""
Calculate the following confusion matrix metrics:
Area Under Curve, accuracy, precision, recall, F1-score and Kappa for the given model.

In the experiments, these metrics are only reported (and thus implemented) for the ACRIMA dataset.
"""
import argparse, os, torch
from sklearn.metrics import confusion_matrix, roc_auc_score, cohen_kappa_score

from train import get_args_parser
from sloter.slot_model import SlotModel

def calc_metrics(model, imgs, labels):
    """Calculate all metrics."""
    model.eval()
    with torch.no_grad():
        out = model(imgs)
    pred_probs = out[0] if isinstance(out, (tuple, list)) else out  # (B, C)
    preds = torch.argmax(pred_probs, dim=1)

    tn, fp, fn, tp = confusion_matrix(labels.cpu(), preds.cpu()).ravel()

    # 먼저 precision, recall 계산
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    metrics = {
        "auc":       roc_auc_score(labels.cpu(), pred_probs[:, 1].cpu()),
        "accuracy":  (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0,
        "precision": precision,
        "recall":    recall,
        "f1":        2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0,
        "kappa":     cohen_kappa_score(labels.cpu(), preds.cpu())
    }
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Metrics", parents=[get_args_parser()])
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="(optional) full path to the .pth checkpoint. "
             "If not set, will look in output_dir for "
             "<dataset>_<model>_<use_slot|no_slot>_checkpoint.pth"
    )
    args = parser.parse_args()

    assert args.dataset in ["ACRIMA", "Blastocyst"], \
        "Only ACRIMA or Blastocyst supported."

    device = torch.device(args.device)

    # 1) Prepare validation data_loader (same as before)
    if args.dataset == "ACRIMA":
        from dataset.ACRIMA import get_data, ACRIMA
        from dataset.transform_func import make_transform
        _, val_data = get_data(args.dataset_dir)
        val_dataset = ACRIMA(val_data, transform=make_transform(args, "val"))
        bs = len(val_dataset)
    else:
        from dataset.ConText import ConText, MakeListImage
        from torchvision import transforms
        _, val_list = MakeListImage(args).get_data()
        tf = transforms.Compose([
            transforms.Resize((args.img_size,args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225]),
        ])
        val_dataset = ConText(val_list, transform=tf)
        bs = args.batch_size

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=bs, shuffle=False, num_workers=1, pin_memory=True
    )

    # 2) Load checkpoint
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        mode_tag = "use_slot" if args.use_slot else "no_slot"
        ckpt_fname = f"{args.dataset}_{args.model}_{mode_tag}_checkpoint.pth"
        ckpt_path = os.path.join(args.output_dir, ckpt_fname)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    model = SlotModel(args).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    # Slot/non-slot 간 state_dict 키 차이 무시하고 불러오기
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if missing or unexpected:
        print("⚠️  load_state_dict 경고:")
        if missing:
            print("   누락된 파라미터:", missing)
        if unexpected:
            print("   예기치 않은 파라미터:", unexpected)

    # 3) Compute metrics on one batch
    for batch in val_loader:
        imgs   = batch["image"].to(device, dtype=torch.float32)
        labels = batch["label"].to(device)
        metrics = calc_metrics(model, imgs, labels)
        print(metrics)
        break