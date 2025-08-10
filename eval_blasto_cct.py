from __future__ import annotations
import argparse, os, importlib, types
import torch, torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# ---- SCOUTER metrics ----
from metrics.utils import exp_data
from metrics.IAUC_DAUC import calc_iauc_and_dauc_batch
from metrics.saliency_evaluation.eval_infid_sen import calc_infid_and_sens
from metrics.area_size import calc_area_size

# ============================================================
# 1) 파서
# ============================================================
def build_parser():
    p = argparse.ArgumentParser("CCT on Blastocyst — IAUC/DAUC & Infid/Sens")

    # --- 핵심 경로/모델 ---
    p.add_argument("--checkpoint", required=True, type=str, help=".ckpt 경로")
    p.add_argument("--model_entry", type=str, default=None,
                   help="모델 빌더 진입점 'pkg.mod:func' (예: models.cct_builder:build_model). "
                        "미지정 시 체크포인트에 저장된 모델 객체를 사용 시도")

    # --- 데이터셋/전처리 ---
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=1)

    # --- 평가 토글 ---
    p.add_argument("--auc", action="store_true", help="IAUC/DAUC 계산")
    p.add_argument("--saliency", action="store_true", help="Infidelity/Sensitivity 계산")
    p.add_argument("--area_prec", action="store_true", help="area-size 평균")
    p.add_argument("--loss_status", type=int, default=1,
                   help=">0 이면 positive, <0 이면 negative(L.S.C 타겟). 기본=1")

    # --- 데이터 로더 옵션(Blastocyst) ---
    p.add_argument("--dataset_root", type=str, default=None,
                   help="(선택) 데이터 루트. ConText/MakeListImage가 내부 경로를 알고 있으면 생략 가능")

    return p


# ============================================================
# 2) 데이터 로더 (Blastocyst)
# ============================================================

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)      # img: Tensor, label: int
        path, _ = self.imgs[index]                  # (path, class_idx)
        return {"image": img,
                "label": torch.tensor(label, dtype=torch.long),
                "names": path}

def get_val_loader(args):
    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])
    val_dir = os.path.join(args.dataset_root, "val")
    ds = ImageFolderWithPaths(val_dir, transform=tf)
    return DataLoader(ds, batch_size=args.batch_size,
                      shuffle=False, num_workers=args.num_workers,
                      pin_memory=True)


# ============================================================
# 3) CCT 모델 로더 (동적 import + 다양한 ckpt 형태 지원)
# ============================================================
def _dynamic_import(entry: str):
    """'pkg.mod:func' 형태를 (module, function)으로 로드."""
    mod_name, fn_name = entry.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    return fn

def load_cct_model(args, device):
    ckpt = torch.load(args.checkpoint, map_location=device)

    # A) 체크포인트가 '그 자체가 모델 객체'인 경우
    if isinstance(ckpt, torch.nn.Module):
        model = ckpt.to(device).eval()
        return model

    # B) state_dict 형태 (ckpt['model'] 또는 ckpt 자체가 dict)
    #    -> model_entry로 빌더를 불러 모델 생성 후 로드
    if args.model_entry is None:
        raise ValueError(
            "checkpoint가 state_dict 형태로 보입니다. "
            "--model_entry '패키지.모듈:함수' 를 지정해 모델 인스턴스를 만들어 주세요."
        )

    build_fn = _dynamic_import(args.model_entry)
    # build_fn은 사용자 레포의 빌더 규약에 맞춰야 합니다.
    # 필요 시 args를 확장하세요.
    model = build_fn(args)
    model = model.to(device)

    state_dict = ckpt.get("model", None)
    if state_dict is None:
        # ckpt가 곧 state_dict인 경우
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("⚠️ load_state_dict 경고:")
        if missing:   print("  누락된 키:", missing)
        if unexpected:print("  예상 밖 키:", unexpected)

    model.eval()
    return model


# ============================================================
# 4) CCT → 확률 출력 래퍼 (metrics 가 기대하는 인터페이스)
# ============================================================
class CCTModelWrapper(torch.nn.Module):
    """metrics.* 코드가 model(x)->(B,C) 확률을 기대한다고 가정하여 래핑."""
    def __init__(self, cct_model: torch.nn.Module):
        super().__init__()
        self.cct = cct_model

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cct(x)
        # CCT forward 가 (logits, unsup, concept, spatial_concept) 형태라고 가정
        if isinstance(out, (tuple, list)) and len(out) >= 1:
            logits = out[0]
        else:
            logits = out
        probs = F.softmax(logits, dim=1)
        return probs


# ============================================================
# 5) CCT 설명맵 (Grad-CAM식, 클래스-특이)
# ============================================================
class CCTExplainer:
    def __init__(self, cct_model: torch.nn.Module, input_size=(224,224)):
        self.model = cct_model
        self.input_size = input_size

    def saliency(self, x: torch.Tensor, target_idx: int) -> torch.Tensor:
        """
        Grad-CAM 스타일로 spatial_concept_attn에 대한 클래스-특이 saliency 생성.
        반환: [1, 1, H, W] 에서 [0,1] 정규화된 텐서 (입력 해상도)
        """
        self.model.eval()
        x = x.requires_grad_(True)

        # forward
        logits, *rest = self.model(x)  # (B,C), ...
        spatial = rest[-1]             # spatial_concept_attn [B, K, Hs, Ws] 라고 가정
        if not spatial.requires_grad:
            spatial = spatial.requires_grad_(True)

        # 스칼라 점수: 타겟 클래스 로짓 합
        score = logits[:, target_idx].sum()
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        grads = spatial.grad           # [B,K,Hs,Ws]
        # 채널 중요도: GAP over H,W
        weights = grads.mean(dim=(2,3), keepdim=True)   # [B,K,1,1]
        cam = (weights * spatial).sum(dim=1, keepdim=True)  # [B,1,Hs,Ws]
        cam = F.relu(cam)

        # 업샘플 & [0,1] 정규화
        cam = F.interpolate(cam, size=self.input_size, mode="bilinear", align_corners=False)
        cam_min = cam.amin(dim=(2,3), keepdim=True)
        cam_max = cam.amax(dim=(2,3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)

        # 그래디언트 덮어쓰기 방지
        x.requires_grad_(False)
        spatial.requires_grad_(False)
        return cam.detach()  # [B,1,H,W]


# ============================================================
# 6) 설명맵 PNG 저장 (SCOUTER와 동일 폴더 규약)
# ============================================================
def save_gray_png(mask_01: torch.Tensor, save_path: str):
    """mask_01: [1,1,H,W] in [0,1]"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    m = mask_01.squeeze().cpu().numpy()
    m = np.clip(m * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(m).save(save_path)


def generate_exps_cct(model, loader, device, loss_status=1, img_size=224):
    """
    SCOUTER의 generate_exps와 동등 동작:
    exps/positive or exps/negative 아래에 {파일명}.png 저장.
    """
    explainer = CCTExplainer(model, input_size=(img_size, img_size))
    subdir = "positive" if loss_status > 0 else "negative"
    save_root = os.path.join("exps", subdir)
    os.makedirs(save_root, exist_ok=True)

    model.eval()
    for batch in loader:
        imgs, labels, paths = batch["image"].to(device), batch["label"], batch["names"]

        for i in range(imgs.size(0)):
            img  = imgs[i].unsqueeze(0)  # [1,3,H,W]
            lab  = int(labels[i].item())
            filepath = paths[i]
            base = os.path.splitext(os.path.basename(filepath))[0]
            out_path = os.path.join(save_root, base + ".png")

            if os.path.exists(out_path):
                continue

            # 타겟 클래스: positive = GT, negative = LSC(이진이면 1-lab)
            if loss_status > 0:
                target_idx = lab
            else:
                # 이진 분류 가정. 다중 분류면 argmin(logits) 등 LSC 규칙으로 확장 필요.
                target_idx = 1 - lab

            with torch.enable_grad():
                mask = explainer.saliency(img, target_idx)  # [1,1,H,W] in [0,1]
            save_gray_png(mask, out_path)


# ============================================================
# 7) area-size 유틸 (선택)
# ============================================================
def area_size_only(subdir: str):
    sizes = []
    root = os.path.join("exps", subdir)
    if not os.path.isdir(root):
        return None
    for fname in os.listdir(root):
        if not fname.lower().endswith(".png"):
            continue
        path = os.path.join(root, fname)
        try:
            sizes.append(calc_area_size(Image.open(path)))
        except Exception:
            pass
    return (sum(sizes) / len(sizes)) if sizes else None


# ============================================================
# 8) main
# ============================================================
def main():
    args = build_parser().parse_args()
    device = torch.device(args.device)

    # 1) 데이터 & 모델
    val_loader = get_val_loader(args)
    cct = load_cct_model(args, device)
    model = CCTModelWrapper(cct).to(device)  # metrics가 기대하는 (B,C) 확률 출력을 보장

    # 2) 설명맵 생성 (IAUC/DAUC·Infidelity/Sensitivity·area-size 모두 선행 필요)
    if args.auc or args.saliency or args.area_prec:
        print("[Info] generating explanation images …")
        generate_exps_cct(cct, val_loader, device, loss_status=args.loss_status, img_size=args.img_size)

    subdir = "positive" if args.loss_status > 0 else "negative"
    exp_root = os.path.join("exps", subdir)

    # 3) IAUC / DAUC
    if args.auc:
        files = exp_data.get_exp_filenames(exp_root)
        exp_loader = torch.utils.data.DataLoader(
            exp_data.ExpData(files, args.img_size, resize=True),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        iauc, dauc = calc_iauc_and_dauc_batch(model, val_loader, exp_loader, args.img_size, device)
        print(f"IAUC={iauc:.4f} | DAUC={dauc:.4f}")

    # 4) Infidelity / Sensitivity
    if args.saliency:
        # negative일 때 LSC 사전 — 이진이면 {0:1, 1:0}
        lsc_dict = {"0": 1, "1": 0} if args.loss_status < 0 else {}
        infid_scores, sens_scores = calc_infid_and_sens(
            model, val_loader, exp_root, loss_status=args.loss_status, lsc_dict=lsc_dict
        )
        avg_infid = sum(infid_scores.values()) / max(len(infid_scores), 1)
        avg_sens  = sum(sens_scores.values())  / max(len(sens_scores), 1)
        print(f"Infidelity={avg_infid:.4f} | Sensitivity={avg_sens:.4f}")
        print("Infidelity per pert:", infid_scores)
        print("Sensitivity per pert:", sens_scores)

    # 5) Area-size (선택)
    if args.area_prec:
        avg = area_size_only(subdir)
        if avg is None:
            print("[Warn] heat-map을 찾지 못했습니다 → area-size 계산 실패")
        else:
            print(f"Average area-size = {avg:.4f}")


if __name__ == "__main__":
    main()
