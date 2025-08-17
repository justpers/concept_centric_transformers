import os, torch
from PIL import Image
import matplotlib.cm as cm
from embryo_unsup import LitCCT, EmbryoDataModule, CCTCfg
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import argparse, os

# ───────────────────────────── 헬퍼 ──────────────────────────────
def to_rgb(img_t, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    # 역정규화 → 0~1
    for t, m, s in zip(img_t, mean, std):
        t.mul_(s).add_(m)
    img = img_t.clamp(0,1).permute(1,2,0).cpu().numpy()
    return Image.fromarray((img*255).astype("uint8"), mode="RGB")
def attn_to_mask(vec, n_slots, img_size=224, patch=16):
    """
    vec : (C,N) or (N,C) 1-이미지 어텐션
    return (C,H,W)
    """
    if vec.dim() == 1:                  # (N,) 단일 슬롯인 경우
        vec = vec.unsqueeze(0)          # (1,N)
    if vec.size(0) != n_slots:          # (N,C) → (C,N)
        vec = vec.transpose(0,1)

    h_tok = w_tok = img_size // patch   # 224/16=14
    vec = vec.reshape(n_slots, h_tok, w_tok)
    vec = torch.nn.functional.interpolate(
        vec.unsqueeze(1).float(), size=img_size,
        mode="bilinear", align_corners=False
    ).squeeze(1)                        # (C,H,W)
    return vec

def overlay(img_pil, mask, cmap='jet', alpha=0.4):
    heat = cm.get_cmap(cmap)(mask.numpy())     # (H,W,4)
    heat[..., 3] = alpha                       # 투명도
    heatmap = Image.fromarray((heat*255).astype("uint8"))
    return Image.alpha_composite(img_pil.convert("RGBA"), heatmap)

# ───────────────────────────── main ──────────────────────────────
def main(
    ckpt_path="checkpoints/lightning_logs/version_1/checkpoints/epoch=39-step=1440.ckpt",
    out_dir="vis_slots",
    n_unsup=4,
    img_size=224,
    top_k=30,
    thres=0.6,
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) datamodule
    dm = EmbryoDataModule(root="./kromp", img_size=img_size,
                          batch_size=16, num_workers=4)
    dm.setup("test")

    # 2) 모델 복원 (학습 때 사용한 하이퍼파라미터 그대로)
    print("🔹 checkpoint 로드 중 …")
    cfg = CCTCfg(backbone="swin_tiny", slot_type="qsa", n_unsup=n_unsup)
    model = LitCCT.load_from_checkpoint(
        ckpt_path,
        cfg=cfg,
        lr=1e-4, weight_decay=0.05,
        lambda_sparse=1.0, lambda_div=1.0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    model.freeze()
    #model.eval();  model.freeze()

    # 3) test forward → spatial_attn 수집
    print("🔹 test set forward (예측) …")
    spat_list = []
    for imgs, _ in dm.test_dataloader():
        with torch.no_grad():
            imgs = imgs.to(device)
            *_, spatial_attn = model.model(imgs)        # (B,N,C)  or (B,C,N)
        spat_list.append(spatial_attn.cpu())
    attn = torch.cat(spat_list)                         # (B,*,*)
    if attn.shape[1] != n_unsup:                        # (B,N,C) → (B,C,N)
        attn = attn.transpose(1,2)

    B, C, N = attn.shape
    print(f"after squeeze : {attn.shape}")              # (B,C,N) 확인

    # 4) 슬롯별 Top-k 시각화
    print("🔹 시각화 저장 …")
    for s in range(n_unsup):
        slot_dir = os.path.join(out_dir, f"slot{s}")
        os.makedirs(slot_dir, exist_ok=True)

        # patch 평균값으로 Top-k 선정
        scores = attn[:, s, :].mean(-1)                 # (B,)
        k = min(top_k, B)
        top_idx = torch.topk(scores, k).indices

        for rank, idx in enumerate(top_idx):
            img_t, _ = dm.test_ds[idx]                  # (3,H,W)
            img_pil = to_rgb(img_t)                     # 0-1 tensor → PIL

            mask_map = attn_to_mask(attn[idx], n_unsup, img_size)[s]  # (H,W)
            mask_bin = (mask_map > thres * mask_map.max()).float()

            heat = overlay(img_pil, mask_bin)
            img_pil.save(os.path.join(slot_dir, f"{rank:02d}_origin.png"))
            heat.save   (os.path.join(slot_dir, f"{rank:02d}_mask.png"))

        print(f"✅ slot{s}: {k}장 저장 완료")

    print("🎉 완료!")

# ──────────────────────────── 실행 ─────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", type=str,
                   default="checkpoints/lightning_logs/version_1/checkpoints/epoch=39-step=1440.ckpt")
    p.add_argument("--out_dir", type=str, default="vis_slots")
    p.add_argument("--n_unsup", type=int, default=4)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--top_k", type=int, default=30)
    p.add_argument("--thres", type=float, default=0.6)
    args = p.parse_args()

    # (선택) 경로 확인 메시지
    print("ckpt_path =", os.path.abspath(args.ckpt_path))

    main(
        ckpt_path=args.ckpt_path,
        out_dir=args.out_dir,
        n_unsup=args.n_unsup,
        img_size=args.img_size,
        top_k=args.top_k,
        thres=args.thres,
    )