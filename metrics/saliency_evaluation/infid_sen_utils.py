"""
Code was taken and adapted from:

Yeh, C. K., Hsieh, C. Y., Suggala, A., Inouye, D. I., & Ravikumar, P. K. (2019).
On the (in) fidelity and sensitivity of explanations.
NeurIPS 32, 10967-10978.

Original: https://github.com/chihkuanyeh/saliency_evaluation (Commit: 44a66e2, Oct 5, 2020)
"""

import os, math
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

FORWARD_BZ = 5000

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def forward_batch(model, input, batchsize):
    inputsize = input.shape[0]
    for count in range((inputsize - 1) // batchsize + 1):
        end = min(inputsize, (count + 1) * batchsize)
        tempinput = input[count * batchsize : end]
        if count == 0:
            out = model(tempinput.cuda()).data.cpu().numpy()
        else:
            temp = model(tempinput.cuda()).data
            out = np.concatenate([out, temp.cpu().numpy()], axis=0)
    return out

def sample_eps_Inf(image, epsilon, N):
    images = np.tile(image, (N, 1, 1, 1))
    dim = images.shape
    return np.random.uniform(-1 * epsilon, epsilon, size=dim)

def set_zero_infid(array, size, point, pert):
    if pert == "Gaussian":
        ind = np.random.choice(size, point, replace=False)
        randd = np.random.normal(size=point) * 0.2 + array[ind]
        randd = np.minimum(array[ind], randd)
        randd = np.maximum(array[ind] - 1, randd)
        array[ind] -= randd
        return np.concatenate([array, ind, randd])
    else:
        raise ValueError(f"pert {pert} is not supported.")

def _to_uint8(arr):
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        if a.max() <= 1.0 + 1e-6 and a.min() >= 0.0 - 1e-6:
            a = (a * 255.0).astype(np.uint8)
        else:
            a = np.clip(a, 0, 255).astype(np.uint8)
    return a

def _resize_map_np(expl_like, H, W):
    if expl_like.ndim == 3:
        expl_like = expl_like[0]
    if expl_like.ndim != 2:
        expl_like = np.asarray(expl_like).squeeze()
    img = Image.fromarray(_to_uint8(expl_like))
    return np.array(img.resize((W, H), resample=Image.BILINEAR), dtype=np.uint8)

def _resize_map_torch(expl_t: torch.Tensor, H: int, W: int) -> np.ndarray:
    m = expl_t.detach()
    if m.ndim == 2:
        m = m.unsqueeze(0).unsqueeze(0)            # [1,1,H',W']
    elif m.ndim == 3:
        if m.size(0) != 1:
            m = m.mean(dim=0, keepdim=True).unsqueeze(0)
        else:
            m = m.unsqueeze(0)
    elif m.ndim == 4 and m.size(1) != 1:
        m = m.mean(dim=1, keepdim=True)
    m = F.interpolate(m.float(), size=(H, W), mode="bilinear", align_corners=False)  # [1,1,H,W]
    return _to_uint8(m[0, 0].cpu().numpy())

def get_exp(ind, exp):
    return exp[ind.astype(int)]

# ------------------------------------------------------------
# Infidelity
# ------------------------------------------------------------
def get_exp_infid(image, model, exp, label, pdt, binary_I, pert):
    """
    image : torch.Tensor [3,H,W] 또는 [1,3,H,W]
    exp   : np.ndarray/torch.Tensor [H',W'] / [1,H',W'] / [1,1,H',W']
    label : 정답 클래스 (torch.Tensor or int)
    """
    # 1) 입력 해상도
    if isinstance(image, torch.Tensor):
        img_t = image.detach()
        if img_t.ndim == 4:
            img_t = img_t[0]
        C, H, W = img_t.shape
        img_np = img_t.cpu().numpy()
    else:
        raise TypeError("image must be a torch.Tensor")

    # 라벨을 int로 보장
    if isinstance(label, torch.Tensor):
        label = int(label.item())
    else:
        label = int(label)

    point = H * W
    num   = 100

    # 2) 설명맵을 (H,W)로 리사이즈 후 평탄화
    exp_t = torch.from_numpy(exp).float() if isinstance(exp, np.ndarray) else torch.as_tensor(exp).float()
    if exp_t.ndim == 2:
        exp_t = exp_t.unsqueeze(0).unsqueeze(0)
    elif exp_t.ndim == 3:
        if exp_t.size(0) != 1:
            exp_t = exp_t.mean(dim=0, keepdim=True).unsqueeze(0)
        else:
            exp_t = exp_t.unsqueeze(0)
    elif exp_t.ndim == 4 and exp_t.size(1) != 1:
        exp_t = exp_t.mean(dim=1, keepdim=True)
    exp_t = F.interpolate(exp_t, size=(H, W), mode="bilinear", align_corners=False)  # [1,1,H,W]
    exp_copy = exp_t.view(-1).cpu().numpy()      # 길이 = point
    total = point

    # 3) 이미지 평탄화 & 샘플 복제
    image_copy = np.tile(img_np.reshape(1, C, point), [num, 1, 1])  # [num,3,point]

    # 4) perturbation 인덱스/노이즈 생성
    image_copy_ind = np.apply_along_axis(set_zero_infid, 2, image_copy, total, point, pert)

    if pert == "Gaussian" and not binary_I:
        image_copy = image_copy_ind[:, :, :total]                           # [num,3,point]
        ind       = image_copy_ind[:, :, total : total + point]             # [num,3,point]
        rand      = image_copy_ind[:, :, total + point : total + 2*point]   # [num,3,point]
        exp_sum   = np.sum(rand * np.apply_along_axis(get_exp, 1, ind, exp_copy), axis=2)  # [num,3]
        ks        = np.ones(num, dtype=np.float32)
    else:
        raise ValueError("Perturbation type and binary_I do not match.")

    # 5) 배치 추론
    image_copy = image_copy.reshape(num, C, H, W)
    image_v = Variable(torch.from_numpy(image_copy.astype(np.float32)).cuda(), requires_grad=False)
    out = forward_batch(model, image_v, FORWARD_BZ)        # (num, num_classes)
    pdt_rm   = out[:, label]
    pdt_diff = np.squeeze(pdt - pdt_rm)                    # [num]
    exp_sum  = np.mean(exp_sum, axis=1)                    # [num]

    # 6) 최적 스케일링 & Infidelity
    eps = 1e-12
    beta = np.mean(ks * pdt_diff * exp_sum) / (np.mean(ks * exp_sum * exp_sum) + eps)
    exp_sum *= beta
    infid = np.mean(ks * np.square(pdt_diff - exp_sum)) / (np.mean(ks) + eps)
    return float(infid)

# ------------------------------------------------------------
# Sensitivity
# ------------------------------------------------------------
def get_exp_sens(X, model, expl, yy, sen_r, sen_N, norm):
    """
    Sensitivity = max_{||ε||<=r} ||Φ(x+ε) - Φ(x)|| / ||Φ(x)||
    여기서는 PNG가 아닌 explainer로 clean/noisy 설명맵을 직접 생성합니다.
    X  : [3,H,W] 또는 [1,3,H,W] (torch.Tensor, 정규화 완료)
    model.explainer : CCTExplainer 가 연결되어 있어야 함 (eval_blasto_cct.py에서 이미 연결함)
    yy : 타깃 클래스 (tensor scalar)
    sen_r : 노이즈 범위
    sen_N : 샘플 횟수
    norm 파라미터는 유지하지만, 실제로는 여기서 clean 맵으로 다시 계산합니다.
    """
    # 0) 배치 차원 보장
    if X.ndim == 3:
        X = X.unsqueeze(0)
    assert X.ndim == 4 and X.size(0) == 1, "X는 [1,3,H,W]여야 합니다."
    device = X.device
    H, W = X.shape[-2], X.shape[-1]
    cls_idx = int(yy.item())

    # 1) 기준(깨끗한 X)의 설명맵을 explainer로 생성 (0..1) → uint8 [H,W]
    if hasattr(model, "explainer") and model.explainer is not None:
        with torch.enable_grad():
            sal_clean = model.explainer.saliency(X, cls_idx)     # [1,1,H,W], 0..1
        expl_ref = _resize_map_torch(sal_clean[0,0], H, W)        # uint8 [H,W]
    else:
        # 폴백: 주어진 expl(보통 PNG)을 리사이즈
        if isinstance(expl, torch.Tensor):
            expl_ref = _resize_map_torch(expl, H, W)
        else:
            expl_ref = _resize_map_np(np.asarray(expl), H, W)

    ref_norm = float(np.linalg.norm(expl_ref.astype(np.float32)))
    if ref_norm < 1e-6:
        # 완전 검정/거의 0 맵 → 폭주 방지 (스킵하거나 0 반환)
        return 0.0

    max_diff = -float("inf")
    for _ in range(sen_N):
        # 2) 작은 섭동 샘플
        eps_np = sample_eps_Inf(X.detach().cpu().numpy(), sen_r, 1)
        sample = torch.from_numpy(eps_np).to(device=device, dtype=X.dtype)
        X_noisy = X + sample

        # 3) noisy 설명맵을 explainer로 생성
        if hasattr(model, "explainer") and model.explainer is not None:
            with torch.enable_grad():
                sal_noisy = model.explainer.saliency(X_noisy, cls_idx)   # [1,1,H,W]
            expl_eps = _resize_map_torch(sal_noisy[0,0], H, W)           # uint8 [H,W]
        else:
            # 폴백: SCOUTER 방식
            _ = model(X_noisy, sens=cls_idx)  # side effect 가정
            expl_eps = np.array(
                Image.open("noisy.png").resize((W, H), resample=Image.BILINEAR),
                dtype=np.uint8
            )

        # 4) L2 차이 & 최대값 갱신
        diff = float(np.linalg.norm(
            expl_ref.astype(np.float32) - expl_eps.astype(np.float32)
        ))
        if diff > max_diff:
            max_diff = diff

    return float(max_diff / (ref_norm + 1e-12))

# ------------------------------------------------------------
# Driver
# ------------------------------------------------------------
def evaluate_infid_sen(loader, model,
                       exp_path, loss_status, lsc_dict,
                       pert, sen_r, sen_N):
    if pert != "Gaussian":
        raise NotImplementedError("Only support Gaussian perturbation.")
    binary_I = False

    model.eval()
    infids, max_sens = [], []

    for i, batch in enumerate(loader):
        if i >= 50:                      # 논문 설정(최대 50장)
            break

        X_all = batch["image"].cuda()    # (B,3,H,W)
        y_all = batch["label"].cuda()    # (B,)
        names = batch["names"]           # 길이 B 리스트

        # negative 모델일 때 LSC 매핑
        if loss_status < 0:
            y_all = torch.tensor([lsc_dict[str(y.item())] for y in y_all],
                                 device=y_all.device)

        for img, yy, fname in zip(X_all, y_all, names):
            # ----- (1) 예측 확률 p(x)[y] -----
            with torch.no_grad():
                pdt_val = model(img.unsqueeze(0))[:, yy].cpu().numpy()  # shape (1,)

            # ----- (2) Infidelity: PNG 설명맵 사용 (기존 유지) -----
            base_id = os.path.basename(fname)
            ex_png = os.path.join(exp_path, base_id)  # generate_exps_cct가 만든 PNG
            if not os.path.isfile(ex_png):
                # 히트맵 누락 시 이 샘플은 스킵
                # print(f"[warn] missing heatmap: {ex_png}")
                continue

            expl_png = np.array(
                Image.open(ex_png),
                dtype=np.uint8
            )
            # 원 함수는 내부에서 260 고정 리사이즈 등을 처리하므로 여기선 그대로 전달
            infid = get_exp_infid(img, model, expl_png, yy, pdt_val,
                                  binary_I=binary_I, pert=pert)
            infids.append(infid)

            # ----- (3) Sensitivity: explainer로 clean/noisy 모두 생성 -----
            # norm 인자는 사용하지 않지만 시그니처 유지 위해 값을 넣습니다.
            sens  = get_exp_sens(img, model, expl_png, yy, sen_r, sen_N, norm=0.0)
            max_sens.append(sens)

    # 평균 반환
    return float(np.mean(infids)) if infids else 0.0, float(np.mean(max_sens)) if max_sens else 0.0