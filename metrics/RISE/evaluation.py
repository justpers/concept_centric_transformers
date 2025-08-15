"""
Code was partially taken and adapted from:

Petsiuk, V., Das, A., & Saenko, K. (2018).
RISE: Randomized Input Sampling for Explanation of Black-box Models.
arXiv:1806.07421

Original code: https://github.com/eclique/RISE  (Commit: d91ea00 on Sep 17, 2018)

This version removes fixed image size assumptions (e.g., 260x260),
supports model(x, softmax=True), and handles arbitrary batch sizes.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import Callable, Optional
from tqdm import tqdm
try:
    # deprecated path in older SciPy but keep compatibility
    from scipy.ndimage.filters import gaussian_filter
except Exception:  # pragma: no cover
    from scipy.ndimage import gaussian_filter  # type: ignore

from metrics.RISE.utils import get_class_name, tensor_imshow


# ------------------------------- utils -------------------------------

def gkern(klen: int, nsig: float) -> torch.Tensor:
    """Returns a (3,3,klen,klen) Gaussian kernel to blur RGB images via conv2d."""
    inp = np.zeros((klen, klen), dtype=np.float32)
    inp[klen // 2, klen // 2] = 1.0
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen), dtype=np.float32)
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern)


def auc(arr: np.ndarray) -> float:
    """Normalized Area Under Curve of the array (trapezoidal rule)."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.size < 2:
        return float(arr.sum())
    return float((arr.sum() - arr[0] / 2.0 - arr[-1] / 2.0) / (arr.shape[0] - 1))


def _ensure_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def _resize_to(img_or_map: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    """Resize a tensor [B,*,H,W] or [*,H,W] to size_hw using bilinear (no align_corners)."""
    if not isinstance(img_or_map, torch.Tensor):
        img_or_map = torch.as_tensor(img_or_map)
    need_unsqueeze = img_or_map.ndim == 3  # [C,H,W] or [1,H,W]
    if img_or_map.ndim == 2:
        img_or_map = img_or_map.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    elif img_or_map.ndim == 3:
        img_or_map = img_or_map.unsqueeze(0)               # [1,C,H,W]
    # else: [B,C,H,W] OK
    out = F.interpolate(img_or_map.float(), size=size_hw, mode="bilinear", align_corners=False)
    return out if not need_unsqueeze else out.squeeze(0)


# --------------------------- main metric -----------------------------

class CausalMetric:
    def __init__(self,
                 model: torch.nn.Module,
                 mode: str,
                 step: int,
                 substrate_fn: Callable[[torch.Tensor], torch.Tensor]):
        """
        Args:
            model: black-box model, must accept model(x, softmax=True|False)
            mode:  'del' or 'ins'
            step:  number of pixels modified at each iteration (>=1)
            substrate_fn: maps original pixels to baseline pixels (e.g., blur/mean/zeros)
        """
        assert mode in ("del", "ins"), "mode must be 'del' or 'ins'"
        self.model = model
        self.mode = mode
        self.step = max(int(step), 1)
        self.substrate_fn = substrate_fn

    def _model_device(self):
    # 모델이 올라간 디바이스 추정
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            # 파라미터가 없는 래퍼일 수도 있으니, CUDA 가능하면 CUDA로
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------ single image ------------------------

    @torch.no_grad()
    def _predict_softmax(self, x: torch.Tensor) -> torch.Tensor:
        dev = self._model_device()
        return self.model(x.to(dev, non_blocking=True).float(), softmax=True)

    def single_run(self,
                   img_tensor: torch.Tensor,
                   explanation: np.ndarray | torch.Tensor,
                   verbose: int = 0,
                   save_to: Optional[str] = None) -> np.ndarray:
        """
        Run metric on one image-saliency pair.

        Args:
            img_tensor: [1,3,H,W] normalized image tensor (on any device)
            explanation: [H',W'] or [1,H',W'] or [1,1,H',W'] saliency in [0,1]
            verbose: 0 (silent), 1 (plot final), 2 (plot each step + top-2)
            save_to: optional directory to save plots

        Returns:
            scores: np.ndarray of length n_steps+1 with probability of top class across steps
        """
        device = img_tensor.device
        B, C, H, W = img_tensor.shape
        assert B == 1, "single_run expects a single image (batch=1)"

        # 1) get target class (top-1 on original)
        pred0 = self._predict_softmax(img_tensor)
        _, c = torch.max(pred0, dim=1)
        c = int(c.item())

        # 2) ensure explanation matches image resolution
        exp = _ensure_tensor(explanation).float().to(device)
        if exp.ndim == 4 and exp.size(0) == 1 and exp.size(1) == 1:
            exp = exp[0, 0]  # [H',W']
        elif exp.ndim == 3 and exp.size(0) == 1:
            exp = exp[0]     # [H',W']
        elif exp.ndim == 2:
            pass
        else:
            # [B,H,W] or others → take first sample/map
            exp = exp.view(-1, *exp.shape[-2:])[0]
        exp = _resize_to(exp.unsqueeze(0).unsqueeze(0), (H, W))[0, 0]  # [H,W] at image size

        # 3) number of steps
        HW = H * W
        n_steps = (HW + self.step - 1) // self.step

        # 4) start/finish images
        if self.mode == "del":
            title = "Deletion game"
            ylabel = "Pixels deleted"
            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)
        else:
            title = "Insertion game"
            ylabel = "Pixels inserted"
            start = self.substrate_fn(img_tensor)
            finish = img_tensor.clone()

        # 5) saliency descending order (flatten)
        salient_order = torch.argsort(exp.reshape(-1), descending=True).detach().cpu().numpy()[None, :]  # [1,HW]

        scores = np.empty(n_steps + 1, dtype=np.float32)
        for i in range(n_steps + 1):
            pred = self._predict_softmax(start)
            scores[i] = float(pred[0, c].item())

            # Verbose plotting
            if verbose == 2 or (verbose == 1 and i == n_steps) or save_to:
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.title(f"{ylabel} {100 * i / n_steps:.1f}%, P={scores[i]:.4f}")
                plt.axis("off")
                tensor_imshow(start[0].detach().cpu())

                plt.subplot(122)
                xs = np.arange(i + 1) / max(n_steps, 1)
                plt.plot(xs, scores[: i + 1])
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(xs, 0, scores[: i + 1], alpha=0.4)
                plt.title(title)
                plt.xlabel(ylabel)
                plt.ylabel(get_class_name(c))
                if save_to:
                    plt.savefig(f"{save_to}/{i:03d}.png")
                    plt.close()
                elif verbose:
                    plt.show()

            # modify next chunk
            if i < n_steps:
                coords = salient_order[:, self.step * i : self.step * (i + 1)]  # [1,k]
                # unravel into (row, col)
                rows, cols = np.unravel_index(coords, (H, W))
                # apply to current image
                start[0, :, rows[0], cols[0]] = finish[0, :, rows[0], cols[0]]

        return scores

    # ------------------------ batched images ------------------------

    def evaluate(self,
                 img_batch: torch.Tensor,
                 exp_batch: np.ndarray | torch.Tensor,
                 batch_size: int) -> np.ndarray:
        """
        Efficiently evaluate a batch of images.

        Args:
            img_batch:  [B,3,H,W] normalized tensor (device can be cpu/cuda)
            exp_batch:  [B,H',W'] or [B,1,H',W'] saliency (0..1)
            batch_size: micro-batch size for model forward

        Returns:
            scores: np.ndarray of shape [(n_steps+1), B] with probabilities
        """
        assert img_batch.ndim == 4 and img_batch.size(1) == 3, "img_batch must be [B,3,H,W]"
        device = img_batch.device
        B_img, _, H, W = img_batch.shape

        # normalize exp_batch shape and move to device
        exp_t = _ensure_tensor(exp_batch).float().to(device)
        if exp_t.ndim == 4:
            # [B',C',H',W'] -> 채널 평균으로 단일 채널
            if exp_t.size(1) > 1:
                exp_t = exp_t.mean(dim=1, keepdim=True)          # [B',1,H',W']
        elif exp_t.ndim == 3:
            exp_t = exp_t.unsqueeze(1)                            # [B',1,H',W']
        elif exp_t.ndim == 2:
            # [H',W'] → 일단 이미지 배치 크기로 브로드캐스트
            exp_t = exp_t.unsqueeze(0).unsqueeze(0).expand(B_img, 1, -1, -1)  # [B_img,1,H',W']
        else:
            raise ValueError(f"Unexpected exp_batch ndim={exp_t.ndim}")

        # 배치 수 정렬: N = min(B_img, B_exp)
        B_exp = exp_t.size(0)
        if B_exp != B_img:
            N = min(B_img, B_exp)
            img_batch = img_batch[:N]
            exp_t     = exp_t[:N]
            B = N
            H, W = img_batch.shape[-2], img_batch.shape[-1]
        else:
            B = B_img

        HW = H * W
        n_steps = (HW + self.step - 1) // self.step

        # 3) 설명맵을 이미지 해상도(H,W)로 리사이즈 + 평탄화
        exp_resized = F.interpolate(exp_t, size=(H, W), mode="bilinear", align_corners=False)  # [B,1,H,W]
        exp_flat    = exp_resized.view(B, -1)                                         
        salient_order = torch.argsort(exp_flat, dim=1, descending=True).detach().cpu().numpy()  # [B, HW]

        # 2) determine top-1 class per image (once)
        probs_list = []
        img_cpu = img_batch  # keep reference
        n_full = B // batch_size
        rest = B % batch_size

        # dynamic n_classes from first forward
        with torch.no_grad():
            s0 = self._predict_softmax(img_batch[: min(batch_size, B)])
        n_classes = int(s0.shape[1])
        predictions = torch.empty(B, n_classes, dtype=torch.float32)

        # full chunks
        idx = 0
        for _ in tqdm(range(n_full + (1 if rest > 0 else 0)), desc="Predicting labels"):
            bs = batch_size if idx + batch_size <= B else (B - idx)
            preds = self._predict_softmax(img_batch[idx: idx + bs])
            predictions[idx: idx + bs] = preds.cpu()
            idx += bs

        top = torch.argmax(predictions, dim=1).numpy()  # [B]

        # 3) precompute substrate
        substrate = torch.zeros_like(img_batch)
        idx = 0
        for _ in tqdm(range(n_full + (1 if rest > 0 else 0)), desc="Substrate"):
            bs = batch_size if idx + batch_size <= B else (B - idx)
            substrate[idx: idx + bs] = self.substrate_fn(img_batch[idx: idx + bs])
            idx += bs

        # 4) init start/finish
        if self.mode == "del":
            caption = "Deleting  "
            start = img_batch.clone()
            finish = substrate
        else:
            caption = "Inserting "
            start = substrate
            finish = img_batch.clone()

        # 5) iterate pixels
        scores = np.empty((n_steps + 1, B), dtype=np.float32)

        for i in tqdm(range(n_steps + 1), desc=caption + "pixels"):
            # (a) model forward on current start
            idx = 0
            for _ in range(n_full + (1 if rest > 0 else 0)):
                bs = batch_size if idx + batch_size <= B else (B - idx)
                preds = self._predict_softmax(start[idx: idx + bs])
                # gather prob of each sample's top class
                pr = preds[torch.arange(bs), torch.as_tensor(top[idx: idx + bs], device=preds.device)]
                scores[i, idx: idx + bs] = pr.detach().cpu().numpy()
                idx += bs

            # (b) change next chunk of pixels
            if i < n_steps:
                coords = salient_order[:, self.step * i: self.step * (i + 1)]  # [B, k]
                rows, cols = np.unravel_index(coords, (H, W))                   # each is [B, k]
                # vectorized replace per sample
                for b in range(B):
                    r = rows[b]
                    c = cols[b]
                    if r.size == 0:  # no pixels left
                        continue
                    start[b, :, r, c] = finish[b, :, r, c]

        # 6) summary
        print("AUC: {:.6f}".format(auc(scores.mean(1))))
        return scores
