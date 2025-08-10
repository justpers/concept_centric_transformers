""" 
Code was taken and adapted from the following paper:

Yeh, C. K., Hsieh, C. Y., Suggala, A., Inouye, D. I., & Ravikumar, P. K. (2019). 
On the (in) fidelity and sensitivity of explanations. 
Advances in Neural Information Processing Systems, 32, 10967-10978.

Code available at: https://github.com/chihkuanyeh/saliency_evaluation
Commit: 44a66e2 on Oct 5, 2020
"""

import os
from PIL import Image
import torch
from torch.autograd import Variable
import numpy as np
import math

FORWARD_BZ = 5000


def forward_batch(model, input, batchsize):
    inputsize = input.shape[0]
    for count in range((inputsize - 1) // batchsize + 1):
        end = min(inputsize, (count + 1) * batchsize)
        if count == 0:
            tempinput = input[count * batchsize : end]
            out = model(tempinput.cuda())
            out = out.data.cpu().numpy()
        else:
            tempinput = input[count * batchsize : end]
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
        raise f"pert {pert} is not supported."


def get_exp(ind, exp):
    return exp[ind.astype(int)]


def get_exp_infid(image, model, exp, label, pdt, binary_I, pert):
    point = 260 * 260
    total = np.prod(exp.shape)
    num = 100

    exp = np.squeeze(exp)
    exp_copy = np.reshape(np.copy(exp), -1)
    image_copy = np.tile(np.reshape(np.copy(image.cpu()), (1, 3, 260 * 260)), [num, 1, 1])

    image_copy_ind = np.apply_along_axis(set_zero_infid, 2, image_copy, total, point, pert)

    if pert == "Gaussian" and not binary_I:
        image_copy = image_copy_ind[:, :, :total]
        ind = image_copy_ind[:, :, total : total + point]
        rand = image_copy_ind[:, :, total + point : total + 2 * point]
        exp_sum = np.sum(rand * np.apply_along_axis(get_exp, 1, ind, exp_copy), axis=2)
        ks = np.ones(num)
    else:
        raise ValueError("Perturbation type and binary_I do not match.")

    image_copy = np.reshape(image_copy, (num, 3, 260, 260))
    image_v = Variable(torch.from_numpy(image_copy.astype(np.float32)).cuda(), requires_grad=False)
    out = forward_batch(model, image_v, FORWARD_BZ)
    pdt_rm = out[:, label]
    pdt_diff = np.squeeze(pdt - pdt_rm)
    exp_sum = np.mean(exp_sum, axis=1)

    # performs optimal scaling for each explanation before calculating the infidelity score
    beta = np.mean(ks * pdt_diff * exp_sum) / np.mean(ks * exp_sum * exp_sum)
    exp_sum *= beta
    infid = np.mean(ks * np.square(pdt_diff - exp_sum)) / np.mean(ks)
    return infid


def get_exp_sens(X, model, expl, yy, sen_r, sen_N, norm):
    max_diff = -math.inf
    for _ in range(sen_N):
        sample = torch.FloatTensor(sample_eps_Inf(X.cpu().numpy(), sen_r, 1)).cuda()
        X_noisy = X + sample
        _ = model(X_noisy, sens=yy.item())
        expl_eps = np.array(Image.open("noisy.png").resize((260, 260), resample=Image.BILINEAR), dtype=np.uint8)

        max_diff = max(max_diff, np.linalg.norm(expl - expl_eps)) / norm
    return max_diff


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

        X_all = batch["image"].cuda()    # (B,3,260,260)
        y_all = batch["label"].cuda()    # (B,)
        names = batch["names"]           # 길이 B 리스트

        # 클래스 보정(negative-SCOUTER)
        if loss_status < 0:
            y_all = torch.tensor([lsc_dict[str(y.item())] for y in y_all],
                                 device=y_all.device)

        # ---- 이미지별로 루프 --------------------------------------
        for img, yy, fname in zip(X_all, y_all, names):
            # ① 예측 확률
            with torch.no_grad():
                pdt_val = model(img.unsqueeze(0))[:, yy].cpu().numpy()  # shape (1,)

            # ② 대응 saliency 파일 읽기
            base_id = os.path.basename(fname)  # ex) 0054_0.png
            expl = np.array(
                Image.open(os.path.join(exp_path, base_id))
                     .resize((260, 260), resample=Image.BILINEAR),
                dtype=np.uint8
            )
            norm = np.linalg.norm(expl)

            # ③ Infidelity / Sensitivity
            infid = get_exp_infid(img, model, expl, yy, pdt_val,
                                  binary_I=binary_I, pert=pert)
            sens  = get_exp_sens(img, model, expl, yy,
                                 sen_r, sen_N, norm)

            infids.append(infid)
            max_sens.append(sens)

    return float(np.mean(infids)), float(np.mean(max_sens))