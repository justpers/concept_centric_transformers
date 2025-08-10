from __future__ import annotations
from embryo_unsup import CCTCfg, LitCCT  # ← 경로는 레포 구조에 맞게 조정

def build_model(args):
    """
    평가용 CCT 빌더: 학습과 동일한 설정으로 LitCCT를 만들어 반환.
    eval_blasto_cct.py의 --model_entry 에서 이 함수를 호출합니다.
    """
    # 평가 시에도 학습 때와 '동일'한 하이퍼파라미터가 필요합니다.
    # (백본/슬롯타입/컨셉 수/unsup 슬롯 수 등)
    cfg = CCTCfg(
        backbone     = getattr(args, "backbone", "swin_tiny"),
        slot_type    = getattr(args, "slot_type", "qsa"),
        model_name   = getattr(args, "model_name", None),
        num_classes  = getattr(args, "num_classes", 2),
        n_unsup      = getattr(args, "n_unsup", 4),
        n_concepts   = getattr(args, "n_concepts", 0),
        n_spatial    = getattr(args, "n_spatial", 4),
        pretrained   = True,
        attn_drop    = getattr(args, "attn_drop", 0.1),
        proj_drop    = getattr(args, "proj_drop", 0.1),
    )

    model = LitCCT(
        cfg,
        freeze_stages = getattr(args, "freeze_stages", 1),
        warmup_epochs = getattr(args, "warmup_epochs", 3),
        lr            = getattr(args, "lr", 1e-4),
        weight_decay  = getattr(args, "weight_decay", 0.05),
        lambda_sparse = getattr(args, "lambda_sparse", 1.0),
        lambda_div    = getattr(args, "lambda_div", 1.0),
        n_unsup       = cfg.n_unsup,
    )

    return model