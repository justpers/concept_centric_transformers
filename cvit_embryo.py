from argparse import ArgumentParser
import numpy as np, torch
from ctc import CTCModel, run_exp
torch.set_float32_matmul_precision('high')

DATA_NAME = "EmbryoDatamodule"  

def get_parser(parser):
    parser = ArgumentParser(
        description="Embryo dataset (pregnancy success) with explanations",
        parents=[parser], conflict_handler="resolve"
    )
    # ---- 기본 하이퍼파라미터 ----
    parser.add_argument("--learning_rate",  default=1e-4,  type=float)
    parser.add_argument("--batch_size",     default=8,     type=int)
    parser.add_argument("--max_epochs",     default=40,    type=int)
    parser.add_argument("--weight_decay",   default=1e-3,  type=float)
    parser.add_argument("--expl_lambda",    default=0.0,   type=float)
    parser.add_argument("--finetune_unfreeze_epoch", default=0, type=int)
    parser.add_argument("--disable_lr_scheduler",  action="store_true")
    parser.add_argument("--data_dir",       default="~/embryo_vit224/", type=str)
    parser.add_argument("--baseline", action="store_true", help="use baseline model")
    # ---- CCT 공통 옵션 ----
    parser.add_argument("--attention_sparsity", default=0., type=float)
    parser.add_argument("--num_workers",    default=4, type=int)
    parser.add_argument("--seed",           default=1, type=int)
    parser.add_argument("--task",           default="binary", type=str)
    parser.add_argument("--model",          default="cifar100superclass_slotcvit_sa",
                        type=str, help="ViT-Tiny + SA (2-class)")
    parser.add_argument("--no_cuda",        action="store_true")
    parser.add_argument("--gpu",            default=0, type=int)

    # 무감독 개념 슬롯 학습 파라미터 추가
    parser.add_argument("--lambda_unsup_sparse", default=0.5, type=float)
    parser.add_argument("--lambda_unsup_div",    default=0.5, type=float)
    return parser

def main():
    parser = CTCModel.get_model_args()
    parser = get_parser(parser)
    args = parser.parse_args()

    # 필수 플래그
    args.baseline = getattr(args, "baseline", False)
    args.ctc_model = args.model
    args.data_name = DATA_NAME

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # --- 학습 + 검증 ---
    model, trainer, data_module = run_exp(args)

    # 필요하면 추가 테스트
    trainer.test(model, data_module)

if __name__ == "__main__":
    main()