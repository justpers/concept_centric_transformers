import os, glob, shutil
from PIL import Image
from pathlib import Path

SRC_ROOT = Path("./embryo")          # 원본 데이터 루트
DST_ROOT = Path("./rgb_only")        # 복사본 저장 루트
DST_ROOT.mkdir(exist_ok=True)

exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
rgb_list = []                        # 3채널 목록 기록

for split in ("train", "val", "test"):
    for cls in ("failure", "success"):
        dir_path = SRC_ROOT / split / cls
        for ext in exts:
            for fp in dir_path.glob(ext):
                try:
                    mode = Image.open(fp).mode   # 'L'·'RGB'·'RGBA' 등
                except Exception as e:
                    print("열기 실패:", fp, e)
                    continue

                if mode == "RGB":
                    rgb_list.append(fp)
                    # 동일한 하위 구조 유지하며 복사
                    dst_fp = DST_ROOT / split / cls / fp.name
                    dst_fp.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(fp, dst_fp)

print(f"총 이미지  : {sum(len(list((SRC_ROOT/s).glob('**/*.*'))) for s in ('train','val','test'))}")
print(f"RGB(3채널) : {len(rgb_list)}  → {DST_ROOT} 에 복사 완료")