from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import shutil

ROOT = Path(r"C:\Users\HXY\Desktop\Design")
TARGET = ROOT / "datasets" / "laboratory_safety"
SOURCE = ROOT / "Safe Lab AI.v4i.yolov11"
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP = ROOT / "datasets" / f"laboratory_safety_backup_{STAMP}"
TEMP = ROOT / "datasets" / f"laboratory_safety_merged_tmp_{STAMP}"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = [
    "Drinking",
    "Eating",
    "Gloves",
    "Googles",
    "Head Mask",
    "Lab Coat",
    "Mask",
    "No Gloves",
    "No Head Mask",
    "No Lab coat",
    "No Mask",
    "No googles",
]

if not SOURCE.is_dir():
    raise RuntimeError(f"新数据集不存在: {SOURCE}")
if TARGET.exists() and BACKUP.exists():
    raise RuntimeError(f"备份目录已存在: {BACKUP}")
if TEMP.exists():
    raise RuntimeError(f"临时目录已存在: {TEMP}")


def find_image(image_dir: Path, stem: str) -> Optional[Path]:
    matches = [
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and path.stem == stem
    ]
    if len(matches) > 1:
        raise RuntimeError(f"同一标签对应多个图片: {image_dir / (stem + '.txt')}")
    return matches[0] if matches else None


def transform_label(label_path: Path) -> List[str]:
    output: List[str] = []
    for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.strip().split()
        if not fields:
            continue
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise RuntimeError(f"标签无法解析: {label_path}:{line_number}") from exc
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            raise RuntimeError(f"非法类别 ID {class_id}: {label_path}:{line_number}")
        if len(values) == 4:
            output.append(" ".join(fields))
            continue
        if len(values) < 6 or len(values) % 2:
            raise RuntimeError(f"标签既不是检测框也不是有效多边形: {label_path}:{line_number}")
        xs = values[0::2]
        ys = values[1::2]
        x_min, x_max = max(0.0, min(xs)), min(1.0, max(xs))
        y_min, y_max = max(0.0, min(ys)), min(1.0, max(ys))
        if x_max <= x_min or y_max <= y_min:
            raise RuntimeError(f"多边形面积无效: {label_path}:{line_number}")
        output.append(
            f"{class_id} {(x_min + x_max) / 2:.6f} {(y_min + y_max) / 2:.6f} "
            f"{x_max - x_min:.6f} {y_max - y_min:.6f}"
        )
    return output


def copy_split(source_split: str, target_split: str, stats: Dict[str, int]) -> None:
    source_images = SOURCE / source_split / "images"
    source_labels = SOURCE / source_split / "labels"
    target_images = TEMP / "images" / target_split
    target_labels = TEMP / "labels" / target_split
    if not source_images.is_dir() or not source_labels.is_dir():
        raise RuntimeError(f"数据集目录不完整: {SOURCE / source_split}")
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    images = sorted(path for path in source_images.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
    image_stems = {path.stem for path in images}
    for image_path in images:
        label_path = source_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise RuntimeError(f"图片没有对应标签: {image_path}")
        transformed = transform_label(label_path)
        output_stem = image_path.stem
        shutil.copy2(image_path, target_images / f"{output_stem}{image_path.suffix.lower()}")
        (target_labels / f"{output_stem}.txt").write_text(
            "\n".join(transformed) + ("\n" if transformed else ""),
            encoding="utf-8",
        )
        stats["images"] += 1
        stats["label_files"] += 1
        stats["objects"] += len(transformed)
        if not transformed:
            stats["empty_labels"] += 1

    for label_path in source_labels.glob("*.txt"):
        if label_path.stem not in image_stems:
            raise RuntimeError(f"标签没有对应图片: {label_path}")


TEMP.mkdir(parents=True, exist_ok=False)
stats = {}
for source_split, target_split in (("train", "train"), ("valid", "val"), ("test", "test")):
    stats[target_split] = {"images": 0, "label_files": 0, "objects": 0, "empty_labels": 0}
    copy_split(source_split, target_split, stats[target_split])

(TEMP / "data.yaml").write_text(
    "path: C:/Users/HXY/Desktop/Design/datasets/laboratory_safety\n"
    "train: images/train\n"
    "val: images/val\n"
    "test: images/test\n\n"
    f"nc: {len(CLASS_NAMES)}\n"
    "names: [" + ", ".join(repr(name) for name in CLASS_NAMES) + "]\n",
    encoding="utf-8",
)
(TEMP / "MERGE_INFO.txt").write_text(
    "数据集模式: 完全替换 laboratory_safety\n"
    "来源: Safe Lab AI.v4i.yolov11\n"
    "类别: " + ", ".join(f"{index}={name}" for index, name in enumerate(CLASS_NAMES)) + "\n"
    "标签处理: 分割多边形已转换为检测边界框\n"
    "旧类别: fire, smoke 已移除\n",
    encoding="utf-8",
)

if TARGET.exists():
    TARGET.rename(BACKUP)
TEMP.rename(TARGET)
print("MERGE_OK")
print(f"BACKUP={BACKUP if BACKUP.exists() else 'none'}")
print(f"FINAL={TARGET}")
for split, split_stats in stats.items():
    print(f"{split}={split_stats}")
