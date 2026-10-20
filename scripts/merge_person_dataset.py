"""Merge person.v2i.yolov11 into datasets/laboratory_safety.

- Person class id 1 (person) -> new class id 5
- Person class id 0 (junk '0' class) is dropped
- Person bbox labels (cx cy w h) are converted to 4-point polygons
  to match the existing segmentation labels.
- Files are prefixed with 'person_' to avoid name collisions.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "person.v2i.yolov11"
DST = ROOT / "datasets" / "laboratory_safety"

PERSON_PREFIX = "person_"
PERSON_SRC_CLASS = 1
PERSON_DST_CLASS = 5

# Map source split name -> dst split name
SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "test": "test",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def bbox_to_polygon(cx: float, cy: float, w: float, h: float) -> list[float]:
    """Convert normalized bbox (cx, cy, w, h) to 4-point polygon."""
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy - h / 2
    x3, y3 = cx + w / 2, cy + h / 2
    x4, y4 = cx - w / 2, cy + h / 2
    # Clamp to [0, 1]
    return [
        max(0.0, min(1.0, v))
        for v in (x1, y1, x2, y2, x3, y3, x4, y4)
    ]


def convert_label(src_path: Path, dst_path: Path) -> int:
    """Convert a person label file to polygon format with remapped class.

    Returns the number of person objects written.
    """
    written = 0
    out_lines: list[str] = []
    for line in src_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        if class_id != PERSON_SRC_CLASS:
            # Skip junk class 0 and any unexpected classes
            continue
        try:
            cx, cy, w, h = (float(p) for p in parts[1:5])
        except ValueError:
            continue
        poly = bbox_to_polygon(cx, cy, w, h)
        coords = " ".join(f"{v:.6f}" for v in poly)
        out_lines.append(f"{PERSON_DST_CLASS} {coords}")
        written += 1

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    return written


def main() -> None:
    total_images = 0
    total_labels = 0
    total_person_objs = 0

    for src_split, dst_split in SPLIT_MAP.items():
        src_images = SRC / src_split / "images"
        src_labels = SRC / src_split / "labels"
        dst_images = DST / "images" / dst_split
        dst_labels = DST / "labels" / dst_split

        if not src_images.exists():
            print(f"[skip] {src_images} does not exist")
            continue

        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        img_files = [p for p in src_images.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        split_imgs = 0
        split_labels = 0
        split_objs = 0

        for img in sorted(img_files):
            new_stem = PERSON_PREFIX + img.stem
            dst_img = dst_images / (new_stem + img.suffix)
            shutil.copy2(img, dst_img)
            split_imgs += 1

            # Label file has same stem with .txt
            src_lbl = src_labels / (img.stem + ".txt")
            if src_lbl.exists():
                dst_lbl = dst_labels / (new_stem + ".txt")
                split_objs += convert_label(src_lbl, dst_lbl)
                split_labels += 1
            else:
                # Write an empty label file for consistency
                dst_lbl = dst_labels / (new_stem + ".txt")
                dst_lbl.write_text("", encoding="utf-8")

        print(f"[{dst_split}] images={split_imgs} labels={split_labels} person_objects={split_objs}")
        total_images += split_imgs
        total_labels += split_labels
        total_person_objs += split_objs

    print(f"\nTotal: images={total_images} labels={total_labels} person_objects={total_person_objs}")

    # Update data.yaml
    data_yaml = DST / "data.yaml"
    yaml_text = data_yaml.read_text(encoding="utf-8")
    yaml_text = yaml_text.replace("nc: 5", "nc: 6")
    yaml_text = yaml_text.replace(
        "names: ['gloves', 'goggles', 'labcoat', 'fire', 'smoke']",
        "names: ['gloves', 'goggles', 'labcoat', 'fire', 'smoke', 'person']",
    )
    data_yaml.write_text(yaml_text, encoding="utf-8")
    print("\nUpdated data.yaml -> nc: 6, added 'person' class (id 5)")


if __name__ == "__main__":
    main()
