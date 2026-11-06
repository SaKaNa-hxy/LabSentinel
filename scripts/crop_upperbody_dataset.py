"""Build a separate, reproducible training-only upper-body crop dataset.

Original images/labels are referenced, never overwritten. All 12 classes are
transformed together. Ambiguous partial objects reject a proposed crop instead
of silently becoming unlabelled training targets.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

try:
    from .dataset_audit import source_key
except ImportError:
    from dataset_audit import source_key

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "datasets/laboratory_safety"
DEFAULT_OUTPUT = ROOT / "datasets/laboratory_safety_upperbody_20261028"
REFERENCE = ROOT / "runs/laboratory_safety_corrected/recovery_20261027/image_decode_validation.json"
NAMES = ["Drinking", "Eating", "Gloves", "Goggles", "Head Mask", "Lab Coat", "Mask", "No Gloves", "No Head Mask", "No Lab coat", "No Mask", "No goggles"]
META = ("data.yaml", "train.txt", "val.txt", "test.txt", "audit.json", "label_manifest.json", "annotation_version.json")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label_path(image):
    parts = list(Path(image).parts)
    index = parts.index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_boxes(text, width, height):
    boxes = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError("Expected YOLO detection box")
        cls, x, y, w, h = map(float, fields)
        if not all(math.isfinite(v) for v in (cls, x, y, w, h)) or cls != int(cls) or not 0 <= cls < 12 or min(w, h) <= 0:
            raise ValueError("Invalid detection label")
        box = [int(cls), (x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]
        if box[1] < -1e-3 or box[2] < -1e-3 or box[3] > width+1e-3 or box[4] > height+1e-3:
            raise ValueError("Out-of-bounds source box")
        boxes.append(box)
    return boxes


def transform_boxes(boxes, crop, target_index):
    """Return normalized labels, or reject a crop with ambiguous partial boxes."""
    left, top, right, bottom = crop
    width, height = right-left, bottom-top
    if min(width, height) <= 0:
        raise ValueError("Empty crop")
    labels, retained = [], []
    for index, (cls, x1, y1, x2, y2) in enumerate(boxes):
        a, b, c, d = max(x1, left), max(y1, top), min(x2, right), min(y2, bottom)
        w, h = c-a, d-b
        if min(w, h) <= 0:
            continue
        fraction = (w*h)/((x2-x1)*(y2-y1))
        if min(w, h) < 3 or w*h < 16:
            if index == target_index:
                return None, "target_too_small"
            # Visible slivers below 3 px are not treated as reliable objects.
            continue
        minimum = 0.98 if cls in (0, 1) else (0.40 if cls in (5, 9) else 0.65)
        if fraction < minimum:
            return None, "ambiguous_partial_object"
        labels.append([cls, ((a+c)/2-left)/width, ((b+d)/2-top)/height, w/width, h/height])
        retained.append(index)
    if target_index not in retained:
        return None, "target_not_visible"
    return labels, None


def plan_crop(boxes, target_index, width, height):
    _, x1, y1, x2, y2 = boxes[target_index]
    w, h = x2-x1, y2-y1
    if w < 45 or h < 100 or w*h < 6000:
        return None, "small_garment"
    reasons = Counter()
    for fraction in (0.65, 0.75, 0.55, 0.80):
        top = max(0, math.floor(y1-0.60*h))
        bottom = min(height, math.ceil(y1+fraction*h))
        # Keep context around shoulders and the head rather than a fabric-only crop.
        middle = (x1+x2)/2
        crop_width = max(1.50*w, 0.72*(bottom-top), 160)
        left, right = max(0, math.floor(middle-crop_width/2)), min(width, math.ceil(middle+crop_width/2))
        crop = [left, top, right, bottom]
        if right-left < 128 or bottom-top < 160 or (right-left)*(bottom-top) > width*height*0.95:
            reasons["insufficient_crop"] += 1
            continue
        labels, reason = transform_boxes(boxes, crop, target_index)
        if reason:
            reasons[reason] += 1
            continue
        return {"crop_xyxy": crop, "labels": labels, "target_index": target_index,
                "target_class": int(boxes[target_index][0]), "target_retained_height": (bottom-y1)/h}, None
    return None, reasons.most_common(1)[0][0] if reasons else "no_valid_crop"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_verified_parent():
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    label_manifest = json.loads((PARENT/"label_manifest.json").read_text(encoding="utf-8"))
    metadata = {str((PARENT/name).relative_to(ROOT)): sha256(PARENT/name) for name in META}
    splits = {name: [Path(p) for p in (PARENT/f"{name}.txt").read_text(encoding="utf-8").splitlines() if p] for name in ("train", "val", "test")}
    all_images = [p for paths in splits.values() for p in paths]
    if len(set(all_images)) != 7217 or len(label_manifest) != 7217:
        raise ValueError("Unexpected parent membership")
    groups = {name: {source_key(p) for p in paths} for name, paths in splits.items()}
    if any(groups[a] & groups[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Parent filename groups overlap")
    def verify_image(path):
        if sha256(path) != reference["image_sha256"][str(path)]:
            raise ValueError(f"Original image changed: {path}")
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(verify_image, all_images))
    def verify_label(item):
        name, expected = item
        raw = (ROOT/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Original label changed: {name}")
        return name, raw.decode("utf-8-sig")
    with ThreadPoolExecutor(max_workers=6) as pool:
        texts = dict(pool.map(verify_label, label_manifest.items()))
    return splits, reference, label_manifest, texts, metadata


def build(output, dry_run=False):
    output = output.resolve()
    if output == PARENT or PARENT in output.parents or output.parent != ROOT/"datasets":
        raise ValueError("Output must be a new sibling of the original dataset")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    print("Verifying original images, labels, and frozen split lists...", flush=True)
    splits, reference, original_labels, texts, metadata = load_verified_parent()
    plans, skipped = {5: [], 9: []}, Counter()
    for image in splits["train"]:
        height, width = reference["sizes"][str(image)]
        boxes = read_boxes(texts[label_path(image).relative_to(ROOT).as_posix()], width, height)
        for cls in (5, 9):
            targets = [i for i, box in enumerate(boxes) if box[0] == cls]
            if not targets:
                continue
            index = max(targets, key=lambda i: (boxes[i][3]-boxes[i][1])*(boxes[i][4]-boxes[i][2]))
            plan, reason = plan_crop(boxes, index, width, height)
            if reason:
                skipped[reason] += 1
                continue
            identity = hashlib.sha256(f"{image.relative_to(ROOT).as_posix()}|{cls}|upperbody_v1".encode()).hexdigest()[:24]
            plans[cls].append({**plan, "id": f"upperbody_c{cls}_{identity}", "source_image": str(image),
                               "source_sha256": reference["image_sha256"][str(image)],
                               "source_group": source_key(image), "source_size": [width, height]})
    # Equal numbers of primary coat/non-coat targets avoid positive-only augmentation.
    count = min(len(plans[5]), len(plans[9]))
    chosen = [item for cls in (5, 9) for item in sorted(plans[cls], key=lambda item: item["id"])[:count]]
    summary = {"eligible_by_target_class": {str(k): len(v) for k, v in plans.items()},
               "selected_by_target_class": {"5": count, "9": count}, "crops": len(chosen),
               "omitted_for_class_balance": {str(k): len(v)-count for k, v in plans.items()},
               "skipped_candidates": dict(skipped), "original_train": len(splits["train"]),
               "new_train": len(splits["train"])+len(chosen), "val": len(splits["val"]), "test": len(splits["test"])}
    print(json.dumps(summary, indent=2), flush=True)
    if dry_run:
        return summary
    if count == 0:
        raise RuntimeError("No balanced usable crops")
    (output/"images/train").mkdir(parents=True)
    (output/"labels/train").mkdir(parents=True)
    write_json(output/"build_status.json", {"state": "building", "summary": summary})
    def create(item):
        raw = Path(item["source_image"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["source_sha256"]:
            raise ValueError("Source image changed during generation")
        image_path = output/"images/train"/(item["id"]+".jpg")
        with Image.open(io.BytesIO(raw)) as source:
            if list(source.size) != item["source_size"]:
                raise ValueError("Image dimensions differ from source reference")
            source.convert("RGB").crop(item["crop_xyxy"]).save(image_path, quality=95, subsampling=0)
        label = label_path(image_path)
        lines = [str(row[0])+" "+" ".join(f"{value:.8f}" for value in row[1:]) for row in item["labels"]]
        if len(lines) != len(set(lines)):
            raise ValueError("Duplicate crop labels")
        label.write_text("\n".join(lines)+"\n", encoding="utf-8")
        return {**{k: v for k, v in item.items() if k != "labels"}, "image": str(image_path),
                "image_sha256": sha256(image_path), "label": str(label), "label_sha256": sha256(label),
                "class_counts": dict(Counter(int(row[0]) for row in item["labels"]))}
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, record in enumerate(pool.map(create, chosen), 1):
            records.append(record)
            if index % 250 == 0 or index == len(chosen):
                print(f"Generated {index}/{len(chosen)} crops", flush=True)
    for name in ("val", "test"):
        (output/f"{name}.txt").write_bytes((PARENT/f"{name}.txt").read_bytes())
    all_train = [p.as_posix() for p in splits["train"]]+[Path(item["image"]).as_posix() for item in records]
    (output/"train.txt").write_text("\n".join(all_train)+"\n", encoding="utf-8")
    (output/"data.yaml").write_text(f"path: {output.as_posix()}\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnc: 12\nnames: {json.dumps(NAMES)}\n", encoding="utf-8")
    manifest = {**original_labels, **{Path(item["label"]).relative_to(ROOT).as_posix(): item["label_sha256"] for item in records}}
    write_json(output/"label_manifest.json", manifest)
    write_json(output/"crop_manifest.json", records)
    write_json(output/"parent_manifest.json", {"metadata": metadata, "image_sha256": reference["image_sha256"], "label_sha256": original_labels})
    write_json(output/"annotation_version.json", {"version": "upperbody_20261028_v1", "parent_version": "coat_audit_20261027_v1",
               "label_file_count": len(manifest), "generated_crops": len(records), "scope": "Training-only deterministic upper-body crops; original dataset untouched",
               "created_at_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha256(__file__),
               "parameters": {"upper_fractions": [0.65, 0.75, 0.55, 0.80], "min_garment_width": 45, "min_garment_height": 100,
                              "minimum_retained_area": {"behavior": 0.98, "garment": 0.40, "other_ppe": 0.65}, "jpeg_quality": 95}})
    audit = json.loads((PARENT/"audit.json").read_text(encoding="utf-8"))
    audit["parent_dataset"] = str(PARENT)
    audit["augmentation_summary"] = summary
    audit["augmentation_group_basis"] = "Every generated crop inherits its original train image group; original held-out lists are byte-identical. Real capture sources remain unverified."
    train_stats = audit["grouped"]["splits"]["train"]
    train_stats["images"] += len(records)
    for item in records:
        for cls, value in item["class_counts"].items():
            train_stats["objects"][int(cls)] += value
    write_json(output/"audit.json", audit)
    # Recheck original labels/metadata after all writes; independent verification
    # additionally decodes every crop and hashes original images.
    if any(sha256(ROOT/p) != digest for p, digest in metadata.items()) or any(sha256(ROOT/p) != digest for p, digest in original_labels.items()):
        raise RuntimeError("Original dataset changed during build")
    write_json(output/"build_status.json", {"state": "generated_pending_independent_verification", "summary": summary})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    build(args.output, args.dry_run)
