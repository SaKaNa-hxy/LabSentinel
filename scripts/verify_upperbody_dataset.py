"""Independently verify crop geometry, files, frozen membership and provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
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
PARENT = ROOT/"datasets/laboratory_safety"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify(output):
    output = output.resolve()
    records = json.loads((output/"crop_manifest.json").read_text(encoding="utf-8"))
    parent = json.loads((output/"parent_manifest.json").read_text(encoding="utf-8"))
    labels = json.loads((output/"label_manifest.json").read_text(encoding="utf-8"))
    splits = {key: [Path(p) for p in (output/f"{key}.txt").read_text(encoding="utf-8").splitlines() if p] for key in ("train", "val", "test")}
    original_train = [Path(p) for p in (PARENT/"train.txt").read_text(encoding="utf-8").splitlines() if p]
    crops = [Path(item["image"]) for item in records]
    check(len(crops) == len(set(crops)), "Duplicate crop membership")
    check(splits["train"] == original_train+crops, "Training membership changed unexpectedly")
    for key in ("val", "test"):
        check((output/f"{key}.txt").read_bytes() == (PARENT/f"{key}.txt").read_bytes(), f"Changed frozen {key} list")
    all_paths = [p for values in splits.values() for p in values]
    check(len(all_paths) == len(set(all_paths)), "Image appears across splits")
    for path, expected in parent["metadata"].items():
        check(digest(ROOT/path) == expected, f"Changed parent metadata: {path}")
    all_image_hashes = {Path(p).relative_to(ROOT).as_posix(): sha for p, sha in parent["image_sha256"].items()}
    all_image_hashes.update({Path(item["image"]).relative_to(ROOT).as_posix(): item["image_sha256"] for item in records})
    expected_labels = {**parent["label_sha256"], **{Path(item["label"]).relative_to(ROOT).as_posix(): item["label_sha256"] for item in records}}
    check(labels == expected_labels, "Incorrect complete label manifest")
    check(set(all_image_hashes) == {p.relative_to(ROOT).as_posix() for p in all_paths}, "Incorrect complete image membership")
    print(f"Verifying {len(all_image_hashes)} image hashes and {len(labels)} label hashes...", flush=True)
    def verify_file(item):
        name, expected = item
        check(digest(ROOT/name) == expected, f"Changed dataset file: {name}")
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(verify_file, list(all_image_hashes.items())+list(labels.items())))
    target_counts, crop_counts = Counter(), Counter()
    train_set = set(original_train)
    held_out_groups = {source_key(p) for key in ("val", "test") for p in splits[key]}

    def verify_crop(item):
        image, source = Path(item["image"]), Path(item["source_image"])
        check(source in train_set, "Crop comes from outside the actual training list")
        check(item["source_group"] == source_key(source) and item["source_group"] not in held_out_groups, "Crop group leaks into held-out data")
        left, top, right, bottom = item["crop_xyxy"]
        width, height = right-left, bottom-top
        source_w, source_h = item["source_size"]
        check(0 <= left < right <= source_w and 0 <= top < bottom <= source_h, "Crop outside source image")
        with Image.open(image) as decoded:
            decoded.load()
            check(decoded.size == (width, height), "Decoded crop dimensions differ")
        # Reconstruct visible source boxes independently, then compare every
        # normalized output row, including non-garment classes and removed boxes.
        original_label = Path(str(source).replace("\\images\\", "\\labels\\")).with_suffix(".txt")
        expected = []
        for index, line in enumerate(original_label.read_text(encoding="utf-8-sig").splitlines()):
            if not line.strip():
                continue
            cls, cx, cy, bw, bh = map(float, line.split())
            x1, y1 = max(left, (cx-bw/2)*source_w), max(top, (cy-bh/2)*source_h)
            x2, y2 = min(right, (cx+bw/2)*source_w), min(bottom, (cy+bh/2)*source_h)
            w, h = x2-x1, y2-y1
            if min(w, h) <= 0 or min(w, h) < 3 or w*h < 16:
                continue
            area_fraction = (w*h)/(bw*source_w*bh*source_h)
            minimum = 0.98 if cls in (0, 1) else (0.40 if cls in (5, 9) else 0.65)
            check(area_fraction >= minimum-1e-8, "Ambiguous partial object in accepted crop")
            expected.append([int(cls), ((x1+x2)/2-left)/width, ((y1+y2)/2-top)/height, w/width, h/height])
        actual = [list(map(float, line.split())) for line in Path(item["label"]).read_text(encoding="utf-8").splitlines() if line.strip()]
        check(len(actual) == len(expected) > 0, "Missing or extra transformed box")
        check(len({tuple(row) for row in actual}) == len(actual), "Duplicate output boxes")
        for actual_row, expected_row in zip(actual, expected):
            check(len(actual_row) == 5 and all(abs(a-b) <= 1e-7 for a, b in zip(actual_row, expected_row)), "Incorrect transformed coordinates")
            cls, cx, cy, bw, bh = actual_row
            check(cls == int(cls) and 0 <= cls < 12 and min(bw, bh) > 0 and cx-bw/2 >= -1e-7 and cy-bh/2 >= -1e-7 and cx+bw/2 <= 1+1e-7 and cy+bh/2 <= 1+1e-7, "Invalid crop label")
        return Counter(int(row[0]) for row in actual)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, (item, counts) in enumerate(zip(records, pool.map(verify_crop, records)), 1):
            target_counts[item["target_class"]] += 1
            check(counts[item["target_class"]] > 0, "Primary target missing")
            crop_counts.update(counts)
            if index % 500 == 0 or index == len(records):
                print(f"Decoded and independently checked {index}/{len(records)} crops", flush=True)
    check(target_counts[5] == target_counts[9] > 0, "Unbalanced primary target selection")
    stats = {}
    for split, images in splits.items():
        counts = Counter()
        for image in images:
            path = Path(str(image).replace("\\images\\", "\\labels\\")).with_suffix(".txt")
            counts.update(int(line.split()[0]) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip())
        stats[split] = {"images": len(images), "objects": [counts[i] for i in range(12)]}
    audit = json.loads((output/"audit.json").read_text(encoding="utf-8"))
    for split in stats:
        check(all(stats[split][key] == audit["grouped"]["splits"][split][key] for key in ("images", "objects")), "Audit statistics mismatch")
    (output/"image_manifest.json").write_text(json.dumps(all_image_hashes, indent=2), encoding="utf-8")
    important = ("data.yaml", "train.txt", "val.txt", "test.txt", "label_manifest.json", "image_manifest.json", "annotation_version.json", "audit.json", "crop_manifest.json", "parent_manifest.json")
    report = {"state": "verified", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "generated_crops": len(records), "labels_checked": len(labels), "images_hashed": len(all_image_hashes),
              "all_crops_decoded": True, "all_crop_coordinates_independently_checked": True,
              "original_images_and_labels_unchanged": True, "frozen_val_test_lists_byte_identical": True,
              "train_only_provenance_verified": True, "overlapping_parent_filename_groups": 0,
              "target_class_counts": dict(target_counts), "generated_object_counts": dict(crop_counts), "split_stats": stats,
              "semantic_review_scope": "Geometric validation for all crops; visual spot check recorded separately. No claim that all source labels are semantically certified.",
              "verified_dataset_files_sha256": {name: digest(output/name) for name in important}, "verifier_script_sha256": digest(__file__)}
    (output/"verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    status = json.loads((output/"build_status.json").read_text(encoding="utf-8"))
    status["state"] = "verified"
    (output/"build_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("state", "generated_crops", "labels_checked", "images_hashed", "split_stats")}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    verify(parser.parse_args().dataset)
