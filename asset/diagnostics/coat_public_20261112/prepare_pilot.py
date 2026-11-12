"""Materialize and verify the pilot dataset without editing the frozen parent split."""
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[2]
PARENT = ROOT / "datasets/laboratory_safety_upperbody_20261028"
DATA = ROOT / "datasets/laboratory_safety_public_coat_20261112"
sys.path.insert(0, str(ROOT))
from scripts.dataset_audit import source_key

SEED = 42
REPLAY_ORIGINALS = 660
REPLAY_CROPS_PER_CLASS = 120
REPLAY_VAL = 240
PUBLIC_REPEAT = 6


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def phash(image):
    small = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    low = cv2.dct(small)[:8, :8].flatten()
    return int.from_bytes(np.packbits(low > np.median(low[1:])).tobytes(), "big")


def relative(file):
    return file.relative_to(ROOT).as_posix()


def corresponding_label(path):
    head, tail = path.as_posix().rsplit("/images/", 1)
    return Path(head + "/labels/" + tail).with_suffix(".txt")


def review_boxes(item, boxes):
    resolved = []
    for box in boxes:
        if isinstance(box, int):
            label = 5 if item["group"] == "lab" else 9
            proposal = item["boxes"][box]
            if proposal["label"] == -1:
                raise ValueError("Full-person proposals require a manual garment box")
            xyxy = proposal["xyxy"]
        else:
            label, *xyxy = box
        x1, y1, x2, y2 = xyxy
        if label not in (5, 9) or not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(f"Invalid box on {item['id']}: {box}")
        resolved.append([label, *xyxy])
    if not resolved or len({tuple(x) for x in resolved}) != len(resolved):
        raise ValueError("Missing or duplicate reviewed boxes on " + item["id"])
    return resolved


def grouped_sample(paths, limit, rng):
    groups = defaultdict(list)
    for path in paths:
        groups[source_key(path)].append(path)
    queues = []
    for group in groups.values():
        rng.shuffle(group)
        queues.append(deque(group))
    rng.shuffle(queues)
    output = []
    while len(output) < limit and queues:
        for queue in queues:
            if len(output) == limit:
                break
            output.append(queue.popleft())
        queues = [queue for queue in queues if queue]
    assert len(output) == limit
    return output


def main():
    if DATA.exists():
        raise FileExistsError(DATA)
    catalog = {item["id"]: item for item in json.loads((FOLDER / "proposals.json").read_text(encoding="utf-8"))}
    review = json.loads((FOLDER / "reviewed_boxes.json").read_text(encoding="utf-8"))
    download_manifest = FOLDER / "high_resolution_sources.json"
    high_resolution = json.loads(download_manifest.read_text(encoding="utf-8")) if download_manifest.exists() else {}
    old_hashes = set(json.loads((PARENT / "image_manifest.json").read_text(encoding="utf-8")).values())
    old_phash = [int(h, 16) for h in json.loads((FOLDER / ".original_phash.json").read_text(encoding="utf-8")).values()]
    test_phash = []
    for name in (FOLDER / ".holdout_frames").glob("*.jpg"):
        with Image.open(name) as image:
            test_phash.append(phash(image))
    if not test_phash:
        raise RuntimeError("Video holdout fingerprints are missing")
    selected = {"train": [], "val": []}
    author_splits = {}
    media_fingerprints = set()
    accepted_phash = []
    source_records = []
    for split, items in review.items():
        for key, boxes in items.items():
            item = catalog[key]
            labels = review_boxes(item, boxes)
            author_group = item["author"].casefold().strip()
            if not author_group:
                print("Missing creator attribution; skipped", key, flush=True)
                continue
            if author_group in author_splits and author_splits[author_group] != split:
                raise ValueError("Author group straddles splits: " + author_group)
            preview = FOLDER / "previews" / (key + ".jpg")
            material = FOLDER / "high_resolution" / (key + ".jpg")
            if not material.is_file():
                material = preview
            with Image.open(preview) as reviewed:
                aspect = reviewed.width / reviewed.height
            with Image.open(material) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
            if abs(image.width / image.height / aspect - 1) > 0.025:
                material = preview
                with Image.open(preview) as reviewed:
                    image = reviewed.convert("RGB")
            fingerprint = phash(image)
            nearest = min(bin(fingerprint ^ other).count("1") for other in old_phash + test_phash + accepted_phash)
            if sha(material) in old_hashes or item["commons_sha1"] in media_fingerprints or nearest <= 7:
                print("Duplicate candidate skipped", key, "pHash distance", nearest, flush=True)
                continue
            author_splits[author_group] = split
            media_fingerprints.add(item["commons_sha1"])
            accepted_phash.append(fingerprint)
            image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            image_path = DATA / "images" / f"public_{split}" / (key + ".jpg")
            label_path = corresponding_label(image_path)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(image_path, quality=94)
            yolo_lines = []
            for label, x1, y1, x2, y2 in labels:
                yolo_lines.append(f"{label} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}")
            label_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
            selected[split].append(image_path)
            source_records.append({
                **{k: item[k] for k in ("id", "title", "file_page", "url", "thumb", "author", "credit", "license",
                                         "license_url", "commons_sha1", "group")},
                "split": split, "image": relative(image_path), "boxes_xyxy": labels,
                "source_variant": material.parent.name, "review_copy_sha256": sha(material),
                "higher_resolution_url": high_resolution.get(key) if material.parent.name == "high_resolution" else None,
                "phash_min_distance": nearest, "dimensions": list(image.size),
                "modifications": "EXIF orientation normalization, optional resize and JPEG re-encoding; no crop",
                "annotation": "clothing only, visually reviewed proposals and manual corrections",
            })
    for split in selected:
        targets = Counter(label[0] for r in source_records if r["split"] == split for label in r["boxes_xyxy"])
        assert set(targets) == {5, 9}
        print(split, len(selected[split]), dict(targets), flush=True)

    rng = random.Random(SEED)
    old_train = [Path(x) for x in (PARENT / "train.txt").read_text(encoding="utf-8").splitlines()]
    old_val = [Path(x) for x in (PARENT / "val.txt").read_text(encoding="utf-8").splitlines()]
    old_test = [Path(x) for x in (PARENT / "test.txt").read_text(encoding="utf-8").splitlines()]
    originals = grouped_sample([p for p in old_train if PARENT not in p.parents], REPLAY_ORIGINALS, rng)
    crops = [
        path for label in (5, 9)
        for path in rng.sample([p for p in old_train if p.name.startswith(f"upperbody_c{label}_")], REPLAY_CROPS_PER_CLASS)
    ]
    replay = originals + crops
    classes = Counter(int(line.split()[0]) for path in replay
                      for line in corresponding_label(path).read_text(encoding="utf-8").splitlines() if line.strip())
    assert set(classes) == set(range(12)), classes
    training = selected["train"] * PUBLIC_REPEAT + replay
    validation = grouped_sample(old_val, REPLAY_VAL, rng)
    assert not set(training) & (set(validation) | set(old_test) | set(selected["val"]))
    assert not set(validation) & set(old_test)
    for split, paths in (("train", training), ("val", validation), ("test", old_test), ("public_val", selected["val"])):
        (DATA / f"{split}.txt").write_text("\n".join(path.as_posix() for path in paths) + "\n", encoding="utf-8")
    yaml = (PARENT / "data.yaml").read_text(encoding="utf-8").splitlines()
    yaml[0] = "path: " + DATA.as_posix()
    (DATA / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    (DATA / "public_sources.json").write_text(json.dumps(source_records, ensure_ascii=False, indent=2), encoding="utf-8")
    reviewed_files = list(dict.fromkeys(training + validation + selected["val"] + old_test))
    manifest = {
        "seed": SEED, "full_labels_training_images": len(replay), "full_labels_val_images": len(validation),
        "frozen_test_images": len(old_test), "public_train_unique": len(selected["train"]),
        "public_train_repeats_per_image": PUBLIC_REPEAT, "public_val_unique": len(selected["val"]),
        "replay_training_class_counts": dict(classes), "sources": source_records,
        "verified_files": {name: sha(DATA / name) for name in ("data.yaml", "train.txt", "val.txt", "test.txt", "public_val.txt", "public_sources.json")},
        "images": {relative(path): sha(path) for path in reviewed_files},
        "labels": {relative(corresponding_label(path)): sha(corresponding_label(path)) for path in reviewed_files},
    }
    (DATA / "verification.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Train exposures:", len(training), "val:", len(validation), "frozen test:", len(old_test), flush=True)


if __name__ == "__main__":
    main()
