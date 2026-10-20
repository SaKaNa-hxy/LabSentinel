"""Audit filename groups and create a reproducible split with disjoint groups.

The original images and labels are never moved. Ultralytics reads the generated
image lists and finds each label through the existing images/labels paths.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "laboratory_safety"
OUTPUT = ROOT / "datasets" / "laboratory_safety_grouped"
SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def source_key(path: Path) -> str:
    """Group images by naming patterns; actual capture sources are unverified.

    The legacy ``video:`` key denotes an MP4-like filename prefix only.
    It does not establish that video files exist or share a capture source.
    """
    original = re.sub(r"\.rf\.[^.]+$", "", path.stem, flags=re.IGNORECASE)
    original = re.sub(r"_(?:jpg|jpeg|png)$", "", original, flags=re.IGNORECASE)
    video = re.match(r"^(.*?_MP4)-\d+$", original, flags=re.IGNORECASE)
    return ("video:" + video.group(1)) if video else ("image:" + original)


def read_samples() -> list[dict]:
    samples = []
    for split in SPLITS:
        for image in sorted((DATASET / "images" / split).iterdir()):
            if image.suffix.lower() not in IMAGE_EXTS:
                continue
            label = DATASET / "labels" / split / (image.stem + ".txt")
            if not label.is_file():
                raise FileNotFoundError(f"Missing label: {label}")
            classes = Counter()
            for line in label.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    fields = line.split()
                    if len(fields) != 5:
                        raise ValueError(f"Expected detection label in {label}: {line}")
                    class_id = int(fields[0])
                    if not 0 <= class_id < 12:
                        raise ValueError(f"Invalid class {class_id} in {label}")
                    classes[class_id] += 1
            samples.append({"path": image, "original_split": split, "source": source_key(image), "classes": classes})
    return samples


def summarize(samples: list[dict], assignment: dict[str, str] | None = None) -> dict:
    counts = {split: {"images": 0, "objects": [0] * 12, "sources": set()} for split in SPLITS}
    for item in samples:
        split = assignment[item["source"]] if assignment else item["original_split"]
        counts[split]["images"] += 1
        counts[split]["sources"].add(item["source"])
        for class_id, count in item["classes"].items():
            counts[split]["objects"][class_id] += count
    overlap = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap[f"{left}_{right}"] = len(counts[left]["sources"] & counts[right]["sources"])
    return {
        "splits": {key: {"images": value["images"], "objects": value["objects"], "sources": len(value["sources"])} for key, value in counts.items()},
        "overlapping_sources": overlap,
    }


def make_grouped_assignment(samples: list[dict], seed: int) -> dict[str, str]:
    grouped = defaultdict(list)
    for item in samples:
        grouped[item["source"]].append(item)
    total = len(samples)
    group_sizes = {source: len(items) for source, items in grouped.items()}
    group_classes = {source: Counter() for source in grouped}
    for source, items in grouped.items():
        for item in items:
            group_classes[source].update(item["classes"])
    class_totals = Counter()
    for classes in group_classes.values():
        class_totals.update(classes)

    def score(sizes: dict, classes: dict) -> float:
        size_error = sum(abs(sizes[split] / (total * RATIOS[split]) - 1) for split in SPLITS) / 3
        class_error = sum(
            abs(classes[split][index] / (class_totals[index] * RATIOS[split]) - 1)
            for split in SPLITS for index in range(12) if class_totals[index]
        ) / 36
        return size_error + 2 * class_error

    best_assignment = None
    best_score = float("inf")
    for attempt in range(12):
        rng = random.Random(seed + attempt)
        sources = list(grouped)
        rng.shuffle(sources)
        sizes = dict.fromkeys(SPLITS, 0)
        classes = {split: Counter() for split in SPLITS}
        assignment = {}
        for source in sources:
            split = min(SPLITS, key=lambda key: (sizes[key] / (total * RATIOS[key]), rng.random()))
            assignment[source] = split
            sizes[split] += group_sizes[source]
            classes[split].update(group_classes[source])
        current_score = score(sizes, classes)
        for _ in range(10000):
            source = rng.choice(sources)
            old = assignment[source]
            new = rng.choice([split for split in SPLITS if split != old])
            sizes[old] -= group_sizes[source]
            sizes[new] += group_sizes[source]
            classes[old].subtract(group_classes[source])
            classes[new].update(group_classes[source])
            candidate = score(sizes, classes)
            if candidate < current_score:
                assignment[source] = new
                current_score = candidate
            else:
                sizes[old] += group_sizes[source]
                sizes[new] -= group_sizes[source]
                classes[old].update(group_classes[source])
                classes[new].subtract(group_classes[source])
        if current_score < best_score:
            best_score, best_assignment = current_score, assignment
    return best_assignment


def write_grouped_split(samples: list[dict], assignment: dict[str, str], seed: int) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        paths = sorted(str(item["path"].resolve()).replace("\\", "/") for item in samples if assignment[item["source"]] == split)
        (OUTPUT / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    names = ["Drinking", "Eating", "Gloves", "Googles", "Head Mask", "Lab Coat", "Mask", "No Gloves", "No Head Mask", "No Lab coat", "No Mask", "No googles"]
    yaml = "path: " + str(OUTPUT.resolve()).replace("\\", "/") + "\n"
    yaml += "train: train.txt\nval: val.txt\ntest: test.txt\nnc: 12\n"
    yaml += "names: " + json.dumps(names, ensure_ascii=False) + "\n"
    (OUTPUT / "data.yaml").write_text(yaml, encoding="utf-8")
    report = {"seed": seed, "grouping": "filename prefix or normalized image name", "source_verification": "unverified; filename groups do not establish actual capture sources", "original": summarize(samples), "grouped": summarize(samples, assignment)}
    (OUTPUT / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create-grouped-split", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    samples = read_samples()
    if args.create_grouped_split:
        assignment = make_grouped_assignment(samples, args.seed)
        write_grouped_split(samples, assignment, args.seed)
    else:
        print(json.dumps(summarize(samples), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
