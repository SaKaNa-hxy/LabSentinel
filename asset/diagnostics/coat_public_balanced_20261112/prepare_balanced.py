"""Add reviewed lab-coat photos while holding the first round's step count fixed."""
import json
import random
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

FOLDER = Path(__file__).resolve().parent
FIRST = FOLDER.parent / "coat_public_20261112"
sys.path.insert(0, str(FIRST))
from prepare_pilot import DATA as FIRST_DATA, ROOT, corresponding_label, phash, relative, review_boxes, sha

DATA = ROOT / "datasets/laboratory_safety_public_balanced_20261112"


def repeat_to_count(paths, count, rng):
    if not paths:
        raise ValueError("Both clothing groups must have training examples")
    shuffled = list(paths)
    rng.shuffle(shuffled)
    repeats, remainder = divmod(count, len(shuffled))
    return shuffled * repeats + shuffled[:remainder]


def main():
    if DATA.exists():
        raise FileExistsError(DATA)
    old = json.loads((FIRST_DATA / "verification.json").read_text(encoding="utf-8"))
    assert all(sha(FIRST_DATA / filename) == expected for filename, expected in old["verified_files"].items())
    assert all(sha(ROOT / filename) == expected for filename, expected in old["images"].items())
    assert all(sha(ROOT / filename) == expected for filename, expected in old["labels"].items())
    sources = list(old["sources"])
    creator_split = {x["author"].strip().casefold(): x["split"] for x in sources}
    old_commons = {r["commons_sha1"] for r in sources}
    hashes = [int(value, 16) for value in json.loads((FIRST / ".original_phash.json").read_text(encoding="utf-8")).values()]
    holdout = sorted((FIRST / ".holdout_frames").glob("*.jpg"))
    assert len(holdout) == 28
    for photo in holdout + [ROOT / r["image"] for r in sources]:
        with Image.open(photo) as image:
            hashes.append(phash(image))
    catalog = {item["id"]: item for item in json.loads((FOLDER / "proposals.json").read_text(encoding="utf-8"))}
    decisions = json.loads((FOLDER / "reviewed_boxes.json").read_text(encoding="utf-8"))
    downloaded = json.loads((FOLDER / "high_resolution_sources.json").read_text(encoding="utf-8"))
    counts = Counter()
    for split, items in decisions.items():
        if split not in {"train", "stress"}:
            raise ValueError("Unexpected supplemental split")
        for key, entry in items.items():
            item = catalog[key]
            boxes = review_boxes(item, entry["boxes"])
            if any(box[0] != 5 for box in boxes):
                raise ValueError("The supplement is for verified lab coats only")
            creator = item["author"].casefold().strip()
            if not creator or (creator in creator_split and creator_split[creator] != split):
                print("Excluded for source overlap or missing creator", key, flush=True)
                continue
            preview = Path(item["preview_path"])
            high_res = FOLDER / "high_resolution" / (key + ".jpg")
            media = high_res if high_res.is_file() else preview
            with Image.open(preview) as reviewed, Image.open(media) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
                if abs(image.width / image.height / (reviewed.width / reviewed.height) - 1) > 0.025:
                    image = reviewed.convert("RGB")
                    media = preview
            fingerprint = phash(image)
            nearest = min(bin(fingerprint ^ other).count("1") for other in hashes)
            if item["commons_sha1"] in old_commons or nearest <= 7:
                print("Excluded duplicate", key, "distance", nearest, flush=True)
                continue
            hashes.append(fingerprint)
            old_commons.add(item["commons_sha1"])
            creator_split[creator] = split
            output = DATA / "images" / f"public_{split}" / (key + ".jpg")
            label_file = corresponding_label(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            label_file.parent.mkdir(parents=True, exist_ok=True)
            image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            image.save(output, quality=94)
            rows = [
                f"{label} {(a+c)/2:.6f} {(b+d)/2:.6f} {c-a:.6f} {d-b:.6f}"
                for label, a, b, c, d in boxes
            ]
            label_file.write_text("\n".join(rows) + "\n", encoding="utf-8")
            counts[split] += 1
            sources.append({
                **{k: item[k] for k in ("id", "title", "file_page", "url", "thumb", "author", "credit", "license",
                                         "license_url", "commons_sha1", "group")},
                "split": split, "image": relative(output), "boxes_xyxy": boxes,
                "source_variant": media.parent.name, "review_copy_sha256": sha(media),
                "higher_resolution_url": downloaded.get(key) if media == high_res else None,
                "phash_min_distance": nearest, "dimensions": list(image.size),
                "modifications": "EXIF normalization, optional resize and JPEG encoding; no crop",
                "annotation": "clothing only, visually reviewed",
                "selection_reason": entry["reason"], "round_added": 2,
            })
    if counts["train"] < 8 or counts["stress"] < 3:
        raise RuntimeError("Insufficient independent reviewed additions: " + str(counts))
    print("Accepted public additions", dict(counts), flush=True)

    first_train = [Path(line) for line in (FIRST_DATA / "train.txt").read_text(encoding="utf-8").splitlines()]
    first_public_train = {ROOT / r["image"] for r in old["sources"] if r["split"] == "train"}
    replay = [p for p in first_train if p not in first_public_train]
    assert len(replay) == old["full_labels_training_images"] == 900
    public_count = len(first_train) - len(replay)
    assert public_count == 252
    groups = {5: [], 9: []}
    for item in sources:
        if item["split"] != "train":
            continue
        labels = {box[0] for box in item["boxes_xyxy"]}
        if len(labels) != 1:
            raise ValueError("A clothing-only image mixes positive and negative labels")
        groups[next(iter(labels))].append(ROOT / item["image"])
    rng = random.Random(42)
    train = repeat_to_count(groups[5], public_count // 2, rng) + repeat_to_count(groups[9], public_count // 2, rng) + replay
    assert len(train) == len(first_train)
    for name in ("val.txt", "test.txt", "public_val.txt"):
        (DATA / name).write_bytes((FIRST_DATA / name).read_bytes())
    (DATA / "train.txt").write_text("\n".join(p.as_posix() for p in train) + "\n", encoding="utf-8")
    stress = [ROOT / x["image"] for x in sources if x["split"] == "stress"]
    assert not set(stress) & set(train)
    (DATA / "stress.txt").write_text("\n".join(p.as_posix() for p in stress) + "\n", encoding="utf-8")
    yaml = (FIRST_DATA / "data.yaml").read_text(encoding="utf-8").splitlines()
    yaml[0] = "path: " + DATA.as_posix()
    (DATA / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    (DATA / "public_sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    images, labels = dict(old["images"]), dict(old["labels"])
    for item in sources:
        if item.get("round_added") == 2:
            image = ROOT / item["image"]
            label = corresponding_label(image)
            images[relative(image)] = sha(image)
            labels[relative(label)] = sha(label)
    manifest = {
        "seed": 42, "parent_verification_sha256": sha(FIRST_DATA / "verification.json"),
        "full_labels_training_images": len(replay), "full_labels_val_images": old["full_labels_val_images"],
        "frozen_test_images": old["frozen_test_images"], "public_train_unique": sum(map(len, groups.values())),
        "public_train_class_images": {k: len(v) for k, v in groups.items()},
        "public_train_class_exposures": {5: public_count // 2, 9: public_count // 2},
        "public_val_unique": old["public_val_unique"], "new_public": dict(counts), "sources": sources,
        "images": images, "labels": labels,
        "verified_files": {name: sha(DATA / name) for name in ("data.yaml", "train.txt", "val.txt", "test.txt",
                                                              "public_val.txt", "stress.txt", "public_sources.json")},
    }
    (DATA / "verification.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Training unique public images", manifest["public_train_class_images"],
          "public exposures per class", manifest["public_train_class_exposures"], "total", len(train), flush=True)


if __name__ == "__main__":
    main()
