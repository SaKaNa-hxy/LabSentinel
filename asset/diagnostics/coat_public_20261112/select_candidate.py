"""Select one pilot checkpoint using source-separated public validation only."""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
from ultralytics import YOLO
from app.backend.inference import resolve_coat_conflicts

DATA = ROOT / "datasets/laboratory_safety_public_coat_20261112"
RUN = ROOT / "runs/laboratory_safety_public_coat_20261112"
BASE = ROOT / "models/laboratory_safety/weights/best.pt"


def iou(left, right):
    a, b, c, d = left
    e, f, g, h = right
    overlap = max(0, min(c, g) - max(a, e)) * max(0, min(d, h) - max(b, f))
    return overlap / max(1e-9, (c-a)*(d-b) + (g-e)*(h-f) - overlap)


def evaluate(path, records):
    model = YOLO(str(path))
    by_group = defaultdict(Counter)
    rows = []
    for item in records:
        with Image.open(ROOT / item["image"]) as image:
            prediction = model.predict(image.convert("RGB"), imgsz=640, conf=0.15, max_det=300, device=0, verbose=False)[0]
        candidates = [
            {"class_id": int(box.cls.item()), "class_name": "lab_coat" if int(box.cls.item()) == 5 else "no_lab_coat",
             "bbox": box.xyxyn[0].tolist(), "confidence": float(box.conf.item())}
            for box in prediction.boxes if int(box.cls.item()) in (5, 9)
        ]
        indices, _ = resolve_coat_conflicts(candidates)
        kept = [candidates[i] for i in indices]
        for index, (truth, *target_box) in enumerate(item["boxes_xyxy"], 1):
            matches = {box["class_id"] for box in kept if iou(box["bbox"], target_box) >= 0.5}
            state = "no_box" if not matches else "both" if len(matches) > 1 else "correct" if truth in matches else "wrong"
            by_group[item["group"]][state] += 1
            rows.append({"image": item["id"], "target": index, "truth": truth, "state": state})
    groups = {name: dict(by_group[name]) for name in ("lab", "shirt", "outer") if by_group[name]}
    rates = [counts.get("correct", 0) / sum(counts.values()) for counts in groups.values()]
    return {
        "checkpoint": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "counts": groups, "balanced_accuracy": sum(rates)/len(rates),
        "wrong_or_both": sum(group.get("wrong", 0) + group.get("both", 0) for group in groups.values()),
        "targets": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--output-dir", type=Path, default=FOLDER)
    args = parser.parse_args()
    data_dir, run_dir, output_dir = (p.resolve() for p in (args.data_dir, args.run_dir, args.output_dir))
    manifest = json.loads((data_dir / "verification.json").read_text(encoding="utf-8"))
    records = [row for row in manifest["sources"] if row["split"] == "val"]
    weights = sorted((run_dir / "weights").glob("epoch*.pt"), key=lambda p: int(p.stem[5:]))
    assert weights
    results = {"baseline": evaluate(BASE, records)}
    for path in weights:
        label = f"epoch_{int(path.stem[5:])+1}"
        result = evaluate(path, records)
        results[label] = result
        print(label, result["counts"], result["balanced_accuracy"], flush=True)
    ranked = sorted(
        (name for name in results if name != "baseline"),
        key=lambda name: (-results[name]["balanced_accuracy"], results[name]["wrong_or_both"], int(name.split("_")[1])),
    )
    best = ranked[0]
    output = {
        "rule": "Highest average per-garment recall at IoU >= 0.50, then fewest matched wrong or contradictory labels, then earliest epoch",
        "imgsz": 640, "confidence": 0.15, "selected": best, "results": results,
    }
    (output_dir / "public_validation.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("Selected", best, results[best]["counts"], "baseline", results["baseline"]["counts"], flush=True)
    stress = [row for row in manifest["sources"] if row["split"] == "stress"]
    if stress:
        separate = {
            "selection_independent": True,
            "baseline": evaluate(BASE, stress),
            "candidate": evaluate(ROOT / results[best]["checkpoint"], stress),
        }
        (output_dir / "public_lab_stress.json").write_text(json.dumps(separate, indent=2), encoding="utf-8")
        print("Held-out lab coat images", len(stress), "baseline", separate["baseline"]["counts"],
              "candidate", separate["candidate"]["counts"], flush=True)


if __name__ == "__main__":
    main()
