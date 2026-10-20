"""Prepare manual frame-event review and score it against one checkpoint.

The annotation cell is blank until reviewed. Enter pipe-separated event keys,
or `none` when the frame contains no supported safety event.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.inference import CLASS_NAME_MAP
from app.backend.risk_engine import analyze_risk
from scripts.dataset_audit import OUTPUT, source_key
from scripts.evaluate_provenance import ROOT, sha256

EVENT_KEYS = {"drinking", "eating", "no_gloves", "no_head_mask", "no_lab_coat", "no_mask", "no_goggles"}
DEFAULT_CSV = ROOT / "asset" / "event_review.csv"


def prepare(destination: Path, count: int, seed: int) -> None:
    rng = random.Random(seed)
    groups = defaultdict(list)
    for line in (OUTPUT / "test.txt").read_text(encoding="utf-8").splitlines():
        path = Path(line)
        groups[source_key(path)].append(path)
    for paths in groups.values():
        rng.shuffle(paths)
    sources = list(groups)
    rng.shuffle(sources)
    selected = []
    while len(selected) < count:
        added = False
        for source in sources:
            if groups[source] and len(selected) < count:
                selected.append((groups[source].pop(), source))
                added = True
        if not added:
            break
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image_path", "source", "true_events", "notes"))
        writer.writeheader()
        writer.writerows({"image_path": str(path), "source": source, "true_events": "", "notes": ""} for path, source in selected)
    print(f"Prepared {len(selected)} frames from {len({source for _, source in selected})} sources: {destination}")


def score(review_csv: Path, model_path: Path, confidence: float, device: str) -> Path:
    model = YOLO(str(model_path))
    counts = {key: Counter() for key in sorted(EVENT_KEYS)}
    reviewed = 0
    with review_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        raw = row["true_events"].strip()
        if not raw:
            continue
        truth = set() if raw.lower() == "none" else {part.strip() for part in raw.split("|") if part.strip()}
        if truth - EVENT_KEYS:
            raise ValueError(f"Unsupported event key in {row['image_path']}: {truth - EVENT_KEYS}")
        result = model.predict(source=row["image_path"], conf=confidence, device=device, verbose=False)[0]
        detections = []
        for class_id in result.boxes.cls.int().tolist():
            name = str(result.names[class_id])
            detections.append({"class_name": CLASS_NAME_MAP.get(name, name.lower().replace(" ", "_"))})
        predicted = {event["key"] for event in analyze_risk(detections)["events"]}
        for key in EVENT_KEYS:
            counts[key]["tp" if key in truth and key in predicted else "fp" if key in predicted else "fn" if key in truth else "tn"] += 1
        reviewed += 1
    if not reviewed:
        raise ValueError("No reviewed rows. Fill true_events with event keys or `none` first.")
    total = Counter()
    per_event = {}
    for key, counter in counts.items():
        total.update(counter)
        precision = counter["tp"] / max(counter["tp"] + counter["fp"], 1)
        recall = counter["tp"] / max(counter["tp"] + counter["fn"], 1)
        per_event[key] = {"tp": counter["tp"], "fp": counter["fp"], "fn": counter["fn"], "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12)}
    precision = total["tp"] / max(total["tp"] + total["fp"], 1)
    recall = total["tp"] / max(total["tp"] + total["fn"], 1)
    report = {
        "reviewed_frames": reviewed,
        "model_sha256": sha256(model_path),
        "review_csv_sha256": sha256(review_csv),
        "confidence_threshold": confidence,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "per_event": per_event,
        "note": "Frame-level manually reviewed events; temporal alert metrics require annotated videos.",
    }
    destination = review_csv.with_name(review_csv.stem + "_score.json")
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_CSV)
    prepare_parser.add_argument("--count", type=int, default=120)
    prepare_parser.add_argument("--seed", type=int, default=42)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--review", type=Path, default=DEFAULT_CSV)
    score_parser.add_argument("--model", type=Path, required=True)
    score_parser.add_argument("--conf", type=float, default=0.25)
    score_parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output, args.count, args.seed)
    else:
        score(args.review, args.model, args.conf, args.device)


if __name__ == "__main__":
    main()
