"""Compare garment-only second opinions on validation data, then score a fixed rule."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
SCALES = ("small", "large", "older", "beforecrop", "previous", "public")


def iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return overlap / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap)


def resolve(boxes):
    positives = [box for box in boxes if box[0] == 5]
    return [box for box in boxes if box[0] == 5 or all(iou(box[2:], p[2:]) < 0.7 for p in positives)]


def opinion(anchor, boxes, cls, threshold, margin, overlap):
    same_place = [box for box in boxes if iou(box[2:], anchor[2:]) >= overlap]
    score = max((box[1] for box in same_place if box[0] == cls), default=0.0)
    other = max((box[1] for box in same_place if box[0] != cls), default=0.0)
    return score if score >= threshold and score - other >= margin else 0.0


def revise(row, source, threshold, margin, overlap, reverse_source=None, reverse_threshold=0.0):
    boxes = resolve(row["base"])
    out = []
    for box in boxes:
        if box[0] == 9:
            score = opinion(box, row[source], 5, threshold, margin, overlap)
            out.append([5, score, *box[2:]] if score else box)
        elif reverse_source:
            score = opinion(box, row[reverse_source], 9, reverse_threshold, margin, overlap)
            out.append([9, score, *box[2:]] if score else box)
        else:
            out.append(box)
    return out


def count(labels, boxes, counts):
    pairs = []
    for i, target in enumerate(labels):
        for j, box in enumerate(boxes):
            score = iou(target[1:], box[2:])
            if score >= 0.5:
                pairs.append((score, i, j))
    targets, assigned = {}, set()
    for _, i, j in sorted(pairs, reverse=True):
        if i not in targets and j not in assigned:
            targets[i] = boxes[j][0]
            assigned.add(j)
    for i, label in enumerate(labels):
        counts[f"{label[0]}>{targets.get(i, 'none')}"] += 1
    for j, box in enumerate(boxes):
        if j not in assigned:
            counts[f"none>{box[0]}"] += 1


def evaluate(rows, rule):
    counts = defaultdict(Counter)
    for row in rows:
        boxes = rule(row)
        if row["split"] != "video":
            count(row["labels"], boxes, counts[row["split"]])
            continue
        name = Path(row["image"]).stem.split("-")[0]
        truth = 5 if name in {"161840", "222722"} else 9
        labels = {b[0] for b in boxes}
        counts["video_" + name]["correct" if labels == {truth} else
                                   "both" if labels == {5, 9} else "wrong" if labels else "none"] += 1
    return {k: dict(v) for k, v in counts.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--variant", choices=SCALES)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--overlap", type=float, default=0.7)
    parser.add_argument("--reverse-source", choices=SCALES)
    parser.add_argument("--reverse-threshold", type=float, default=0.65)
    args = parser.parse_args()
    rows = json.loads((DIRECTORY / "evaluation_probes.json").read_text(encoding="utf-8"))
    if not args.frozen:
        rows = [r for r in rows if r["split"] != "frozen"]
    elif not args.variant:
        parser.error("Choose a rule before opening frozen results")
    print("Baseline", evaluate(rows, lambda r: resolve(r["base"])))
    if args.variant:
        print("Second opinion", evaluate(rows, lambda r: revise(
            r, args.variant, args.threshold, args.margin, args.overlap, args.reverse_source, args.reverse_threshold)))
        return
    for variant in SCALES:
        for threshold in (0.5, 0.65, 0.75, 0.85):
            print(variant, threshold, evaluate(rows, lambda r: revise(r, variant, threshold, 0.1, 0.7)))


if __name__ == "__main__":
    main()
