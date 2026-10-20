"""Evaluate one exact checkpoint and save metrics with reproducibility metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import ultralytics
from ultralytics import YOLO
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "laboratory_safety" / "weights" / "best.pt"
DEFAULT_DATA = ROOT / "datasets" / "laboratory_safety" / "data.yaml"
OUTPUT = ROOT / "asset" / "evaluations"
EXPECTED_NAMES = ["Drinking", "Eating", "Gloves", "Googles", "Head Mask", "Lab Coat", "Mask", "No Gloves", "No Head Mask", "No Lab coat", "No Mask", "No googles"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plot_metrics(report: dict, save_dir: Path) -> None:
    overall = report["overall"]
    figure, axis = plt.subplots(figsize=(8, 5), dpi=150)
    labels = ["Precision", "Recall", "F1", "mAP50", "mAP50:95"]
    values = [overall[key] for key in ("precision", "recall", "f1", "map50", "map50_95")]
    bars = axis.bar(labels, values, color="#315c4c")
    axis.set_ylim(0, 1)
    axis.bar_label(bars, fmt="%.3f")
    axis.set_title("Overall detection metrics")
    figure.tight_layout()
    figure.savefig(save_dir / "overall_metrics.png")
    plt.close(figure)

    classes = report["per_class"]
    figure, axis = plt.subplots(figsize=(11, 6), dpi=150)
    names = list(classes)
    positions = range(len(names))
    width = 0.38
    axis.bar([position - width / 2 for position in positions], [classes[name]["map50"] for name in names], width, label="mAP50")
    axis.bar([position + width / 2 for position in positions], [classes[name]["map50_95"] for name in names], width, label="mAP50:95")
    axis.set_xticks(list(positions), names, rotation=35, ha="right")
    axis.set_ylim(0, 1)
    axis.set_title("Per-class detection metrics")
    axis.legend()
    figure.tight_layout()
    figure.savefig(save_dir / "per_class_metrics.png")
    plt.close(figure)


def evaluate(model_path: Path, data_path: Path, split: str, conf: float, device: str, imgsz: int) -> Path:
    if not model_path.is_file() or not data_path.is_file():
        raise FileNotFoundError("Checkpoint or dataset YAML does not exist")
    model_hash = sha256(model_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{data_path.parent.name}_{split}_{model_hash[:12]}_{stamp}"
    model = YOLO(str(model_path))
    names = model.names or {}
    ordered_names = [str(names[index]) for index in range(len(names))]
    if ordered_names != EXPECTED_NAMES:
        raise ValueError("Checkpoint class names/order do not match this dataset")
    train_args = (model.ckpt or {}).get("train_args", {})
    metrics = model.val(
        data=str(data_path), split=split, imgsz=imgsz, conf=conf, device=device,
        workers=0, plots=True, project=str(OUTPUT), name=run_name,
    )
    save_dir = Path(metrics.save_dir)
    box = metrics.box
    class_ids = [int(value) for value in box.ap_class_index]
    per_class = {}
    for index, class_id in enumerate(class_ids):
        name = names[class_id] if isinstance(names, dict) else names[class_id]
        per_class[str(name)] = {
            "precision": float(box.p[index]),
            "recall": float(box.r[index]),
            "map50": float(box.ap50[index]),
            "map50_95": float(box.ap[index].mean()),
        }
    report = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_path": str(model_path.resolve()),
        "model_sha256": model_hash,
        "checkpoint_training_data": train_args.get("data"),
        "dataset_yaml": str(data_path.resolve()),
        "dataset_yaml_sha256": sha256(data_path),
        "split": split,
        "confidence_threshold": conf,
        "imgsz": imgsz,
        "device": device,
        "ultralytics_version": ultralytics.__version__,
        "overall": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "f1": float(2 * box.mp * box.mr / max(box.mp + box.mr, 1e-12)),
            "map50": float(box.map50),
            "map50_95": float(box.map),
        },
        "per_class": per_class,
        "note": "Detection metrics only. These do not measure event-level alerts or person association.",
        "source_overlap_basis": "filename groups only; actual capture sources are unverified",
    }
    audit = data_path.parent / "audit.json"
    if audit.exists():
        report["dataset_audit_sha256"] = sha256(audit)
        report["source_overlap"] = json.loads(audit.read_text(encoding="utf-8"))["grouped"]["overlapping_sources"]
    else:
        grouped_audit = ROOT / "datasets" / "laboratory_safety_grouped" / "audit.json"
        report["source_overlap"] = json.loads(grouped_audit.read_text(encoding="utf-8"))["original"]["overlapping_sources"] if grouped_audit.exists() else "not_verified_for_this_yaml"
    report["image_list_sha256"] = {
        name: sha256(data_path.parent / f"{name}.txt")
        for name in ("train", "val", "test") if (data_path.parent / f"{name}.txt").exists()
    }
    training_manifest = model_path.parent.parent / "split_manifest.json"
    if training_manifest.exists():
        expected = json.loads(training_manifest.read_text(encoding="utf-8"))
        report["training_manifest_matches"] = all(Path(path).is_file() and sha256(Path(path)) == digest for path, digest in expected.items())
        report["training_manifest_sha256"] = sha256(training_manifest)
    else:
        report["training_manifest_matches"] = False
    destination = save_dir / "metrics_with_provenance.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_metrics(report, save_dir)
    print(json.dumps({"report": str(destination), "overall": report["overall"]}, ensure_ascii=False, indent=2))
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    evaluate(args.model, args.data, args.split, args.conf, args.device, args.imgsz)


if __name__ == "__main__":
    main()
