"""Train on disjoint filename groups and preserve image split provenance."""

from __future__ import annotations

import argparse
import json
from multiprocessing import freeze_support
from pathlib import Path

from ultralytics import YOLO

try:
    from .evaluate_provenance import evaluate, sha256
except ImportError:
    from evaluate_provenance import evaluate, sha256

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "laboratory_safety_grouped" / "data.yaml"
BASE = ROOT / "models" / "yolo11s.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="laboratory_safety_grouped")
    parser.add_argument("--evaluate-test", action="store_true", help="Use only for the final selected run")
    args = parser.parse_args()
    if not DATA.is_file():
        raise FileNotFoundError("Run scripts/dataset_audit.py --create-grouped-split first")
    if not BASE.is_file():
        raise FileNotFoundError(BASE)
    audit = DATA.parent / "audit.json"
    if not audit.is_file():
        raise FileNotFoundError("Grouped split audit is missing")
    manifest_files = [DATA, audit] + [DATA.parent / f"{split}.txt" for split in ("train", "val", "test")]
    manifest = {str(path.resolve()): sha256(path) for path in manifest_files}
    model = YOLO(str(BASE))
    model.train(
        data=str(DATA), epochs=args.epochs, patience=args.patience, imgsz=640,
        batch=args.batch, device=args.device, workers=0, seed=42,
        project=str(ROOT / "runs"), name=args.name,
    )
    run_dir = Path(model.trainer.save_dir)
    (run_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if any(sha256(path) != manifest[str(path.resolve())] for path in manifest_files):
        raise RuntimeError("Dataset manifests changed during training; test evaluation was skipped")
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)
    if args.evaluate_test:
        evaluate(best, DATA, "test", 0.001, args.device, 640)
    print(f"Source-disjoint model ready: {best}")
    print("Deployment checkpoint was not replaced. Evaluate the selected final run on test once, then deploy explicitly.")


if __name__ == "__main__":
    freeze_support()
    main()
