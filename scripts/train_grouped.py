"""Train on disjoint filename groups and preserve image split provenance."""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from multiprocessing import freeze_support
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("MPLBACKEND", "Agg")
from ultralytics import YOLO

try:
    from .evaluate_provenance import evaluate, sha256
except ImportError:
    from evaluate_provenance import evaluate, sha256

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "laboratory_safety" / "data.yaml"
BASE = ROOT / "models" / "yolo11s.pt"


def main() -> None:
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="laboratory_safety_corrected")
    parser.add_argument("--evaluate-test", action="store_true", help="Use only for the final selected run")
    parser.add_argument("--resume", type=Path, help="Continue an unfinished local run from its last.pt checkpoint")
    parser.add_argument("--data", type=Path, default=DATA, help="Verified dataset YAML; defaults to the original frozen dataset")
    parser.add_argument("--check-data", action="store_true", help="Verify data/provenance and exit without starting training")
    args = parser.parse_args()
    data = args.data.resolve()
    if not data.is_file():
        raise FileNotFoundError("Run scripts/dataset_audit.py --create-grouped-split first")
    if not BASE.is_file():
        raise FileNotFoundError(BASE)
    audit = data.parent / "audit.json"
    if not audit.is_file():
        raise FileNotFoundError("Grouped split audit is missing")
    label_manifest = data.parent / "label_manifest.json"
    labels = json.loads(label_manifest.read_text(encoding="utf-8"))
    expected_labels = 7217
    extra_files = []
    image_hashes = {}
    if data != DATA.resolve():
        version_path = data.parent / "annotation_version.json"
        version = json.loads(version_path.read_text(encoding="utf-8"))
        expected_labels = version.get("label_file_count")
        if not isinstance(expected_labels, int) or isinstance(expected_labels, bool) or expected_labels <= 7217:
            raise RuntimeError("Additional datasets require an explicit verified label_file_count")
        verification = data.parent / "verification.json"
        report = json.loads(verification.read_text(encoding="utf-8"))
        if report.get("state") != "verified" or report.get("labels_checked") != expected_labels:
            raise RuntimeError("Augmented dataset has not passed independent verification")
        checked_files = report.get("verified_dataset_files_sha256", {})
        required_files = {"data.yaml", "train.txt", "val.txt", "test.txt", "label_manifest.json", "image_manifest.json", "annotation_version.json", "audit.json", "crop_manifest.json", "parent_manifest.json"}
        if set(checked_files) != required_files or any(not (data.parent/name).is_file() or sha256(data.parent/name) != digest for name, digest in checked_files.items()):
            raise RuntimeError("Augmented dataset metadata differs from its verification report")
        extra_files = [data.parent/name for name in ("image_manifest.json", "crop_manifest.json", "parent_manifest.json", "verification.json")]
        image_hashes = json.loads((data.parent/"image_manifest.json").read_text(encoding="utf-8"))
        if len(image_hashes) != expected_labels or any(not (ROOT/p).is_file() or sha256(ROOT/p) != digest for p, digest in image_hashes.items()):
            raise RuntimeError("Augmented dataset images differ from their verified manifest")
    if len(labels) != expected_labels or any(not (ROOT / p).is_file() or sha256(ROOT / p) != digest for p, digest in labels.items()):
        raise RuntimeError("Label files differ from the verified annotation manifest")
    manifest_files = [data, audit, label_manifest, data.parent / "annotation_version.json"] + [data.parent / f"{split}.txt" for split in ("train", "val", "test")] + extra_files
    manifest = {str(path.resolve()): sha256(path) for path in manifest_files}
    if args.check_data:
        print(json.dumps({"state": "verified", "data": str(data), "label_files": len(labels),
                          "image_files_hashed": len(image_hashes),
                          "split_images": {split: len((data.parent/f"{split}.txt").read_text(encoding="utf-8").splitlines()) for split in ("train", "val", "test")}}, indent=2))
        return
    completed_epochs = 0
    checkpoint = None
    if args.resume:
        import torch

        checkpoint = args.resume.resolve()
        if not checkpoint.is_file() or checkpoint.parent.name != "weights" or ROOT / "runs" not in checkpoint.parents:
            raise ValueError("--resume must be a checkpoint in this project's runs/<run>/weights directory")
        run_dir = checkpoint.parent.parent
        recorded_manifest = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8"))
        if recorded_manifest != manifest:
            raise RuntimeError("Dataset changed since the original run; refusing to resume on different annotations")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if ckpt.get("epoch", -1) < 0 or not ckpt.get("optimizer"):
            raise RuntimeError("Checkpoint has no unfinished epoch/optimizer state; cannot resume it")
        completed_epochs = int(ckpt["epoch"]) + 1
        del ckpt
    else:
        run_dir = ROOT / "runs" / args.name
        suffix = 2
        while run_dir.exists():
            run_dir = ROOT / "runs" / f"{args.name}_{suffix}"
            suffix += 1
        run_dir.mkdir(parents=True)
        (run_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    provenance = dict(started_at_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
                      argv=sys.argv, base_checkpoint=str(BASE), base_sha256=sha256(BASE),
                      training_script_sha256=sha256(Path(__file__)), dataset=str(data),
                      annotation_manifest_sha256=sha256(label_manifest),
                      selection_split="val", test_evaluation_requested=args.evaluate_test)
    import cv2
    import numpy as np
    import torch

    provenance["runtime_versions"] = dict(opencv=cv2.__version__, numpy=np.__version__, torch=torch.__version__)
    if checkpoint:
        provenance.update(resume_checkpoint=str(checkpoint), resume_sha256=sha256(checkpoint),
                          completed_epochs_before_resume=completed_epochs)
        previous_status = run_dir / "status.json"
        if previous_status.is_file():
            provenance["previous_status"] = json.loads(previous_status.read_text(encoding="utf-8"))
        resume_dir = run_dir / "resume_events"
        resume_dir.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        (resume_dir / f"{stamp}.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    else:
        (run_dir / "training_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    status = dict(pid=os.getpid(), state="resuming" if checkpoint else "starting", run_dir=str(run_dir),
                  completed_epochs=completed_epochs, runtime_versions=provenance["runtime_versions"])
    def write_status():
        status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = run_dir / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
        temporary.replace(run_dir / "status.json")
    write_status()
    model = YOLO(str(checkpoint or BASE))
    def training_started(trainer):
        if Path(trainer.save_dir).resolve() != run_dir.resolve():
            raise RuntimeError("Trainer selected a different run directory during resume")
        status.update(state="training", completed_epochs=trainer.start_epoch,
                      total_epochs=trainer.epochs, device=str(trainer.device))
        write_status()
    def epoch_finished(trainer):
        status.update(state="training", completed_epochs=trainer.epoch+1,
                      metrics={k:float(v) for k,v in trainer.metrics.items()})
        write_status()
    model.add_callback("on_train_start", training_started)
    model.add_callback("on_fit_epoch_end", epoch_finished)
    try:
        if checkpoint:
            model.train(resume=True, device=args.device)
        else:
            model.train(
                data=str(data), epochs=args.epochs, patience=args.patience, imgsz=640,
                batch=args.batch, device=args.device, workers=0, seed=42,
                project=str(ROOT / "runs"), name=run_dir.name, exist_ok=True, save_period=10,
            )
    except BaseException:
        status.update(state="failed_or_interrupted", error=traceback.format_exc());write_status()
        raise
    if any(sha256(path) != manifest[str(path.resolve())] for path in manifest_files) or any(sha256(ROOT / p) != digest for p,digest in labels.items()) or any(sha256(ROOT/p) != digest for p,digest in image_hashes.items()):
        status.update(state="dataset_changed");write_status()
        raise RuntimeError("Dataset manifests changed during training; test evaluation was skipped")
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)
    if args.evaluate_test:
        evaluate(best, data, "test", 0.001, args.device, 640)
    status.update(state="completed", best_checkpoint=str(best));write_status()
    print(f"Source-disjoint model ready: {best}")
    print("Deployment checkpoint was not replaced. Evaluate the selected final run on test once, then deploy explicitly.")


if __name__ == "__main__":
    freeze_support()
    main()
