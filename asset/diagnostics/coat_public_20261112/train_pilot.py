"""Warm-start a standard YOLO checkpoint using reviewed clothing-only supplements."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import de_parallel

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[2]
DATA = ROOT / "datasets/laboratory_safety_public_coat_20261112"
BASELINE = ROOT / "models/laboratory_safety/weights/best.pt"
COAT_IDS = (5, 9)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask_noncoat_loss(loss, partial_rows):
    keep_class = torch.zeros(loss.shape[-1], dtype=torch.bool, device=loss.device)
    keep_class[list(COAT_IDS)] = True
    partial = torch.as_tensor(partial_rows, dtype=torch.bool, device=loss.device)
    keep = (~partial[:, None, None]) | keep_class[None, None, :]
    return loss.masked_fill(~keep, 0)


class PublicCoatLoss(v8DetectionLoss):
    """Train only the two annotated class channels on public images."""

    def __init__(self, model, public_train_paths):
        super().__init__(model)
        self.public_train_paths = public_train_paths
        self.partial_images_seen = 0

    def __call__(self, preds, batch):
        partial_rows = [str(Path(file)).casefold() in self.public_train_paths for file in batch["im_file"]]
        if not any(partial_rows):
            return super().__call__(preds, batch)

        self.partial_images_seen += sum(partial_rows)

        def apply_mask(_module, _inputs, loss):
            return mask_noncoat_loss(loss, partial_rows)

        with self.bce.register_forward_hook(apply_mask):
            return super().__call__(preds, batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--name", default="laboratory_safety_public_coat_20261112")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    assert args.epochs >= 1
    assert torch.cuda.is_available()
    run = ROOT / "runs" / args.name
    if run.exists():
        raise FileExistsError(run)
    run.mkdir(parents=True)
    manifest_path = data_dir / "verification.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename, checksum in manifest["verified_files"].items():
        if sha(data_dir / filename) != checksum:
            raise ValueError("Dataset metadata changed: " + filename)
    for filename, checksum in manifest["images"].items():
        if sha(ROOT / filename) != checksum:
            raise ValueError("Dataset image changed: " + filename)
    for filename, checksum in manifest["labels"].items():
        if sha(ROOT / filename) != checksum:
            raise ValueError("Dataset label changed: " + filename)
    public_train_paths = {
        str((ROOT / item["image"]).resolve()).casefold()
        for item in manifest["sources"] if item["split"] == "train"
    }
    if not public_train_paths:
        raise ValueError("The pilot requires reviewed public training images")
    provenance = {"baseline": str(BASELINE.relative_to(ROOT)), "baseline_sha256": sha(BASELINE),
                  "dataset": str(data_dir.relative_to(ROOT)), "verification_sha256": sha(manifest_path),
                  "config": {"epochs": args.epochs, "seed": 42, "imgsz": 640, "batch": 8, "freeze": 10,
                             "optimizer": "AdamW", "lr0": 0.0004, "lrf": 0.2, "mosaic": 0, "mixup": 0,
                             "close_mosaic": 0}, "device": torch.cuda.get_device_name(0), "pid": os.getpid(),
                  "partial_label_classes": COAT_IDS}
    (run / "pilot_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    model = YOLO(str(BASELINE))
    observed = {}

    def attach_loss(trainer):
        train_model = de_parallel(trainer.model)
        observed["criterion"] = PublicCoatLoss(train_model, public_train_paths)
        train_model.criterion = observed["criterion"]
        assert trainer.ema.ema.__dict__.get("criterion") is None

    def verify_batch_coverage(trainer):
        seen = observed["criterion"].partial_images_seen
        if seen < (trainer.epoch + 1) * len(public_train_paths):
            raise RuntimeError("Public clothing supervision was not applied as expected")
        print(f"Public training exposures through epoch {trainer.epoch+1}: {seen}", flush=True)

    model.add_callback("on_train_start", attach_loss)
    model.add_callback("on_train_epoch_end", verify_batch_coverage)
    model.train(data=str(data_dir / "data.yaml"), device=0, workers=0, save_period=1,
                project=str(run.parent), name=run.name, exist_ok=True, patience=args.epochs,
                copy_paste=0, warmup_epochs=0.5, plots=False, **provenance["config"])
    assert sha(BASELINE) == provenance["baseline_sha256"]
    assert (run / "weights" / "best.pt").is_file()
    (run / "pilot_complete.json").write_text(json.dumps({
        "public_image_exposures": observed["criterion"].partial_images_seen,
        "best_sha256": sha(run / "weights" / "best.pt"),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
