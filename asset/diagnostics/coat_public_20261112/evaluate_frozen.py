"""Measure final candidate regressions on the untouched 735-image test split."""
import argparse
import gc
import hashlib
import json
import os
import tempfile
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("MPLBACKEND", "Agg")
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator


class ConfusionValidator(DetectionValidator):
    def update_metrics(self, preds, batch):
        # In this Ultralytics release the matrix is only updated in plots mode.
        self.args.plots = True
        try:
            super().update_metrics(preds, batch)
        finally:
            self.args.plots = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets/laboratory_safety_public_coat_20261112")
    parser.add_argument("--output-dir", type=Path, default=FOLDER)
    args = parser.parse_args()
    selected = json.loads((args.output_dir / "public_validation.json").read_text(encoding="utf-8"))
    finalist = selected["results"][selected["selected"]]
    checkpoint = ROOT / finalist["checkpoint"]
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert sha == finalist["sha256"]
    official = ROOT / "datasets/laboratory_safety/data.yaml"
    parent_test = official.parent / "test.txt"
    pilot_test = args.data_dir / "test.txt"
    assert parent_test.read_bytes().splitlines() == pilot_test.read_bytes().splitlines()
    base = ROOT / "models/laboratory_safety/weights/best.pt"
    expected_base = selected["results"]["baseline"]["sha256"]
    assert hashlib.sha256(base.read_bytes()).hexdigest() == expected_base
    results = {}
    with tempfile.TemporaryDirectory(prefix="coat-public-eval-") as temp:
        for name, path in (("baseline", base), ("candidate", checkpoint)):
            model = YOLO(str(path))
            val = model.val(validator=ConfusionValidator, data=str(official), split="test", conf=0.001, imgsz=640,
                            workers=0, device=0, plots=False, save=False, project=temp, name=name, verbose=False)
            matrix = val.confusion_matrix.matrix.astype(int)
            assert matrix[:, 5].sum() == 502 and matrix[:, 9].sum() == 538
            classes = {int(label): i for i, label in enumerate(val.box.ap_class_index)}
            results[name] = {
                "checkpoint": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "map50": float(val.box.map50), "map50_95": float(val.box.map),
                "coat_to_no_coat": int(matrix[9, 5]), "no_coat_to_coat": int(matrix[5, 9]),
                "coat_matched_correctly": int(matrix[5, 5]), "no_coat_matched_correctly": int(matrix[9, 9]),
                "no_coat_unmatched": int(matrix[-1, 9]), "coat_unmatched": int(matrix[-1, 5]),
                "confusion_conf": val.confusion_matrix.conf, "confusion_iou": val.confusion_matrix.iou_thres,
                "all_classes": {
                    model.names[label]: {"ap50": float(val.box.ap50[i]), "ap50_95": float(val.box.ap[i].mean())}
                    for label, i in classes.items()
                },
                "confusion": matrix.tolist(),
            }
            print(name, "mAP", results[name]["map50"], results[name]["map50_95"],
                  "coat mistakes", results[name]["coat_to_no_coat"], results[name]["no_coat_to_coat"], flush=True)
            del model, val
            gc.collect()
            torch.cuda.empty_cache()
    output = {"test_list_sha256": hashlib.sha256(parent_test.read_bytes()).hexdigest(),
              "images": 735, "imgsz": 640, "nms_conf": 0.001, "results": results}
    (args.output_dir / "frozen_test_comparison.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
