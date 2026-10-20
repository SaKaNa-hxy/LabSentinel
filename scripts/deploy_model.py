"""Deploy a checkpoint evaluated on disjoint filename groups with a backup."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

try:
    from .evaluate_provenance import ROOT, sha256
except ImportError:
    from evaluate_provenance import ROOT, sha256

TARGET = ROOT / "models" / "laboratory_safety" / "weights" / "best.pt"
MANIFEST = TARGET.parent / "deployment.json"


def deploy(report_path: Path) -> Path:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    source = Path(report["model_path"])
    if not source.is_file() or sha256(source) != report["model_sha256"]:
        raise ValueError("Evaluation report does not match the checkpoint")
    if report.get("split") != "test" or report.get("dataset_yaml", "").replace("\\", "/").split("/")[-2] != "laboratory_safety_grouped":
        raise ValueError("Deployment requires a grouped test evaluation")
    if any(report.get("source_overlap", {}).values()):
        raise ValueError("Grouped train/val/test sources still overlap")
    if "laboratory_safety_grouped" not in str(report.get("checkpoint_training_data", "")):
        raise ValueError("Checkpoint was not trained on the grouped split")
    if not report.get("training_manifest_matches"):
        raise ValueError("Training and evaluation dataset manifests were not verified as identical")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = TARGET.with_name(f"best_backup_before_grouped_{stamp}.pt")
        shutil.copy2(TARGET, backup)
    temporary = TARGET.with_suffix(".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(TARGET)
    manifest = {
        "deployed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "deployed_sha256": sha256(TARGET),
        "source_checkpoint": str(source.resolve()),
        "evaluation_report": str(report_path.resolve()),
        "grouped_test_metrics": report["overall"],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return TARGET


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="metrics_with_provenance.json from the grouped test evaluation")
    deploy(parser.parse_args().report)
