import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "laboratory_safety" / "weights" / "best.pt"
if not MODEL.exists():
    MODEL = ROOT / "models" / "yolo11s.pt"

parser = argparse.ArgumentParser()
parser.add_argument("source")
parser.add_argument("--conf", type=float, default=0.25)
args = parser.parse_args()

model = YOLO(str(MODEL))
model.predict(source=args.source, conf=args.conf, save=True, project=str(ROOT / "runs"), name="predict", exist_ok=True)
print(f"使用模型：{MODEL}")
