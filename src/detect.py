from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

# Load model
model = YOLO(
    str(ROOT / "models/climbing-holds/weights/best.pt")
    )

# Run inference on an image
results = model(
    str(ROOT / "data/raw/1.jpg"),
    conf=0.25
    )

# Save detection results
for result in results:
    result.save(
        filename = str(ROOT / "data/detection_result_climbing-holds-v14.jpg"))

print("Detection complete!")