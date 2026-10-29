import json
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

# create evaluation directory if it doesn't exist
experiment = ROOT / "models/climbing-holds"             # path to model
evaluation = experiment / "evaluation"
evaluation.mkdir(parents = True, exist_ok = True)

# lodad the fine tuned trained model
model = YOLO(str(experiment / "weights/best.pt"))

# compares predictions to ground truth labels
metrics = model.val(
    data = str(
        ROOT / "data/climbing-holds-v14/data.yaml"),
    split = "test",                 # using the test set for evaluation
    project = str(experiment),
    name = "evaluation",
    exist_ok = True,               # overwrite existing evaluation results
    plots = True
)

results = {
    "precision": metrics.box.mp,    # of predicted holds how many actually correct
    "recall": metrics.box.mr,       # of actual holds how many were detected
    "mAP50": metrics.box.map50,     # Mean average precision at an intersection-over-union (IoU) threshold of 0.50
    "mAP50-95": metrics.box.map     # Mean average precision averaged across IoU thresholds from 0.50 to 0.95
}

# Save metrics to the new evaluation directory
with open(Path(metrics.save_dir) / "metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"Evaluation results saved to {evaluation}")