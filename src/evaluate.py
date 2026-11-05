import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():

    # Set up argument parser for command line input
    parser = argparse.ArgumentParser(description="Evaluate a selected training run on the test split.")
    parser.add_argument("--run", required=True, help="Training folder inside models/, e.g. climbing-holds-2")
    args = parser.parse_args()

    if args.run in (".", "..") or Path(args.run).name != args.run:
        parser.error("--run must be a folder name inside models/")

    experiment = ROOT / "models" / args.run
    weights = experiment / "weights/best.pt"

    if not weights.is_file():
        parser.error(f"Model weights not found: {weights}")


    from ultralytics import YOLO

    print(f"Using model: {weights}")
    model = YOLO(str(weights))

    metrics = model.val(
        data=str(ROOT / "data/climbing-holds-v14/data.yaml"),
        split="test",
        project=str(experiment),
        name="evaluation",
        exist_ok=True,  # Reuse this run's evaluation folder.
        plots=True,
    )
    results = {
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
    }
    with open(Path(metrics.save_dir) / "metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation results saved to {metrics.save_dir}")


if __name__ == "__main__":
    main()