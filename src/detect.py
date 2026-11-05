import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    # Set up argument parser for command line input
    parser = argparse.ArgumentParser(description="Detect holds with a selected training run.")
    parser.add_argument("--run", required=True, help="Training folder inside models/, e.g. climbing-holds-2")
    args = parser.parse_args()

    if args.run in (".", "..") or Path(args.run).name != args.run:
        parser.error("--run must be a folder name inside models/")
    experiment = ROOT / "models" / args.run

    weights = experiment / "weights/best.pt"
    if not weights.is_file():
        parser.error(f"Model weights not found: {weights}")


    from ultralytics import YOLO

    # running the detection using the specified training run
    print(f"Using model: {weights}")
    model = YOLO(str(weights))

    # save into data/detection_result_{args.run}.jpg
    results = model(str(ROOT / "data/raw/1.jpg"), conf=0.25)
    output = ROOT / "data" / f"detection_result_{args.run}.jpg"
    for result in results:
        result.save(filename=str(output))
    print(f"Detection saved to {output}")


if __name__ == "__main__":
    main()
