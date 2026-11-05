import argparse
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description="Detect holds in one or more images.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run", help="Run path inside models/, including nested runs")
    selection.add_argument("--weights", help="Weights file; relative paths start at the project root")
    parser.add_argument("--source", nargs="+", default=["data/raw/1.jpg"],
                        help="One or more image paths (quote paths containing spaces)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold, 0 to 1")
    args = parser.parse_args()

    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1")
    if args.run:
        models = (ROOT / "models").resolve()
        experiment = (models / args.run).resolve()
        if experiment == models or not experiment.is_relative_to(models):
            parser.error("--run must point to a training folder inside models/")
        weights = experiment / "weights/best.pt"
    else:
        weights = project_path(args.weights)
    if not weights.is_file():
        parser.error(f"Model weights not found: {weights}")

    sources = [project_path(value) for value in args.source]
    for source in sources:
        if not source.is_file():
            parser.error(f"Image not found: {source}")
        if source.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
            parser.error(f"Unsupported image extension: {source}")

    from ultralytics import YOLO

    print(f"Using model: {weights}")
    model = YOLO(str(weights))
    output_dir = ROOT / "models/detections" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    # Process one image at a time to avoid loading a large batch into memory.
    # Numbered filenames prevent collisions for images with identical basenames.
    for index, source in enumerate(sources, start=1):
        results = model(str(source), conf=args.conf)
        output = output_dir / f"{index:03d}_{source.stem}.jpg"
        results[0].save(filename=str(output))
        print(f"Saved: {output}")
    print(f"Detection complete. Results: {output_dir}")


if __name__ == "__main__":
    main()
