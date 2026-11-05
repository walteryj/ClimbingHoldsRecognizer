from pathlib import Path
from ultralytics import YOLO

# find the root directory of the project
ROOT = Path(__file__).resolve().parents[1]

# define the path to the dataset YAML file
dataset = ROOT / "data/climbing-holds-v14/data.yaml"

# Load a pretrained model
pretrained_weights = ROOT / "models/yolo26n.pt"
if not pretrained_weights.is_file():
    raise FileNotFoundError(f"Pretrained weights not found: {pretrained_weights}")
model = YOLO(str(pretrained_weights))

model.train(
    data = str(dataset), # path to the dataset YAML file
    epochs = 30, # an epoch is a pass through the entire training dataset
    imgsz = 640, 
    batch = 8, # number of images per batch
    device = "mps", # use mac gpu for training
    project = str(ROOT / "models"), # where to save the trained model and results
    name = "climbing-holds"
)

print(f"Training run saved to {model.trainer.save_dir}")
