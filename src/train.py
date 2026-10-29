from pathlib import Path
from ultralytics import YOLO

# find the root directory of the project
ROOT = Path(__file__).resolve().parents[1]

# define the path to the dataset YAML file
dataset = ROOT / "data/climbing-holds-v14/data.yaml"

# Load a pretrained model
model = YOLO("yolo26n.pt")

model.train(
    data = str(dataset), # path to the dataset YAML file
    epochs = 1, # one epoch = one pass through the entire training dataset
    imgsz = 640, 
    batch = 8, # number of images per batch
    device = "mps", # use mac gpu for training
    project = str(ROOT / "models"), # where to save the trained model and results
    name = "climbing-holds"
)