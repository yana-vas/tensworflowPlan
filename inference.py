import argparse
import torch
from PIL import Image

from src.data.preprocessing import ImagePreprocessor
from src.mesh import extract_mesh, save_mesh
from src.model import OccupancyNetwork
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3DScan: image -> STL")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    resolution = args.resolution or config.inference.grid_resolution
    threshold = args.threshold or config.inference.threshold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | resolution {resolution} | threshold {threshold}")

    model = OccupancyNetwork.from_checkpoint(args.checkpoint, map_location=str(device)).to(device)
    model.eval()

    preprocessor = ImagePreprocessor(image_size=config.data.image_size)
    image = Image.open(args.input).convert("RGB")
    image_tensor = preprocessor(image).unsqueeze(0).to(device)

    print("Generating occupancy grid...")
    grid = model.generate_occupancy_grid(
        image_tensor, resolution=resolution,
        query_batch_size=config.inference.query_batch_size,
    )
    print(f"Grid {grid.shape} | prob range [{grid.min():.3f}, {grid.max():.3f}]")

    print("Extracting + cleaning mesh...")
    mesh = extract_mesh(grid, threshold=threshold, pad=True, postprocess=True)
    if mesh is None:
        print("No surface found. Try a lower --threshold (e.g. 0.3).")
        return

    save_mesh(mesh, args.output)
    print(f"Done -> {args.output} | watertight={mesh.is_watertight}")


if __name__ == "__main__":
    main()