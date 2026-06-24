"""Create 10 mock models to satisfy the 80/10/10 split for testing dataset.py."""

import os
import numpy as np
from PIL import Image


def main():
    # Създаваме 10 симулирани модела, за да задоволим разделението 80/10/10
    for i in range(1, 11):
        path = f"data/ShapeNet/02691156/mock_model_{i:02d}"
        os.makedirs(os.path.join(path, "img_choy2016"), exist_ok=True)

        # 1. Симулиране на points.npz
        points = (np.random.rand(10000, 3).astype(np.float32) - 0.5)
        occupancies = np.random.randint(0, 2, size=(10000,), dtype=np.uint8)
        packed_occ = np.packbits(occupancies)
        
        np.savez(os.path.join(path, "points.npz"), points=points, occupancies=packed_occ)

        # 2. Симулиране на pointcloud.npz (за тестовете на оценка)
        np.savez(os.path.join(path, "pointcloud.npz"), points=points, normals=points)

        # 3. Симулиране на едно празно изображение
        img = Image.new("RGB", (224, 224), "blue")
        img.save(os.path.join(path, "img_choy2016/00.jpg"))

    print("Симулираният дейтасет е създаден успешно в data/ShapeNet (10 модела)")


if __name__ == "__main__":
    main()