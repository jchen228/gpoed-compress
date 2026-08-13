import numpy as np

data = np.fromfile("/path/to/TCf48.bin.f32", dtype=np.float32).reshape(100, 500, 500)

for level in range(15):
    layer = data[level]
    print(f"Level {level:3d}: min={layer.min():.2f}  max={layer.max():.2f}  "
          f"std={layer.std():.4f}  NaN={np.isnan(layer).sum()}")