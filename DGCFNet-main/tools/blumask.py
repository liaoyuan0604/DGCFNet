import os
from PIL import Image
import numpy as np

BLU_COLORMAP = [[255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [125, 125, 125], [0, 0, 0]]
BLU_CLASSES = ['Background', 'Building', 'Vegetation', 'Water', 'Farmland', 'Road', 'Invalid']


def normalize_labels(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            label_path = os.path.join(input_folder, filename)
            label_img = Image.open(label_path)

            label_np = np.array(label_img)

            normalized_img = np.zeros_like(label_np, dtype=np.float32)

            for i in range(len(BLU_CLASSES)):
                class_color = BLU_COLORMAP[i]
                class_label = i
                normalized_img[np.all(label_np == class_color, axis=-1)] = class_label / (len(BLU_CLASSES) - 1)

            output_path = os.path.join(output_folder, filename)
            normalized_img = Image.fromarray((normalized_img * 255).astype(np.uint8))
            normalized_img.save(output_path)
            print("Converted", filename)


# 示例用法
input_folder = "/data/AIfusion/Tony2016Edu/Yuan_Liao/GeoSeg-main/bludataa/val/label"
output_folder = "/data/AIfusion/Tony2016Edu/Yuan_Liao/GeoSeg-main/bludataa/val/masks"
normalize_labels(input_folder, output_folder)

