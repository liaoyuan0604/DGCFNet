import os
from PIL import Image
import numpy as np

# 定义颜色映射和类别
BLU_COLORMAP = [[255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [125, 125, 125], [0, 0, 0]]
BLU_CLASSES = ['Background', 'Building', 'Vegetation', 'Water', 'Farmland', 'Road', 'Invalid']


def normalize_labels(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的每个标签图像
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):  # 假设标签图像是PNG格式
            label_path = os.path.join(input_folder, filename)
            label_img = Image.open(label_path)

            # 将标签图像转换为NumPy数组
            label_np = np.array(label_img)

            # 创建新的归一化图像（在0到1范围内）
            normalized_img = np.zeros_like(label_np, dtype=np.float32)

            # 将每个类别的标签值转换为在0到1范围内的值
            for i in range(len(BLU_CLASSES)):
                class_color = BLU_COLORMAP[i]
                class_label = i
                normalized_img[np.all(label_np == class_color, axis=-1)] = class_label / (len(BLU_CLASSES) - 1)

            # 创建保存路径并保存归一化图像
            output_path = os.path.join(output_folder, filename)
            normalized_img = Image.fromarray((normalized_img * 255).astype(np.uint8))
            normalized_img.save(output_path)
            print("Converted", filename)


# 示例用法
input_folder = "/data/AIfusion/Tony2016Edu/Yuan_Liao/GeoSeg-main/bludataa/val/label"
output_folder = "/data/AIfusion/Tony2016Edu/Yuan_Liao/GeoSeg-main/bludataa/val/masks"
normalize_labels(input_folder, output_folder)

# import os
# import numpy as np
# from PIL import Image
#
# # 定义输入和输出目录
# input_dir = "/data/AIfusion/Tony2016Edu/Yuan_Liao/GeoSeg-main/bludataa/test/label"
# output_dir = "/data/AIfusion/Tony2016Edu/Yuan_Liao/GeoSeg-main/bludataa/test/masks"
#
# # 确保输出目录存在
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#
# # 定义颜色映射表和类别列表
# BLU_COLORMAP = [[255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [125, 125, 125], [0, 0, 0]]
# BLU_CLASSES = ['Background', 'Building', 'Vegetation', 'Water', 'Farmland', 'Road', 'Invalid']
#
# # 遍历输入目录中的所有文件
# for filename in os.listdir(input_dir):
#     # 构造输入文件路径
#     input_path = os.path.join(input_dir, filename)
#
#     # 打开输入图像
#     with Image.open(input_path) as img:
#         # 将输入图像转换为numpy数组
#         label_array = np.array(img)
#
#         # 创建空白的mask图像
#         mask_array = np.zeros((label_array.shape[0], label_array.shape[1], 3), dtype=np.uint8)
#
#         # 根据颜色映射表，将标签值映射为相应的颜色
#         for index, color in enumerate(BLU_COLORMAP):
#             mask_array[label_array == index] = color
#
#         # 构造输出文件路径
#         output_filename = f"mask_{filename}"
#         output_path = os.path.join(output_dir, output_filename)
#
#         # 保存mask图像
#         mask_image = Image.fromarray(mask_array)
#         mask_image.save(output_path)
#
#     print(f"{filename} processed")
#
# print("All labels converted to masks.")
#

# import os
# from PIL import Image
#
# # 定义输入和输出目录
# input_dir = "/data/AIfusion/Tony2016Edu/Yuan_Liao/GeoSeg-main/bludata/train/image"
# output_dir = "/data/AIfusion/Tony2016Edu/Yuan_Liao/GeoSeg-main/bludataa/train/image"
#
# # 确保输出目录存在
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#
# # 遍历输入目录中的所有文件
# for filename in os.listdir(input_dir):
#     # 构造输入文件路径
#     input_path = os.path.join(input_dir, filename)
#
#     # 打开输入图像
#     with Image.open(input_path) as img:
#         # 检查图像尺寸是否为2048x2048
#         width, height = img.size
#         if width == 2048 and height == 2048:
#             # 分割图像为四个1024x1024的图像
#             for i in range(2):
#                 for j in range(2):
#                     left = j * 1024
#                     top = i * 1024
#                     right = left + 1024
#                     bottom = top + 1024
#                     img_cropped = img.crop((left, top, right, bottom))
#
#                     # 构造输出文件路径
#                     output_filename = f"{os.path.splitext(filename)[0]}_{i}{j}{os.path.splitext(filename)[1]}"
#                     output_path = os.path.join(output_dir, output_filename)
#
#                     # 保存图像
#                     img_cropped.save(output_path)
#
#     print(f"{filename} processed")
#
# print("All images processed.")
