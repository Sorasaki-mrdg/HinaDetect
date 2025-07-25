from random import random

import cv2
import os

# 设置指定的目录路径
directory = 'train_positive'  # 请替换为你的目录路径

# 遍历目录下所有文件
for filename in os.listdir(directory):
    # 获取文件的完整路径
    file_path = os.path.join(directory, filename)

    # 检查文件是否是图片
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.jfif')):
        # 读取图片
        image = cv2.imread(file_path)

        # 如果图像读取成功
        if image is not None:
            # 执行上下翻转
            flipped_image = cv2.flip(image, 0)

            # 保存翻转后的图片
            flipped_image = cv2.flip(image, 0)
            flipped_file_path = os.path.join(directory, 'flipped_up_down_' + filename)
            cv2.imwrite(flipped_file_path, flipped_image)
            print(f"已上下翻转并保存: {flipped_file_path}")

            # 执行左翻转
            #rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            #rotated_file_path = os.path.join(directory, 'flipped_90_' + filename)
            #cv2.imwrite(rotated_file_path, rotated_image)
            #print(f"已旋转90度并保存: {rotated_file_path}")
        else:
            print(f"无法读取图像: {file_path}")
