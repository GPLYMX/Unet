# -*- coding: utf-8 -*-
# @Time : 2025/3/5 14:29
# @Author : GuoPeng
import json
import numpy as np
import cv2
import os
from PIL import Image, ImageDraw


def labelme_json_to_png(json_path, output_path, label_map={"0": 0, "1": 0, "2": 1}):
    """
    读取LabelMe格式的JSON文件，生成灰度图并保存为PNG。

    :param json_path: str, LabelMe JSON文件路径
    :param output_path: str, 输出PNG文件路径
    :param label_map: dict, 标签映射，例如 {"background": 0, "leaf": 1, "impurity": 2}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # image_shape = tuple(data["imageHeight"]), tuple(data["imageWidth"])  # 获取图像尺寸
    image_shape = (data["imageHeight"], data["imageWidth"])  # 直接用小括号表示元组
    mask = np.zeros(image_shape, dtype=np.uint8)  # 初始化灰度图

    for shape in data["shapes"]:
        label = shape["label"]
        if label not in label_map:
            category = 1  # 跳过未定义的标签
        else:
            category = label_map[label]
        polygon = np.array(shape["points"], dtype=np.int32)
        cv2.fillPoly(mask, [polygon], category)  # 填充多边形区域

    # 保存灰度图
    Image.fromarray(mask).save(output_path)
    print(f"保存灰度图到 {output_path}")


# # 示例调用
# json_file = "example.json"  # 你的LabelMe JSON文件路径
# output_png = "output.png"  # 生成的灰度图路径
# label_mapping = {"background": 0, "leaf": 1, "impurity": 2}  # 自定义标签映射
# labelme_json_to_png(json_file, output_png, label_mapping)
if  __name__ == '__main__':
    root = r'G:\datas\data\dayi\linescan\20250114yolo_90us_8k\unet_label_pre\images'
    output_root = r'G:\datas\data\dayi\linescan\20250114yolo_90us_8k\unet_label_pre\masks'
    for file in os.listdir(root):
        if file.endswith(('json')):
            labelme_json_to_png(os.path.join(root, file), os.path.join(output_root, os.path.splitext(file)[0] + '.png')
)
