# -*- coding: utf-8 -*-
# @Time : 2023/9/19 14:15
# @Author : GuoPeng
import os
import math
import yaml
import time
import random

from PIL import Image
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision import transforms

from models.network import AttU_Net, U_Net, R2AttU_Net
from utiles.postprocessing import calculate_center
from utiles.dice_score import dice_coeff

"""
将十三通道的图片放入模型中预测，根据预测的结果生成灰度图，灰度图中的数字代表类别
"""
use_gpu = torch.cuda.is_available()
if use_gpu:
    print('use_GPU:', True)
    device = torch.device('cuda')
else:
    print('use_GPU:', False)
    device = torch.device('cpu')


def load_configs(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


configs = load_configs('configs.yaml')
num_classes = configs['num_classes']
in_channels = configs['in_channels']
base_h, base_w = configs['base_size']
pixel_shift_ratio = configs['pixel_shift_ratio']
batch_size = configs['batch_size']


def combine_img(folder_path, low_pixel_test=True):
    # 获取文件夹中的所有图像文件名
    image_files = [filename for filename in os.listdir(folder_path) if filename.endswith(".png")]
    image_files = [image_files[i] for i in in_channels]

    # 加载灰度图像并添加到列表中
    # image_files.sort()
    image_list = []

    for idx, img_path in enumerate(image_files):
        if idx <= 2:
            img = Image.open(os.path.join(folder_path, img_path)).convert("L")  # 将图像转换为灰度模式
        else:
            if low_pixel_test:
                img = Image.open(os.path.join(folder_path, img_path)).convert("L")
                width, height = img.size
                new_width, new_height = width // 4, height // 4

                # 创建一个新的空白图像
                downsampled_image = Image.new("RGB", (new_width, new_height))
                # 遍历原图像，以局部单元格的左上角元素为基准，取样四方格
                for x in range(0, new_width):
                    for y in range(0, new_height):
                        pixel = img.getpixel((x * 4, y * 4))
                        downsampled_image.putpixel((x, y), pixel)

                img = downsampled_image.resize((width, height), Image.BILINEAR)

            else:
                img = Image.open(os.path.join(folder_path, img_path)).convert("L")

        image_list.append(img)

    # for img_path in image_files:
    #     img = Image.open(os.path.join(folder_path, img_path)).convert("L")  # 将图像转换为灰度模式
    #     image_list.append(img)

    # 确定图像的尺寸（假设所有图像都有相同的尺寸）
    width, height = image_list[0].size

    # 创建一个空的PyTorch张量，用于存储多通道图像
    multi_channel_image = torch.zeros(len(image_list), height, width)

    # 将灰度图像的像素数据叠加到PyTorch张量中
    for i, img in enumerate(image_list):
        # 将PIL图像转换为PyTorch张量
        img_tensor = transforms.ToTensor()(img)
        # 仅使用灰度通道数据
        multi_channel_image[i] = img_tensor[0]

    # # 添加随机偏移
    # random_list = [random.uniform(pixel_shift_ratio[0], pixel_shift_ratio[1]) for i in
    #                range(len(image_files))]  # 偏移率列表
    # print('偏移率“', random_list)
    # random_array = [multi_channel_image[i + 1] - multi_channel_image[i] for i in range(len(image_files) - 1)]
    # random_array.append(random_array[-1])  # 最后一位插值
    # random_array = [random_array[i] * random_list[i] for i in range(len(image_files))]
    # for i in range(3, len(image_files)):
    #     multi_channel_image[i] = multi_channel_image[i] + random_array[i]

    # 添加随机偏移
    random_rate = [random.uniform(pixel_shift_ratio[0], pixel_shift_ratio[1]) for i in
                   range(len(image_files))]  # 偏移率列表
    random_array = [multi_channel_image[i + 1] - multi_channel_image[i] for i in range(len(image_files) - 1)]
    random_array1 = random_array.copy()
    random_array2 = random_array.copy()
    random_array1.append(random_array[0])  # 第一位插值
    random_array2.append(random_array[-1])  # 最后一位插值
    random_array = [random_array1[i] * random_rate[i] if random_rate[i] < 0 else random_array2[i] * random_rate[i] for i
                    in
                    range(len(image_files))]
    for i in range(3, len(image_files)):
        multi_channel_image[i] = multi_channel_image[i] + random_array[i]

    torch.cuda.synchronize()
    a = time.time()
    torch.cuda.synchronize()
    multi_channel_image.to(device)
    torch.cuda.synchronize()
    print('从CPU转移到cuda耗时:', time.time() - a)
    torch.cuda.synchronize()
    return multi_channel_image


def read_img(img_root):
    """
    读取十三通道图片，生成tensor
    :param img_root: 路径应该在通道图片所在文件夹的上一层目录中
    :return:model需要的格式, 原始图片的尺寸
    """
    image = cv2.imread(img_root)
    h, w = image.shape[1], image.shape[2]
    image = img_tensor = transforms.ToTensor()(img)
    image = torch.unsqueeze(image, 0)

    return image, [h, w]


def model_pred(image, model, shape):
    # image = resize_img(image)
    # h = math.ceil(image.shape[2] / 2)
    # w = math.ceil(image.shape[3] / 2)
    # batch_list = [image[:, :, h:, :w], image[:, :, h:, w:], image[:, :, :h, :w],
    #               image[:, :, :h, w:]]
    #
    # outputs = []
    # with torch.no_grad():
    #     for i in range(4):
    #         model.to(device)
    #         batch_list[i] = batch_list[i].requires_grad_(False)
    #         outputs.append(model(batch_list[i].to(device)).to('cpu'))
    #
    # outputs = torch.cat([torch.cat([outputs[2], outputs[3]], dim=3), torch.cat([outputs[0], outputs[1]], dim=3)], dim=2)
    # torch.cuda.synchronize()
    ttt = time.time()
    # torch.cuda.synchronize()
    height, width = shape
    iter_num_h, iter_num_w = math.ceil(height / base_h), math.ceil(width / base_w)
    boundary_pixel_num_h, boundary_pixel_num_w = height % base_h, width % base_w
    padding_pixel_num_h, padding_pixel_num_w = (base_h - boundary_pixel_num_h) % base_h, (
            base_w - boundary_pixel_num_w) % base_w
    image = torch.nn.functional.pad(image, (0, padding_pixel_num_w, 0, padding_pixel_num_h),
                                    mode='replicate').to(device)

    outputs = torch.zeros([batch_size, num_classes, height + padding_pixel_num_h, width + padding_pixel_num_w]).to(
        device)
    # outputs = image[:, :num_classes, :, :]
    print("图片分割时间:", time.time() - ttt)
    # model.eval()
    # if use_gpu:
    #     model.half()
    #     image = image.half()
    with torch.no_grad():
        for i in range(iter_num_h):
            for j in range(iter_num_w):
                offset_h, offset_w = i * base_h, j * base_w
                temp_sample = image[:, offset_h:base_h + offset_h, offset_w:base_w + offset_w]
                temp_sample.requires_grad_(False)
                output = model(temp_sample.unsqueeze(0))
                outputs[:, :num_classes, offset_h:base_h + offset_h, offset_w:base_w + offset_w] = output

    # outputs = outputs[:, :, 0:height, 0:width]
    # torch.cuda.synchronize()
    print('单张预测时间：', time.time() - ttt)
    torch.cuda.synchronize()

    return outputs


def read_gray_label(label):
    """
    读取label，并转化为独热编码格式的图片
    :param label_root:
    :return:
    """
    # label = Image.open(label_root).convert('L')
    label_image = transforms.ToTensor()(label)
    label_image = label_image * 255.
    label_image_onehot = np.zeros((num_classes, label_image.shape[1], label_image.shape[2]))
    for class_idx in range(num_classes):
        label_image_onehot[class_idx, :, :] = (label_image == class_idx)
    label_image_onehot = torch.tensor(label_image_onehot, dtype=torch.float32)  # 转换为 PyTorch 张量
    return label_image_onehot


def get_impurity_mask(root, model):
    """
    读取十三通道图片，生成杂质的掩码图
    :param img_root:
    :return:
    """
    # img_root = os.path.join(root, 'pre_process')
    # torch.cuda.synchronize()
    t = time.time()
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # torch.cuda.synchronize()
    img = Image.open(root).convert('RGB')
    img = transform(img)
    shape = (img.shape[1], img.shape[2])
    # img, shape = read_img(img_root)
    # torch.cuda.synchronize()
    print('预处理时间', time.time() - t)
    # torch.cuda.synchronize()
    # print('np.max(img):', np.max(np.array(img)))
    print('np.max(img):', np.max(np.array(img)))
    img = model_pred(img, model, shape)
    print('np.max(img):', np.max(np.array(img.to('cpu'))))
    img = img[:, :, :shape[0], :shape[1]]
    try:
        label_name = os.path.join(root, 'label.png')
        gray_image = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        gray_image = np.array(gray_image)
        gray_image[gray_image == 3] = 1
        # gray_image = Image.fromarray(gray_image)
        # label_image_onehot = read_gray_label(gray_image)
        # label_image_onehot = torch.unsqueeze(label_image_onehot, 0)
        # print('dice系数：', dice_coeff(F.softmax(img.to('cpu'), dim=1), label_image_onehot))
    except Exception as e:
        print(e)
        print('输出dice系数失败')
        pass
    img = torch.squeeze(img, 0)
    img = torch.argmax(img, axis=0)
    print('after shape:', img.shape)
    img[img == 1] = 100
    img[img == 2] = 1

    torch.cuda.synchronize()
    a = time.time()
    torch.cuda.synchronize()
    img = img.cpu()
    img = img.detach().numpy()
    torch.cuda.synchronize()
    print("数据从cuda迁移到cpu耗时", time.time() - a)
    torch.cuda.synchronize()

    return img


if __name__ == '__main__':
    t0 = time.time()
    model = AttU_Net(img_ch=3, output_ch=2)
    model.load_state_dict(torch.load(r'D:\mycodes\RITH\puer\unet-rgb\avmin_loss.pt', map_location=device))
    # model = torch.load(r'D:\mycodes\RITH\puer\unet-rgb\paramters\min_loss.pt')
    model.to(device)
    model.eval()
    t1 = time.time()
    print("模型加载时间：", t1 - t0)
    roots = r'G:\datas\data\fangshuijingmai\20250303\test'
    save_root = r'G:\datas\data\fangshuijingmai\20250303\pred'
    for i in os.listdir(roots):
        if i.endswith(('bmp', 'tif')):
            root = os.path.join(roots, i)
            image = cv2.imread(root, cv2.COLOR_BGR2RGB)
            mask = get_impurity_mask(root, model)
            color_mask = np.zeros_like(image)
            color_mask[mask > 0] = [0, 0, 255]
            overlay = cv2.addWeighted(image, 1, color_mask, 0.5, 0)
            # calculate_center(img)
            print("计算杂质掩码所需总时间", time.time() - t0)
            # print('np.max(img):', np.max(overlay))
            plt.imshow(overlay[:,:,::-1])
            plt.title(i)
            plt.axis('off')
            plt.show()
            # overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_root, i), overlay)
            print(f"已保存叠加结果：{i}")

    # root = r'D:\mycodes\RITH\puer\data_20231020\test\7-2'
    # img = get_impurity_mask(root, model)
    # calculate_center(img)
    # print("计算杂质掩码所需总时间", time.time() - t0)
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()
