# -*- coding: utf-8 -*-
# @Time : 2025/2/28 14:50
# @Author : GuoPeng
# -*- coding: utf-8 -*-
# @Time : 2025/2/25 16:14
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
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader, random_split

from models.network import AttU_Net, U_Net
from models.network import NestedUNet, NestedAttUNet
from models.network import R2AttU_Net
from utiles.utiles import CustomSegmentationDataset, LabelMeDataset
from utiles.dice_score import dice_coeff, multiclass_dice_coeff, dice_loss
# from models.swin_transformer_unet_skip_expand_decoder_sys import  SwinTransformerSys
from utiles.dice_score import DynamicWeightedLoss

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def split_image(img, mask, patch_size=(256, 256)):
    # 获取原图尺寸
    h, w = int(img.shape[1]), int(img.shape[2])  # 获取图片的高度和宽度
    # 计算需要的填充大小
    pad_h = (patch_size[0] - h % patch_size[0]) % patch_size[0]
    pad_w = (patch_size[1] - w % patch_size[1]) % patch_size[1]

    # 填充图片和掩码，确保它们的大小可以被patch_size整除
    padded_img = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    # padded_mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    # 计算分块的数量
    img_patches = []
    mask_patches = []

    for i in range(0, padded_img.shape[1], patch_size[0]):
        for j in range(0, padded_img.shape[2], patch_size[1]):
            # 对每个图像块进行切割
            img_patch = padded_img[:, i:i + patch_size[0], j:j + patch_size[1]]
            # mask_patch = padded_mask[i:i + patch_size[0], j:j + patch_size[1]]

            # 如果图像块或掩码块的尺寸不等于patch_size，则跳过该块（可选）
            if img_patch.shape[1] != patch_size[0] or img_patch.shape[2] != patch_size[1]:
                continue

            img_patches.append(img_patch)
            # mask_patches.append(mask_patch)

    # 统一转换为NumPy数组
    img_patches = np.array(img_patches)
    # mask_patches = np.array(mask_patches)

    return img_patches, 1



def train(model, train_dataset, val_dataset, batch_size=1, epochs=10, learning_rate=1e-3, patch_size=(256, 256),
          device='cpu' if torch.cuda.is_available() else 'cpu', model_save_path1="best_dice.pth", model_save_path2="best_loss.pth"):
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # 每次加载一张图片
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()  # 交叉熵损失

    model.to(device)
    model.train()

    best_dice = 0.0
    best_loss = float('inf')

    for epoch in range(epochs):
        epoch_train_loss = 0.0
        epoch_train_dice = 0.0

        model.train()
        # 训练阶段
        for img, mask in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            img = img[0].numpy()  # (1, C, H, W) 转为 (C, H, W)
            mask = mask[0].numpy()  # (1, H, W) 转为 (H, W)

            img_patches, mask_patches = split_image(img, mask, patch_size=patch_size)
            optimizer.zero_grad()

            for i in range(len(img_patches)):
                patch_img = torch.tensor(img_patches[i]).float().unsqueeze(0).to(device)  # (1, C, H, W)
                patch_mask = torch.tensor(mask_patches[i]).long().unsqueeze(0).to(device)  # (1, H, W)

                # 前向传播
                output = model(patch_img)
                # 去掉多余的维度
                patch_mask = patch_mask.squeeze(1)

                loss = criterion(output, patch_mask)

                # 反向传播
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                epoch_train_dice += dice_coefficient(output, patch_mask)

        # 计算训练集的平均损失和Dice系数
        epoch_train_loss /= len(train_loader)
        epoch_train_dice /= len(train_loader)

        # 验证阶段
        epoch_val_loss = 0.0
        epoch_val_dice = 0.0
        model.eval()
        with torch.no_grad():
            for img, mask in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                img = img[0].numpy()
                mask = mask[0].numpy()

                img_patches, mask_patches = split_image(img, mask, patch_size=patch_size)

                for i in range(len(img_patches)):
                    patch_img = torch.tensor(img_patches[i]).float().unsqueeze(0).to(device)
                    patch_mask = torch.tensor(mask_patches[i]).long().unsqueeze(0).to(device)

                    # 前向传播
                    output = model(patch_img)
                    loss = criterion(output, patch_mask)

                    epoch_val_loss += loss.item()
                    epoch_val_dice += dice_coefficient(output, patch_mask)

        # 计算验证集的平均损失和Dice系数
        epoch_val_loss /= len(val_loader)
        epoch_val_dice /= len(val_loader)

        # 输出每轮训练和验证的结果
        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Dice: {epoch_train_dice:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Dice: {epoch_val_dice:.4f}")

        # 保存最佳模型
        if epoch_val_dice > best_dice:
            best_dice = epoch_val_dice
            torch.save(model.state_dict(), model_save_path1)
            print(f"Saved Best Model (Dice: {best_dice:.4f})")

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path2)
            print(f"Saved Best Model (Loss: {best_loss:.4f})")

def predict(model, image, patch_size=(4096, 4096), device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    model.eval()

    # 切割大图为小块
    img_patches, _ = split_image(image, np.zeros(image.shape[:2]), patch_size=patch_size)

    pred_patches = []
    with torch.no_grad():
        for patch in img_patches:
            patch_tensor = torch.tensor(patch).float().unsqueeze(0).to(device)  # (1, C, H, W)
            output = model(patch_tensor)
            pred_patches.append(output.cpu().numpy())

    # 合并子图
    h, w = image.shape[:2]
    pad_h = (patch_size[0] - h % patch_size[0]) % patch_size[0]
    pad_w = (patch_size[1] - w % patch_size[1]) % patch_size[1]

    pred_img = np.zeros((h + pad_h, w + pad_w), dtype=np.float32)

    idx = 0
    for i in range(0, h + pad_h, patch_size[0]):
        for j in range(0, w + pad_w, patch_size[1]):
            pred_img[i:i + patch_size[0], j:j + patch_size[1]] = pred_patches[idx][0, 0]
            idx += 1

    return pred_img

def dice_coefficient(pred, target, threshold=0.5):
    """
    计算Dice系数
    :param pred: 模型的输出，形状为 (N, C, H, W)，N为batch_size，C为类别数，H和W为图像尺寸
    :param target: 真实标签，形状为 (N, H, W)
    :param threshold: 用于预测的阈值，默认0.5，适用于二分类问题
    :return: Dice系数
    """
    # 将输出的概率值转化为类别标签
    pred = torch.argmax(pred, dim=1)  # shape: (N, H, W)

    # 计算预测结果和真实标签的交集与并集
    intersection = torch.sum(pred * target)  # 预测和真实标签相交的像素数
    total = torch.sum(pred) + torch.sum(target)  # 预测和真实标签总的像素数

    # 防止除以0
    dice = (2. * intersection + 1e-6) / (total + 1e-6)

    return dice.item()

if __name__ == '__main__':

    model = AttU_Net(img_ch=3, output_ch=2)
    model.load_state_dict(torch.load(r'paramters\best_dice.pth'))
    img_root = r'G:\datas\data\dayi\linescan\20250114yolo_90us_8k\test\chageng.bmp'
    img = cv2.imread(img_root)
    img = np.transpose(img, (2, 1, 0))
    print(img.shape)
    a = predict(model=model, image=img)
    plt.imshow(a)
    plt.show()

