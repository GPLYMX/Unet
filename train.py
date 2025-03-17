import random
import math
import yaml
from pathlib import Path
import sys
import os
from tqdm import tqdm
import csv
from sys import getsizeof as getsize
import gc

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from models.network import AttU_Net, U_Net
from models.network import NestedUNet, NestedAttUNet
from models.network import R2AttU_Net
from utiles.utiles import CustomSegmentationDataset, LabelMeDataset
from utiles.dice_score import dice_coeff, multiclass_dice_coeff, dice_loss
# from models.swin_transformer_unet_skip_expand_decoder_sys import  SwinTransformerSys
from utiles.dice_score import DynamicWeightedLoss

# from utiles.dice_score import LabelSmoothingCrossEntropyWithWeight

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
use_gpu = torch.cuda.is_available()
if use_gpu:
    print('use_GPU:', True)
    device = torch.device('cuda')
    # torch.cuda.manual_seed_all(3407)
else:
    print('use_GPU:', False)
    device = torch.device('cpu')
    # torch.manual_seed(3407)
# random.seed(3407)



def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


configs = load_config('configs.yaml')
num_classes = configs['num_classes']
in_channels = configs['in_channels']
in_channels2 = configs['in_channels2']
batch_size = configs['batch_size']
base_size = configs['base_size']
num_epochs = configs['num_epochs']
pixel_shift_ratio = configs['pixel_shift_ratio']
classes_weight = configs['classes_weight']
in_channels_list = [[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

# data_list1 = [r'/home/guopeng/projects/dayi/datas/unet/1hao-benxian/data20240509',
#              r'/home/guopeng/projects/dayi/datas/unet/1hao-benxian/data20240528-0529-supplemnet',
#              r'/home/guopeng/projects/dayi/datas/unet/1hao-benxian/data20240510',
#              r'/home/guopeng/projects/dayi/datas/unet/1hao-benxian/data20240511',
#              r'/home/guopeng/projects/dayi/datas/unet/1hao-benxian/data20240512',
#              r'/home/guopeng/projects/dayi/datas/unet/1hao-benxian/data20240528-0529-supplemnet',
#              r'/home/guopeng/projects/dayi/datas/unet/4hao-benxian/data20240513-0515']
# data_list2 = [r'/home/guopeng/projects/dayi/datas/unet/1hao-benxian/data20240522-gaicha',
#               r'/home/guopeng/projects/dayi/datas/unet/1hao-benxian/data20240627',
#               r'/home/guopeng/projects/dayi/datas/unet/1hao-benxian/data20240523-gaicha',
#               r'/home/guopeng/projects/dayi/datas/unet/1hao-benxian/data20240628',
#               r'/home/guopeng/projects/dayi/datas/unet/1hao-benxian/data20240613-gaicha-chageng-chaguo']

data_list1 = [r'/data/dayi/datas/unet/1hao-benxian-muti-exposure/20241118',
              r'/data/dayi/datas/unet/1hao-benxian-muti-exposure/20241119',
              r'/data/dayi/datas/unet/1hao-benxian-muti-exposure/20241120',
              r'/data/dayi/datas/unet/1hao-benxian-muti-exposure/20241125',
              r'/data/dayi/datas/unet/1hao-benxian-muti-exposure/20241127']
# data_list2 = [r'/data/dayi/datas_gray_background/1hao-benxian/data20240813']

def clear_catch(n=10):
    for i in range(n):
        torch.cuda.empty_cache()


best_dice = 0.3
best_dice2 = 0
min_loss = 1000

# 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss(weight=torch.tensor(classes_weight))
# criterion = dice_loss(multiclass=True)
criterion = DynamicWeightedLoss(weight=torch.tensor(classes_weight))
model = AttU_Net(img_ch=3, output_ch=2)
# model = NestedAttUNet(num_classes=3, input_channels=len(in_channels), deep_supervision=False)
# model.load_state_dict(torch.load('temp.pt'))
# model = SwinTransformerSys(img_size=base_size[1], img_ch=len(in_channels), output_ch=num_classes)
# model.load_state_dict(torch.load('best_dice.pt'))


optimizer = optim.Adam(model.parameters(), lr=0.000001, weight_decay=1e-4)
model = model.to(device)


def l1_regularization(model, l1_alpha=0.000003):
    l1_loss = []
    for module in model.modules():
        if type(module) is nn.BatchNorm2d:
            l1_loss.append(torch.abs(module.weight).sum())
    return l1_alpha * sum(l1_loss)


def l2_regularization(model, l2_alpha=0.00003):
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Conv2d:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)



def generate_dataset(data_list, train_or_test='train', augment=False):
    temp_lst = []
    for f in data_list:
        # f = os.path.join(f, train_or_test)
        train_dataset = CustomSegmentationDataset(data_dir=f,
                                                  channels=in_channels, channels2=in_channels2,label_format='png', augment=augment,
                                                  low_pixel_test=False, pixel_shift_ratio=pixel_shift_ratio,
                                                  num_classes = num_classes
                                                  )
        temp_lst.append(train_dataset)
    if len(temp_lst) <=1:
        return temp_lst[0]
    else:
        for i in range(1, len(temp_lst)):
            temp_lst[0] += temp_lst[i]
        return temp_lst[0]

        
def save_to_csv(filename, fieldnames, data):
    """
    将数据保存到CSV文件中。

    参数：
    filename (str): 保存文件的名称。
    fieldnames (list): CSV文件的字段名。
    data (list of dict): 需要保存的数据，每个字典代表一行。
    """
    # 检查文件是否存在以决定是否需要写入标题
    try:
        with open(filename, 'r', newline='') as f:
            file_exists = True
    except FileNotFoundError:
        file_exists = False
    
    # 写入数据
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # 如果文件不存在，写入标题
        if not file_exists:
            writer.writeheader()
        
        # 写入数据行
        writer.writerows(data)

# 示例函数，演示如何保存中间值到CSV文件中
def save_function(a, b, c, d, e, f, g):
    # 假设这是你的中间变量

    # 定义CSV文件的字段名
    fieldnames = ['epoch', 'train_loss', 'train_dice', 'val_loss', 'val_dice', 'test_loss', 'test_dice']

    # 准备要保存的数据，每个字典代表一行
    data = [
        {'epoch':a, 'train_loss':b, 'train_dice':c, 'val_loss':d, 'val_dice':e, 'test_loss':f, 'test_dice':g}
    ]

    # 将数据保存到CSV文件中
    save_to_csv('reflog.csv', fieldnames, data)
# dataset = LabelMeDataset(r'G:\datas\data\dayi\linescan\20250114yolo_90us_8k\unet_label')
dataset = generate_dataset([r'G:\datas\data\fangshuijingmai\20250303\train'])
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
# 在训练循环中进行训练
for epoch in range(num_epochs):
    model.train()
    model = model.to(device)

    total_loss = 0
    dice_score = 0
    n = 0

    # if num_epochs - epoch <= 50:
    #
    #     train_dataset1 = generate_dataset(data_list1, train_or_test='train', augment=False)
    #     # train_dataset2 = generate_dataset(data_list1, train_or_test='test', augment=False)
    #     # train_dataset3 = generate_dataset(data_list2, train_or_test='train', augment=False)
    #
    #
    #
    # else:
    #     train_dataset1 = generate_dataset(data_list1, train_or_test='train', augment=True)
        # train_dataset2 = generate_dataset(data_list1, train_or_test='test', augment=True)
        # train_dataset3 = generate_dataset(data_list2, train_or_test='train', augment=True)

    # val_dataset = generate_dataset(data_list, train_or_test='val', augment=True)
    # train_dataset = train_dataset1
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_dataset1 = generate_dataset(data_list1, train_or_test='val', augment=False)
    # # val_dataset2 = generate_dataset(data_list2, train_or_test='val', augment=False)
    # val_dataset = val_dataset1
    #
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # test_dataset = val_dataset
    # test_loader = val_loader
    # if epoch == 0:
    #     print(len(train_loader), len(val_loader), len(test_loader))
    #
    #     print(getsize(train_dataset), getsize(train_loader), getsize(val_dataset), getsize(val_loader))

    # 训练
    num_train = 0
    num_val = 0
    for batch_sample, batch_label in tqdm(train_loader, dynamic_ncols=True, leave=False, mininterval=1.0):
        n = n + 1
        optimizer.zero_grad()  # 梯度清零

        base_h, base_w = configs['base_size']
        height, width = batch_sample.shape[2], batch_sample.shape[3]
        iter_num_h, iter_num_w = math.ceil(height / base_h), math.ceil(width / base_w)
#        boundary_pixel_num_h, boundary_pixel_num_w = height % base_h, width % base_w
#        padding_pixel_num_h, padding_pixel_num_w = (base_h - boundary_pixel_num_h) % base_h, (
#                base_w - boundary_pixel_num_w) % base_w
#
#        batch_sample = torch.nn.functional.pad(batch_sample, (0, padding_pixel_num_w, 0, padding_pixel_num_h),
#                                               mode='replicate').to('cpu')
#        batch_label = torch.nn.functional.pad(batch_label, (0, padding_pixel_num_w, 0, padding_pixel_num_h),
#                                              mode='replicate').to('cpu')

        loss = 0
        for i in range(iter_num_h):
            for j in range(iter_num_w):
                model = model.to(device)
                height, width = batch_sample.shape[2], batch_sample.shape[3]
                iter_num_h, iter_num_w = math.ceil(height / base_h), math.ceil(width / base_w)
                boundary_pixel_num_h, boundary_pixel_num_w = height % base_h, width % base_w
                padding_pixel_num_h, padding_pixel_num_w = (base_h - boundary_pixel_num_h) % base_h, (
                        base_w - boundary_pixel_num_w) % base_w
                offset_h, offset_w = i * base_h, j * base_w
                if i == iter_num_h - 1:
                    offset_h = offset_h - padding_pixel_num_h
                if j == iter_num_w - 1:
                    offset_w = offset_w - padding_pixel_num_w
                temp_sample = batch_sample[:, :, offset_h:base_h + offset_h, offset_w:base_w + offset_w]
                temp_label = batch_label[:, :, offset_h:base_h + offset_h, offset_w:base_w + offset_w]

                outputs = model(temp_sample.to(device))
                temp_sample.to('cpu')
                # 计算损失
                # loss =  dice_loss(outputs.to('cpu'), temp_label.argmax(dim=1), multiclass=True)  # 注意需要将标签从独热编码转换为索引
                # loss =  dice_loss(F.softmax(outputs.to('cpu'), dim=1), temp_label, multiclass=True) 
                loss =  criterion(outputs.to('cpu'), temp_label.argmax(dim=1)) 
                loss = loss + l2_regularization(model).to('cpu')
                loss = loss + l1_regularization(model).to('cpu')

                # loss = loss.detach()

                # loss = 1 - dice_coeff(F.softmax(outputs.to('cpu'), dim=3), temp_label)
                total_loss += loss.item()
                # total_loss = total_loss + l2_regularization(model) + l1_regularization(model)
                dice_score += multiclass_dice_coeff(torch.argmax(F.softmax(outputs.to('cpu'), dim=1), dim=1).unsqueeze(0), temp_label.argmax(dim=1).unsqueeze(0))
                total_loss = total_loss + l2_regularization(model) + l1_regularization(model)

                dice_score = dice_score.detach()

                num_train = num_train + 1
                loss = loss + l2_regularization(model).to('cpu')
                # loss = loss + l1_regularization(model).to('cpu')
                total_loss = total_loss + l2_regularization(model)


                loss.backward()
                optimizer.step()

        
        del batch_sample
        del batch_label
        gc.collect()

        # print('ggggg')

    # 测试
    val_total_loss = 0
    val_dice_score = 0

    test_total_loss = 0
    test_dice_score = 0
    n = 0
    # model.eval()
    with torch.no_grad():
        for batch_sample, batch_label in tqdm(val_loader, dynamic_ncols=True, leave=False, mininterval=1.0):
            base_h, base_w = configs['base_size']
            height, width = batch_sample.shape[2], batch_sample.shape[3]
            iter_num_h, iter_num_w = math.ceil(height / base_h), math.ceil(width / base_w)
            boundary_pixel_num_h, boundary_pixel_num_w = height % base_h, width % base_w
            padding_pixel_num_h, padding_pixel_num_w = (base_h - boundary_pixel_num_h) % base_h, (
                                                        base_w - boundary_pixel_num_w) % base_w
#            batch_sample = torch.nn.functional.pad(batch_sample, (0, padding_pixel_num_w, 0, padding_pixel_num_h), mode='replicate').to('cpu')
#            batch_label = torch.nn.functional.pad(batch_label, (0, padding_pixel_num_w, 0, padding_pixel_num_h), mode='replicate').to('cpu')
            loss = 0
            for i in range(iter_num_h):
                for j in range(iter_num_w):
                    model.to(device)
                    offset_h, offset_w = i * base_h, j * base_w
                    if i == iter_num_h - 1:
                        offset_h = offset_h - padding_pixel_num_h
                    if j == iter_num_w - 1:
                        offset_w = offset_w - padding_pixel_num_w
                    temp_sample = batch_sample[:, :, offset_h:base_h + offset_h, offset_w:base_w + offset_w]
                    temp_label = batch_label[:, :, offset_h:base_h + offset_h, offset_w:base_w + offset_w]
                    outputs = model(temp_sample.to(device))
                    temp_sample.to('cpu')
                    # 计算损失
                    loss = criterion(outputs.to('cpu'), temp_label.argmax(dim=1))  # 注意需要将标签从独热编码转换为索引
                    loss = loss.detach()
                    # loss = 1 - dice_coeff(F.softmax(outputs.to('cpu'), dim=3), temp_label)
                    val_total_loss += loss.item()
                    val_dice_score += multiclass_dice_coeff(torch.argmax(F.softmax(outputs.to('cpu'), dim=1), dim=1).unsqueeze(0), temp_label.argmax(dim=1).unsqueeze(0))
                    val_dice_score = val_dice_score.detach()
                    num_val = num_val + 1

                    del loss
                    del temp_sample
                    gc.collect()
            del batch_sample
            del batch_label
            gc.collect()

                    

#        for batch_sample, batch_label in tqdm(test_loader, dynamic_ncols=True, leave=False, mininterval=1.0):
#            base_h, base_w = configs['base_size']
#            height, width = batch_sample.shape[2], batch_sample.shape[3]
#            iter_num_h, iter_num_w = math.ceil(height / base_h), math.ceil(width / base_w)
#            boundary_pixel_num_h, boundary_pixel_num_w = height % base_h, width % base_w
#            padding_pixel_num_h, padding_pixel_num_w = (base_h - boundary_pixel_num_h) % base_h, (
#                    base_w - boundary_pixel_num_w) % base_w
#
#            batch_sample = torch.nn.functional.pad(batch_sample, (0, padding_pixel_num_w, 0, padding_pixel_num_h),
#                                                   mode='replicate').to('cpu')
#            batch_label = torch.nn.functional.pad(batch_label, (0, padding_pixel_num_w, 0, padding_pixel_num_h),
#                                                  mode='replicate').to('cpu')
#
#            loss = 0
#            for i in range(iter_num_h):
#                for j in range(iter_num_w):
#                    model.to(device)
#                    offset_h, offset_w = i * base_h, j * base_w
#                    temp_sample = batch_sample[:, :, offset_h:base_h + offset_h, offset_w:base_w + offset_w]
#                    temp_label = batch_label[:, :, offset_h:base_h + offset_h, offset_w:base_w + offset_w]
#
#                    outputs = model(temp_sample.to(device))
#                    temp_sample.to('cpu')
#                    # 计算损失
#                    loss = criterion(outputs.to('cpu'), temp_label.argmax(dim=1))  # 注意需要将标签从独热编码转换为索引
#                    loss = loss.detach()
#                    # loss = 1 - dice_coeff(F.softmax(outputs.to('cpu'), dim=3), temp_label)
#                    test_total_loss += loss.item()
#                    test_dice_score += multiclass_dice_coeff(F.softmax(outputs.to('cpu'), dim=1), temp_label)
#                    test_dice_score = test_dice_score.detach()

    test_total_loss = val_total_loss
    test_dice_score = val_dice_score

    train_loss = total_loss / num_train
    train_dice = dice_score / num_train
    val_loss = val_total_loss / num_val
    val_dice = val_dice_score / num_val
    test_loss = test_total_loss / num_val
    test_dice = test_dice_score / num_val
    save_function(epoch+1, train_loss, train_dice.item(), val_loss, val_dice.item(), test_loss, test_dice.item())
    # 输出损失
    print(f'Epoch[{epoch + 1}/{num_epochs}]: Loss:{format(train_loss, ".3f")}, '
          f'dice_score:{format(train_dice.item(), ".3f")}, '
          f'val_loss:{format(val_loss, ".3f")}, '
          f'val_dice_score:{format(val_dice, ".3f")},'
          f'test_loss:{format(test_loss, ".3f")}, '
          f'test_dice_score:{format(test_dice, ".3f")}')

    print(num_train/(iter_num_h * iter_num_w),num_val/(iter_num_h * iter_num_w))
    if val_dice >= best_dice:
        best_dice = val_dice
        best_dice2 = test_dice
        torch.save(model.state_dict(), 'avbest_dice.pt')
    if val_loss <= min_loss:
        min_loss = val_loss
        torch.save(model.state_dict(), 'avmin_loss.pt')
    print('当前最高dice：', best_dice, best_dice2)

    
        # 调整学习率
    if best_dice >= 0.6:
        optimizer = optim.Adam(model.parameters(), lr=0.00003)
    if best_dice >= 0.70:
        optimizer = optim.Adam(model.parameters(), lr=0.00002)
    if best_dice >= 0.80:
        optimizer = optim.Adam(model.parameters(), lr=0.00002)
    if best_dice >= 0.85:
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
    if best_dice >= 0.88:
        optimizer = optim.Adam(model.parameters(), lr=0.000005)
    if best_dice >= 0.92:
        optimizer = optim.Adam(model.parameters(), lr=0.000002)
    if best_dice >= 0.94:
        optimizer = optim.Adam(model.parameters(), lr=0.000001)

