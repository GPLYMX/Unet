# -*- coding: utf-8 -*-
# @Time : 2023/9/19 14:22
# @Author : GuoPeng
import yaml

import torch

from single_img_predict import get_impurity_mask
from models.network import U_Net
from utiles import postprocessing


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
global model
model = U_Net(img_ch=in_channels, output_ch=num_classes)
model.load_state_dict(torch.load('temp.pt'))


def main(path):
    """

    :param path: 十三通道图片所在的文件夹
    :return: 杂质们的坐标点，坐标点规则：左上角为原点，先纵轴、后横轴。
    """
    gray_img = get_impurity_mask(path, model)
    coordinates = postprocessing(gray_img)


if __name__ == "__main__":
    root = r'D:\mycodes\RITH\puer\puer_json\data_20230901_2\data\test\3-test\combined_data'
    main(root)