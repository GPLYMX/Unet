import math
import os
import random
import json
import yaml

from PIL import Image, ImageDraw, ImageEnhance
import torch
import cv2
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
# import albumentations as A
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt


def load_configs(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
configs = load_configs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs.yaml'))
unet_base = configs['unet_base']


def get_size(size, base=unet_base):
    """
    使得图片的尺寸为base的整数倍
    size为原始尺寸，格式是列表譬如[120, 300]
    输出格式为列表
    """
    base = float(base)
    rate1 = math.ceil(size[0] / base)
    rate2 = math.ceil(size[1] / base)
    return (int(rate1 * base), int(rate2 * base))


def resize_img(img, base=unet_base):
    """
    读取并返回修改好尺寸的图片
    base: unet下采样的次数为n，则base为2^n
    """
    try:
        if isinstance(img, Image.Image):
            width, height = img.size
            size = get_size([width, height], base=base)
            img = img.resize(size)
        if isinstance(img, torch.Tensor):
            # 获取原始张量的高度和宽度
            height, width = img.shape[1], img.shape[2]

            # # 计算需要添加的垂直和水平填充量
            # padding_height = (base - (height % base)) % base
            # padding_width = (base - (width % base)) % base

            #
            # # 使用 F.pad 函数添加填充，将高度和宽度调整为16的整数倍
            # img = torch.nn.functional.pad(img, (0, padding_width, 0, padding_height), mode='constant', value=189)

            # 计算要添加的填充量
            H_padding = (base - (height % base)) % base
            W_padding = (base - (width % base)) % base

            # 使用 'replicate' 模式进行填充
            img = torch.nn.functional.pad(img, (0, W_padding, 0, H_padding), mode='replicate')

        return img
    except Exception as e:
        print(e)
        # print('图片读取失败')
        return img


def gamma_correction(image, gamma=0.4):
    # 将图像转换为浮点型
    image = image.astype('float32') / 255.0

    # 进行伽马矫正
    corrected_image = np.power(image, gamma)

    # 将矫正后的图像转换回8位整数型
    corrected_image = (corrected_image * 255).astype('uint8')

    return corrected_image

    
# 将灰度图转换为 LabelMe 格式
def gray_to_labelme(gray_image, class_mapping={0: "background", 1: "tea", 2: "impurity", }):
    #     gray_image = cv2.imread(gray_image_root)
    #     gray_image = np.array(gray_image)

    height, width = gray_image.shape
    labelme_data = {
        "version": "5.3.1",
        "flags": {},
        "shapes": [],
        "imagePath": gray_image,  # 请替换成实际的图像文件名
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }

    for label_value, label_name in class_mapping.items():
        if label_name == "background":
            continue

        mask = (gray_image == label_value).astype(np.uint8)
        print(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            points = contour.squeeze().tolist()
            shape_data = {
                "label": label_name,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            }
            labelme_data["shapes"].append(shape_data)

    # 将数据保存为JSON文件
    #     with open('labelme_data.json', 'w') as json_file:
    #         json.dump(labelme_data, json_file)

    return labelme_data


class CustomSegmentationDataset(Dataset):

    def __init__(self, data_dir, transform=None, num_classes=3, mode='train', augment=True, channels=[0, 1, 2], channels2=[0, 1, 2],
                 label_format='json', low_pixel_test=False, pixel_shift_ratio=[0, 0]):
        """
        :param data_dir: 图片所在路径，一般是combined_data的上一级目录
        :param transform:
        :param num_classes:
        :param mode: mode=train时，__getitem__会返回处理后的训练图像和标签，mode=test时返回训练图片和文件名
        :param augment: 是否用图片拼接的方式做数据增强
        :param num_channel:输入的通道数
        :param label_format:代表标签的格式，标签是json文件时名字为'00.json'，放在combined_data文件夹下；标签是png时，名字是’label.png'，放在combined_data的文件夹的上一级目录下
        :param low_pixel_test:用于测试使用低分辨率高光谱是否可行
        :param pixel_shift_ratio:用于通道的偏移，【0.3， 0.8】表示偏移量为相邻通道差的0.3-0.8倍之间
        """
        self.data_dir = data_dir
        self.transform = transform
        self.file_names = [i for i in os.listdir(os.path.join(self.data_dir, 'images')) if i.endswith(('bmp', 'tif', 'tiff'))]
        self.num_classes = num_classes
        self.mode = mode
        self.augment = augment
        self.channels = channels
        self.channels2 = channels2
        self.label_format = label_format
        self.low_pixel_test = low_pixel_test
        self.pixel_shift_ratio = pixel_shift_ratio

    def __len__(self):
        return len(self.file_names)

    def read_gray_label(self, label):
        """
        读取label，并转化为独热编码格式的图片
        :param label_root:
        :return:
        """
        # label = Image.open(label_root).convert('L')
        label_image = transforms.ToTensor()(label)
        label_image = label_image * 255.

        # if self.transform:
        #     image = self.transform(image) / 255.0
        #     label_image = self.transform(label)
        # 创建独热编码的标签
        label_image_onehot = np.zeros((self.num_classes, label_image.shape[1], label_image.shape[2]))
        for class_idx in range(self.num_classes):
            label_image_onehot[class_idx, :, :] = (label_image == class_idx)
        label_image_onehot = torch.tensor(label_image_onehot, dtype=torch.float32)  # 转换为 PyTorch 张量
        # print('a', label_image_onehot[0,:,:].max())
        # print('b', label_image_onehot[1, :, :].max())
        # print('c', label_image_onehot[2, :, :].max())
        return label_image_onehot

    def json_to_gray(self, json_root):
        """
        读取labelme生成的json文件，然后转化成灰度图，灰度图中的数值代表像素点所属的类别
        """
        # # 1. 解析LabelMe标注文件（JSON格式）
        with open(json_root, 'r') as json_file:
            labelme_data = json.load(json_file)

        # 2. 获取图像尺寸
        image_width = labelme_data['imageWidth']
        image_height = labelme_data['imageHeight']

        # 3. 创建灰度图
        gray_image = Image.new('L', (image_width, image_height), 0)

        # 4. 为每个对象分配类别值
        category_mapping = {}  # 用于将类别名称映射到整数值
        category_id = 1

        for shape in labelme_data['shapes']:
            print(shape)
            category_name = shape['label']
            if category_name not in category_mapping:
                category_mapping[category_name] = category_id
                category_id += 1

            category_value = category_mapping[category_name]
            if isinstance(shape['points'][0], list):
                # 创建多边形的坐标列表
                polygon_points = [(int(x), int(y)) for x, y in shape['points']]

                # 使用PIL的绘图功能填充多边形区域
                draw = ImageDraw.Draw(gray_image)
                draw.polygon(polygon_points, fill=category_value)
        # 5. 保存灰度图
        gray_image = np.array(gray_image)
        gray_image = Image.fromarray(gray_image)
        # gray_image.save('output_gray_image.png')
        return gray_image

    def read_img(self, img_root):
        img_root = os.path.join(self.data_dir, 'images', img_root)
        # image = self.combine_img(img_root)
        # image = resize_img(image)
        image = Image.open(img_root)
        return image

    def read_label(self, label_name):
        """
        读取label的文件名，返回独热编码
        :param label_name:
        :return:
        """
        if self.label_format == 'json':
            label_name = os.path.join(self.data_dir, label_name, 'RGB.json')
            gray_img = self.labelme_to_gray(label_name)
            label_image_onehot = self.read_gray_label(gray_img)
        if self.label_format == 'png':
            label_name = os.path.splitext(label_name)[0] +'.png'
            label_name = os.path.join(self.data_dir, 'masks', label_name)
            # print(label_name)
            gray_image = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
            gray_image = np.array(gray_image)
            # gray_image[gray_image == 3] = 2
            gray_image = gray_image.astype(np.uint8)
            gray_image = Image.fromarray(gray_image)
            label_image_onehot = self.read_gray_label(gray_image)
        return label_image_onehot

    def augment_image(self, image, mask):
        """
        对图像和标签同时进行数据增强变换。
        参数：
        image (np.array): 多通道图像，形状为 (C, H, W)。
        mask (np.array): 标签图像。
        返回：
        augmented_image (np.array): 增强后的多通道图像。
        augmented_mask (np.array): 增强后的标签图像。
        """
        transform1 = A.Compose([A.HorizontalFlip(p=0.3), 
                              A.RandomRotate90(p=0.3)])
        transform2 = A.Compose([A.GaussianBlur(blur_limit=(1, 3), sigma_limit=0, always_apply=False, p=0.1), 
                              A.ShiftScaleRotate(shift_limit=0.06,scale_limit=0.1, rotate_limit=45, p=0.1), 
                              A.GaussNoise(var_limit=(1.0, 5.0), mean=0, per_channel=True, always_apply=False, p=0.1),
                              A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.1),
                              A.RandomGamma(gamma_limit=(80, 120), eps=1e-07, always_apply=False, p=0.1)])
        # 转换通道位置 (C, H, W) -> (H, W, C)
        image = np.transpose(image, (1, 2, 0))
        mask = np.transpose(mask, (1, 2, 0))
        # 应用数据增强
        augmented1 = transform1(image=image, mask=mask)
        # 转换回原来的通道位置 (H, W, C) -> (C, H, W)
        # augmented2 = transform2(image=augmented1['image'])
        augmented_image = np.transpose(augmented1['image'], (2, 0, 1))   
        return augmented_image, np.transpose(augmented1['mask'], (2, 0, 1))
    
    def __getitem__(self, idx):
        # 指定图像和标签的文件格式为PNG
        # file_lst = os.listdir(os.path.join(self.data_dir, 'images'))
        image = self.read_img(self.file_names[idx])
        image = transforms.ToTensor()(image)
        # print('______________', image.shape)
        if self.mode == 'test':
            return image, self.file_names[idx]
        try:
            label_image_onehot = self.read_label(self.file_names[idx])
            # print('______________', label_image_onehot.shape)
            if not self.augment:
                return image, label_image_onehot
            else:
                random_idx = random.randint(0, self.__len__() - 1)
                # print(self.file_names[random_idx])
                # file_lst = os.listdir(os.path.join(self.data_dir, self.file_names[random_idx]))
#                if 'pre_process' not in file_lst:
#                    for i in file_lst:
#                        if str(i)[0:4] == 'cube' and '.' not in str(i)[-5:]:
#                            self.file_names[random_idx] = os.path.join(self.file_names[random_idx], i)
#                            break
                image1, label_image_onehot1 = self.read_img(self.file_names[random_idx]), self.read_label(self.file_names[random_idx])
                image1 = transforms.ToTensor()(image1)
                # print('___________',image1.shape, label_image_onehot1.shape)
                # image = np.array(image)
                # label_image_onehot  = np.array(label_image_onehot)
                # image, label_image_onehot = self.augment_image(image, label_image_onehot)
                # image = torch.from_numpy(image.copy())
                # label_image_onehot  = torch.from_numpy(label_image_onehot.copy())
                # 随机生成一个矩形框
                h, w = min(image.shape[1], image1.shape[1]), min(image.shape[2], image1.shape[2])  # 两张图片的大小可能不相同，需要选择尺寸最小的
                h1 = random.randint(0, h - 3)
                h2 = random.randint(h1 + 1, h - 1)
                w1 = random.randint(0, w - 3)
                w2 = random.randint(w1 + 1, w - 1)
                image[:, h1:h2, w1:w2] = image1[:, h1:h2, w1:w2]
                label_image_onehot[:, h1:h2, w1:w2] = label_image_onehot1[:, h1:h2, w1:w2]

#                try:
#                    image[:, h1:h2, w1:w2] = image1[:, h1:h2, w1:w2]
#                    label_image_onehot[:, h1:h2, w1:w2] = label_image_onehot1[:, h1:h2, w1:w2]
#                except RuntimeError:
#                    # 当两张图片大小不匹配时，不进行数据增强
#                    pass

                return image, label_image_onehot

        except FileNotFoundError:
            """
            测试模式下不需要label信息，直接返回image
            """
            print(self.file_names[idx])
            return image, self.file_names[idx]
        else:
            print(self.file_names[idx])
            # return self.read_img(self.file_names[idx-1]), self.read_label(self.file_names[idx-1])

    def combine_img(self, folder_path):
        # 获取文件夹中的所有图像文件名
#        print(folder_path)
        image_files = [filename for filename in os.listdir(folder_path) if filename.endswith(".png")]
        image_files.sort()
#        print(image_files)
#        print(self.channels)
        image_files = [image_files[i] for i in self.channels]

        # 加载灰度图像并添加到列表中
        image_files.sort()
        image_list = []
        for idx, img_path in enumerate(image_files):
            if idx <= 2:
                img = Image.open(os.path.join(folder_path, img_path)).convert("L")  # 将图像转换为灰度模式
                # img = np.array(img)         
                # img = gamma_correction(img)
                # img = Image.fromarray(img)
            else:
                if self.low_pixel_test:
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
        if len(self.channels2) != 0:
            image_list2 = []
            for n in os.listdir(os.path.dirname(folder_path)):
                if n[0:4] == 'cube':
                    folder_path2 = os.path.join(os.path.dirname(folder_path), n, 'pre_process')
                    break

            image_files2 = [filename for filename in os.listdir(folder_path2) if filename.endswith(".png")]
            image_files2.sort()
            #        print(image_files)
            #        print(self.channels)
            image_files2 = [image_files2[i] for i in self.channels2]

            # 加载灰度图像并添加到列表中
            image_files2.sort()
            image_list2 = []
            for idx, img_path2 in enumerate(image_files2):
                if idx <= 2:
                    img = Image.open(os.path.join(folder_path2, img_path2)).convert("L")  # 将图像转换为灰度模式
                    # img = np.array(img)
                    # img = gamma_correction(img)
                    # img = Image.fromarray(img)
                else:
                    if self.low_pixel_test:
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
                image_list2.append(img)
            image_list = image_list + image_list2

        # 确定图像的尺寸（假设所有图像都有相同的尺寸）
        width, height = image_list[0].size

        # 创建一个空的PyTorch张量，用于存储多通道图像
        multi_channel_image = torch.zeros(len(image_list), height, width)

        # 将灰度图像的像素数据叠加到PyTorch张量中
        for i, img in enumerate(image_list):
            img_tensor = transforms.ToTensor()(img)  # 将PIL图像转换为PyTorch张量
            multi_channel_image[i] = img_tensor[0]  # 仅使用灰度通道数据

        # 添加随机偏移
#        random_rate = [random.uniform(self.pixel_shift_ratio[0], self.pixel_shift_ratio[1]) for i in
#                       range(len(image_files))]  # 偏移率列表
#        random_array = [multi_channel_image[i + 1] - multi_channel_image[i] for i in range(len(image_files) - 1)]
#        random_array1 = random_array.copy()
#        random_array2 = random_array.copy()
#        random_array1.append(random_array[0])  # 第一位插值
#        random_array2.append(random_array[-1])  # 最后一位插值
#        random_array = [random_array1[i] * random_rate[i] if random_rate[i]<0 else random_array2[i] * random_rate[i] for i in range(len(image_files))]
#        for i in range(3, len(image_files)):
#            multi_channel_image[i] = multi_channel_image[i] + random_array[i]

        return multi_channel_image

    def labelme_to_gray(self, labelme_json_root):
        """
        读取labelme生成的json文件，然后转化成灰度图，灰度图中的数值代表像素点所属的类别
        """
        # # 1. 解析LabelMe标注文件（JSON格式）
        with open(labelme_json_root, 'r') as json_file:
            labelme_data = json.load(json_file)

        # 2. 获取图像尺寸
        image_width = labelme_data['imageWidth']
        image_height = labelme_data['imageHeight']

        # 3. 创建灰度图
        gray_image = Image.new('L', (image_width, image_height), 0)

        # 4. 为每个对象分配类别值
        category_mapping = {}  # 用于将类别名称映射到整数值
        category_id = 1

        for shape in labelme_data['shapes']:
            category_name = shape['label']
            if category_name not in category_mapping:
                category_mapping[category_name] = category_id
                category_id += 1

            category_value = 0
            if shape['label'] == '1' or shape['label'] == 'tea':
                category_value = 1
            if shape['label'] == '2' or shape['label'] == '3' or shape['label'] == 'impurity':
                category_value = 2
            if isinstance(shape['points'][0], list):
                # 创建多边形的坐标列表
                polygon_points = [(int(x), int(y)) for x, y in shape['points']]

                # 使用PIL的绘图功能填充多边形区域
                draw = ImageDraw.Draw(gray_image)
                draw.polygon(polygon_points, fill=category_value)
        # 5. 保存灰度图
        gray_image = np.array(gray_image)
        gray_image = Image.fromarray(gray_image)
        # gray_image.save('output_gray_image.png')
        return gray_image


def visualization(torch_matrix):
    """
    将模型训练出来的torch格式矩阵[1*3*h*w]转化为plt格式的图片
    :return:
    """
    torch_matrix = torch_matrix.squeeze(0)
    try:
        np_matrix = torch_matrix.detach().numpy()
    except TypeError:
        torch_matrix = torch_matrix.to('cpu')
        np_matrix = torch_matrix.detach().numpy()

    max_channel_indices = np.argmax(np_matrix, axis=0)

    # 第一个通道最大的情况，将所有值设为0
    np_matrix[0, max_channel_indices == 0] = 0
    np_matrix[1, max_channel_indices == 0] = 0
    np_matrix[2, max_channel_indices == 0] = 0
    # 第二个通道最大的情况，标记为黄色
    np_matrix[0, max_channel_indices == 1] = 255
    np_matrix[1, max_channel_indices == 1] = 255
    np_matrix[2, max_channel_indices == 1] = 0
    # 第三个通道最大的情况，标记为红色
    np_matrix[0, max_channel_indices == 2] = 255
    np_matrix[1, max_channel_indices == 2] = 0
    np_matrix[2, max_channel_indices == 2] = 0

    np_matrix = np.transpose(np_matrix, (1, 2, 0))

    return np_matrix


def dice_coeff(predicted, target, epsilon=1e-5):
    predicted = predicted.squeeze(0)
    target = target.squeeze(0)
    try:
        predicted = predicted.detach().numpy()
    except TypeError:
        predicted = predicted.to('cpu')
        np_matrix = predicted.detach().numpy()
        target = target.to('cpu')

    max_channel_indices = np.argmax(np_matrix, axis=0)

    # 第一个通道最大的情况，将所有值设为0
    np_matrix[0, max_channel_indices == 0] = 1
    np_matrix[1, max_channel_indices == 0] = 0
    np_matrix[2, max_channel_indices == 0] = 0
    # 第二个通道最大的情况，标记为黄色
    np_matrix[0, max_channel_indices == 1] = 0
    np_matrix[1, max_channel_indices == 1] = 1
    np_matrix[2, max_channel_indices == 1] = 0
    # 第三个通道最大的情况，标记为红色
    np_matrix[0, max_channel_indices == 2] = 0
    np_matrix[1, max_channel_indices == 2] = 0
    np_matrix[2, max_channel_indices == 2] = 1

    predicted = torch.tensor(np_matrix)

    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target) + epsilon
    dice = (2.0 * intersection) / union
    return dice


class LabelMeDataset(Dataset):
    def __init__(self, data_dir, label_map=None, augment=True):
        """
        初始化LabelMe数据集类

        :param data_dir: 包含图像和labelme标注文件的目录路径
        :param label_map: 标签映射字典，例如 {1: 0} 将标签1转换为标签0
        :param transform: 可选的数据转换操作，例如对图像和标签进行相同的变换
        :param augment: 是否启用数据增强（默认启用）
        """
        self.data_dir = data_dir
        self.image_paths = sorted([os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if
                                   fname.endswith(('.tiff', '.bmp', '.png', '.jpg', '.tif'))])
        self.label_map = label_map if label_map else {'0':0, '1':0, '2':1}
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        获取一个样本，包括图像和标签
        """
        # 获取图像路径
        image_path = self.image_paths[idx]
        img = Image.open(image_path).convert('RGB')

        # 获取对应的labelme JSON标签路径
        label_path = self.get_label_path(image_path)
        with open(label_path, 'r') as f:
            data = json.load(f)

        # 创建空的掩码图，初始化为全0
        mask = np.zeros((img.height, img.width), dtype=np.uint8)

        # 遍历json中的标注，提取多边形并更新掩码
        for shape in data['shapes']:
            label = shape['label']
            points = np.array(shape['points'], dtype=np.int32)
            # 使用OpenCV绘制多边形，填充指定的标签
            cv2.fillPoly(mask, [points], color=self.get_mapped_label(label))
            # print(self.get_mapped_label(label))

        # 将掩码转换为PIL图像
        mask = Image.fromarray(mask)
        # print('befer tran', np.max(np.array(mask)))
        # 数据增强（如果启用）
        if self.augment:
            img, mask = self.augment_data(img, mask)
        # print('tran', np.max(np.array(mask)))
        # 应用其他转换（例如ToTensor等）
        if self.transform:
            img = self.transform(img)
            mask = torch.tensor(np.array(mask))

        # print('after tran', np.max(np.array(mask)))
        return img, mask

    def get_label_path(self, image_path):
        """
        根据图像路径获取对应的labelme JSON文件路径
        假设标签文件和图像文件有相同的文件名，只是扩展名不同
        """
        base_name = os.path.splitext(os.path.basename(image_path))[0]  # 获取文件名，不带扩展名
        label_path = os.path.join(self.data_dir, f'{base_name}.json')  # 组合得到对应的json文件路径
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file for image {image_path} not found.")
        return label_path

    def get_mapped_label(self, label):
        """
        获取标签的映射值，如果在label_map中找到，返回映射的值，否则返回原始标签
        """
        return self.label_map.get(label, label)

    def augment_data(self, img, mask):
        """
        启用数据增强，包括翻转、旋转、裁剪、颜色增强等
        :param img: 输入图像
        :param mask: 输入标签掩码
        :return: 增强后的图像和掩码
        """
        # 随机水平翻转
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # 随机垂直翻转
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        # 随机旋转（-30 到 30度）
        angle = random.uniform(-30, 30)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True)
        mask = mask.rotate(angle, resample=Image.BICUBIC, expand=True)

        # # 随机裁剪
        # width, height = img.size
        # new_width = random.randint(int(0.8 * width), width)
        # new_height = random.randint(int(0.8 * height), height)
        # left = random.randint(0, width - new_width)
        # top = random.randint(0, height - new_height)
        # img = img.crop((left, top, left + new_width, top + new_height))
        # mask = mask.crop((left, top, left + new_width, top + new_height))

        # 随机调整亮度
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))  # 随机亮度增强

        # 随机调整对比度
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))  # 随机对比度增强

        # 随机调整饱和度
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))  # 随机饱和度增强
        # print((np.array(img).shape),(np.array(mask).shape))
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(mask)
        # print(np.max(mask), np.array(mask).shape)
        # plt.show()

        return img, mask

if __name__ == '__main__':
    # 使用Dataset和DataLoader
    data_dir = r'G:\datas\data\dayi\linescan\20250114yolo_90us_8k\unet_label'  # 图像和标签文件夹路径

    dataset = LabelMeDataset(data_dir=data_dir)

    # 创建DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 示例：遍历DataLoader
    for imgs, masks in dataloader:
        print(np.max(np.array(masks)))
        masks = masks.squeeze(0)
        plt.imshow(np.array(masks))
        plt.show()
        pass
        # print(imgs.shape, masks.shape)  # 输出图像和掩码的形状

