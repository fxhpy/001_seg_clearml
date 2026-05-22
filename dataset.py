import os #
import torch
import random
import numpy as np
import torchvision.transforms.functional as F ## 实现数据增强和预处理
import torchvision.transforms as T ## 图像数据增强和预处理

from PIL import Image ## Python 图像库，用于读取图片
from torch.utils.data import Dataset

def preprocess_image(img, mask):
    # 随机水平翻转
    if random.random() > 0.5:
        img = F.hflip(img)
        mask = F.hflip(mask)

    # 随机垂直翻转
    if random.random() > 0.5:
        img = F.vflip(img)
        mask = F.vflip(mask)

    # 随机旋转
    angle = random.uniform(-30, 30)
    img = F.rotate(img, angle)
    mask = F.rotate(mask, angle)

    # 随机缩放
    scale_factor = random.uniform(0.8, 1.2)
    img = F.resize(img, [int(scale_factor * img.size[1]), int(scale_factor * img.size[0])])
    mask = F.resize(mask, [int(scale_factor * mask.size[1]), int(scale_factor * mask.size[0])])

    # 随机裁剪

    i, j, h, w = T.RandomCrop.get_params(img, (720, 1280))
    
    img = F.crop(img, i, j, h, w)
    mask = F.crop(mask, i, j, h, w)

    # 随机颜色调整
    if random.random() > 0.5:
        brightness_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)
        saturation_factor = random.uniform(0.8, 1.2) ## 调整饱和度的因子
        hue_factor = random.uniform(-0.1, 0.1) ## 调整色调的因子
        img = F.adjust_brightness(img, brightness_factor)
        img = F.adjust_contrast(img, contrast_factor)
        img = F.adjust_saturation(img, saturation_factor)
        img = F.adjust_hue(img, hue_factor)

    return img, mask

class DroneDataset(Dataset):
    def __init__(self, root,need_handle=False):
        self.img_path = os.path.join(root, 'imgs')
        self.mask_path = os.path.join(root, 'masks')
        self.image_filenames = os.listdir(self.img_path)
        self.mask_filenames = os.listdir(self.mask_path)
        #self.mean=[0.485, 0.456, 0.406]
        #self.std=[0.229, 0.224, 0.225] ## ImageNet 预训练模型的标准值
        self.mean=[0.5, 0.5, 0.5]
        self.std=[0.5, 0.5, 0.5]

        self.input_size = 512
        self.need_handle=need_handle ## 是否启用数据增强

    def __len__(self):
        return len(self.image_filenames) ## 返回数据集大小，供 DataLoader 调用
    
    def __getitem__(self, index):

        img = Image.open(os.path.join(self.img_path, self.image_filenames[index])).convert('RGB')

        mask = Image.open(os.path.join(self.mask_path, self.mask_filenames[index])).convert('L')

        if self.need_handle:
            img, mask = preprocess_image(img, mask) ## 数据增强
            
        img = F.resize(img, [self.input_size,self.input_size])

        mask = F.resize(mask, [self.input_size,self.input_size])

        img_array = np.array(img)        
        mask_array = np.array(mask)  ## 转换为numpy数组
        mask_array = np.where(mask_array == 4, 3, mask_array)  ## 将类别4归入Ignore(3)

        img_array = (img_array / 255.0 - self.mean) / self.std  ## 归一化处理

        ## 创建PyTorch张量 ##
        image_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() ## permute 重排维度顺序
        ## PIL/numpy 格式：(H, W, C) 转为 PyTorch 格式：(C, H, W)
        mask_tensor = torch.from_numpy(mask_array).long()

        return image_tensor, mask_tensor