# 文件：data_loader.py
import os.path as osp
import numpy as np
from PIL import Image
import torch.utils.data as data

# 全局数据集根目录，使用常量定义，避免冗余字典
SYSU_ROOT = '/home/s-sunxc/LLCM-main/data/SYSU-MM01/'
LLCM_ROOT = '/home/s-sunxc/LLCM-main/data/LLCM/'
REGDB_ROOT = '/data/RegDB/'

def load_data(input_data_path):
    """从文本文件加载图像路径和标签，简化实现"""
    with open(input_data_path, 'rt') as f:
        lines = f.read().splitlines()
    return [line.split(' ')[0] for line in lines], [int(line.split(' ')[1]) for line in lines]

def load_and_resize_images(data_dir, img_files, size=(384, 384)):
    """加载并调整图像大小，统一处理函数，避免重复代码"""
    images = [np.array(Image.open(osp.join(data_dir, path)).resize(size, Image.Resampling.LANCZOS)) 
              for path in img_files]
    return np.array(images)

class BaseDataset(data.Dataset):
    """基础数据集类，提取公共逻辑，减少冗余"""
    def __init__(self, color_images, color_labels, thermal_images, thermal_labels, 
                 transform=None, cIndex=None, tIndex=None):
        self.color_images = color_images
        self.color_labels = color_labels
        self.thermal_images = thermal_images
        self.thermal_labels = thermal_labels
        self.transform = transform
        self.cIndex = cIndex if cIndex is not None else list(range(len(color_images)))
        self.tIndex = tIndex if tIndex is not None else list(range(len(thermal_images)))

    def __getitem__(self, index):
        """获取图像对及其标签，统一处理RGB和红外"""
        # 确保索引在有效范围内
        c_idx = self.cIndex[index] if index < len(self.cIndex) else self.cIndex[-1]
        t_idx = self.tIndex[index] if index < len(self.tIndex) else self.tIndex[-1]
        
        # 防止越界
        c_idx = min(max(0, c_idx), len(self.color_images) - 1)
        t_idx = min(max(0, t_idx), len(self.thermal_images) - 1)
        
        img1, target1 = self.color_images[c_idx], self.color_labels[c_idx]
        img2, target2 = self.thermal_images[t_idx], self.thermal_labels[t_idx]
        img1 = Image.fromarray(img1.astype(np.uint8))
        img2 = Image.fromarray(img2.astype(np.uint8))
        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)
        return img1, img2, target1, target2

    def __len__(self):
        """返回数据集大小"""
        return len(self.cIndex)

class SYSUData(BaseDataset):
    """SYSU-MM01数据集类，继承BaseDataset，加载预处理数据"""
    def __init__(self, transform=None, colorIndex=None, thermalIndex=None):
        color_images = np.load(osp.join(SYSU_ROOT, 'train_rgb_resized_img.npy'))
        color_labels = np.load(osp.join(SYSU_ROOT, 'train_rgb_resized_label.npy'))
        thermal_images = np.load(osp.join(SYSU_ROOT, 'train_ir_resized_img.npy'))
        thermal_labels = np.load(osp.join(SYSU_ROOT, 'train_ir_resized_label.npy'))
        super().__init__(color_images, color_labels, thermal_images, thermal_labels, 
                         transform, colorIndex, thermalIndex)

class RegDBData(BaseDataset):
    """RegDB数据集类，动态加载并处理图像"""
    def __init__(self, trial, transform=None, colorIndex=None, thermalIndex=None):
        color_files, color_labels = load_data(osp.join(REGDB_ROOT, 'idx', f'train_visible_{trial}.txt'))
        thermal_files, thermal_labels = load_data(osp.join(REGDB_ROOT, 'idx', f'train_thermal_{trial}.txt'))
        color_images = load_and_resize_images(REGDB_ROOT, color_files)
        thermal_images = load_and_resize_images(REGDB_ROOT, thermal_files)
        super().__init__(color_images, color_labels, thermal_images, thermal_labels, 
                         transform, colorIndex, thermalIndex)

class LLCMData(BaseDataset):
    """LLCM数据集类，动态加载VIS和NIR图像"""
    def __init__(self, transform=None, colorIndex=None, thermalIndex=None):
        color_files, color_labels = load_data(osp.join(LLCM_ROOT, 'idx', 'train_vis.txt'))
        thermal_files, thermal_labels = load_data(osp.join(LLCM_ROOT, 'idx', 'train_nir.txt'))
        color_images = load_and_resize_images(LLCM_ROOT, color_files)
        thermal_images = load_and_resize_images(LLCM_ROOT, thermal_files)
        super().__init__(color_images, color_labels, thermal_images, thermal_labels, 
                         transform, colorIndex, thermalIndex)

class TestData(data.Dataset):
    """测试数据集类，简化并统一处理逻辑"""
    def __init__(self, test_img_file, test_label, transform=None, img_size=(384, 384)):
        if isinstance(test_img_file, list):  # 如果是路径列表，动态加载
            self.test_images = load_and_resize_images('', test_img_file, img_size)
        else:  # 如果是NumPy数组，直接使用
            self.test_images = test_img_file
        self.test_labels = test_label
        self.transform = transform

    def __getitem__(self, index):
        """获取单张测试图像及其标签"""
        img = Image.fromarray(self.test_images[index].astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        return img, self.test_labels[index]

    def __len__(self):
        """返回测试数据集大小"""
        return len(self.test_images)