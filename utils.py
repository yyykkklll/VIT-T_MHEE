import os
import os.path as osp
import sys
import numpy as np
import torch
from torch.utils.data import Sampler  # 修复：添加缺失的 Sampler 导入
import errno

def load_data(input_data_path):
    """
    加载数据文件中的图像路径和标签
    
    Args:
        input_data_path (str): 输入数据文件路径
    
    Returns:
        tuple: 图像路径列表和对应标签列表
    """
    with open(input_data_path, 'rt') as f:
        data = f.read().splitlines()
    file_image = [s.split(' ')[0] for s in data]
    file_label = [int(s.split(' ')[1]) for s in data]
    return file_image, file_label

def gen_idx(train_color_label, train_thermal_label):
    """
    为每个身份生成索引位置
    
    Args:
        train_color_label (list): 可见光模态标签
        train_thermal_label (list): 热成像模态标签
    
    Returns:
        tuple: 可见光和热成像的身份索引列表
    """
    color_pos = [[k for k, v in enumerate(train_color_label) if v == label] for label in np.unique(train_color_label)]
    thermal_pos = [[k for k, v in enumerate(train_thermal_label) if v == label] for label in np.unique(train_thermal_label)]
    return color_pos, thermal_pos

def gen_cam_idx(gall_img, gall_label, mode):
    """
    为图库生成摄像头索引
    
    Args:
        gall_img (list): 图库图像路径
        gall_label (list): 图库标签
        mode (str): 模式 ('indoor' 或 'all')
    
    Returns:
        list: 每个身份-摄像头组合的索引
    """
    cam_idx = [1, 2] if mode == 'indoor' else [1, 2, 4, 5]
    gall_cam = [int(img[-10]) for img in gall_img]
    return [[k for k, v in enumerate(gall_label) if v == label and gall_cam[k] == cam] 
            for label in np.unique(gall_label) for cam in cam_idx if any(k for k, v in enumerate(gall_label) if v == label and gall_cam[k] == cam)]

def extract_cam(gall_img):
    """
    提取图库图像的摄像头ID
    
    Args:
        gall_img (list): 图库图像路径
    
    Returns:
        np.ndarray: 摄像头ID数组
    """
    return np.array([int(img[-10]) for img in gall_img])

class IdentitySampler(Sampler):
    """
    均匀采样每个批次中的身份
    
    Args:
        train_color_label (list): 可见光模态标签
        train_thermal_label (list): 热成像模态标签
        color_pos (list): 可见光身份索引
        thermal_pos (list): 热成像身份索引
        num_pos (int): 每个身份的正样本数量
        batch_size (int): 批次大小
        epoch (int): 当前 epoch
    """
    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batch_size, epoch):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)
        N = max(len(train_color_label), len(train_thermal_label))
        index1, index2 = [], []
        for j in range(int(N / (batch_size * num_pos)) + 1):
            batch_idx = np.random.choice(uni_label, batch_size, replace=False)
            for i in range(batch_size):
                index1.extend(np.random.choice(color_pos[batch_idx[i]], num_pos, replace=False))
                index2.extend(np.random.choice(thermal_pos[batch_idx[i]], num_pos, replace=False))
        self.index1 = np.array(index1)
        self.index2 = np.array(index2)
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N

class AverageMeter:
    """
    计算并存储平均值和当前值
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mkdir_if_missing(directory):
    """
    创建目录，如果不存在则忽略EEXIST错误
    
    Args:
        directory (str): 目录路径
    """
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

class Logger:
    """
    将控制台输出写入外部文本文件
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = open(fpath, 'w') if fpath else None
        mkdir_if_missing(osp.dirname(fpath) if fpath else '')

    def __del__(self):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        if self.file:
            self.file.close()

def set_seed(seed, cuda=True):
    """
    设置随机种子
    
    Args:
        seed (int): 随机种子
        cuda (bool): 是否为CUDA设置种子
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def set_requires_grad(nets, requires_grad=False):
    """
    设置网络参数是否需要梯度
    
    Args:
        nets (list or nn.Module): 网络列表或单个网络
        requires_grad (bool): 是否需要梯度
    """
    nets = [nets] if not isinstance(nets, list) else nets
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad