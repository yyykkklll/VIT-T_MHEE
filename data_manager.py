# 文件：data_manager.py
from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
import random
import os

def load_ids(file_path):
    """从文本文件加载身份ID并格式化为4位字符串"""
    with open(file_path, 'r') as f:
        ids = [int(x) for x in f.read().splitlines()[0].split(',')]
    return [f"{x:04d}" for x in ids]

def GenIdx(color_labels, thermal_labels):
    """
    为每个身份生成对应的图像索引
    
    Args:
        color_labels (list or np.ndarray): 可见光图像的标签列表
        thermal_labels (list or np.ndarray): 红外图像的标签列表
    
    Returns:
        tuple: (color_pos, thermal_pos)
            - color_pos: 字典,键为身份ID,值为该身份对应的可见光图像索引列表
            - thermal_pos: 字典,键为身份ID,值为该身份对应的红外图像索引列表
    """
    color_pos = {}
    thermal_pos = {}

    # 遍历可见光标签,生成索引
    for idx, label in enumerate(color_labels):
        if label not in color_pos:
            color_pos[label] = []
        color_pos[label].append(idx)

    # 遍历红外标签,生成索引
    for idx, label in enumerate(thermal_labels):
        if label not in thermal_pos:
            thermal_pos[label] = []
        thermal_pos[label].append(idx)

    return color_pos, thermal_pos

def process_images(data_path, cameras, ids, single=False):
    """
    通用函数：处理图像路径,仅返回图像路径列表
    
    Args:
        data_path (str): 数据集根目录
        cameras (list): 相机列表
        ids (list): 身份ID列表
        single (bool): 是否为每个身份随机选择一张图像
    
    Returns:
        list: 图像路径列表
    """
    img_paths = []
    for id in sorted(ids):
        for cam in cameras:
            img_dir = osp.join(data_path, cam, id)
            if osp.isdir(img_dir):
                files = sorted([osp.join(img_dir, f) for f in os.listdir(img_dir)])
                img_paths.extend([random.choice(files)] if single else files)
    
    return img_paths

def process_query_sysu(data_path, mode='all'):
    """
    处理 SYSU-MM01 数据集的查询集（红外图像）
    
    Args:
        data_path (str): 数据集根目录
        mode (str): 'all' 或 'indoor'
    
    Returns:
        tuple: (query_img, query_label, query_cam)
            - query_img: 查询图像路径列表
            - query_label: 查询标签列表
            - query_cam: 查询摄像头ID列表
    """
    ir_cameras = ['cam3', 'cam6']  # 查询集使用红外相机
    ids = load_ids(osp.join(data_path, 'exp/test_id.txt'))
    img_paths = process_images(data_path, ir_cameras, ids, single=True)
    
    # SYSU格式：camID和ID从路径中提取
    cam_ids = [int(p[-15]) for p in img_paths]  # SYSU格式：camID在倒数第15位
    pids = [int(p[-13:-9]) for p in img_paths]  # SYSU格式：ID在倒数13到9位
    return img_paths, np.array(pids), np.array(cam_ids)

def process_gallery_sysu(data_path, mode='all', trial=0):
    """
    处理 SYSU-MM01 数据集的图库集（可见光图像）
    
    Args:
        data_path (str): 数据集根目录
        mode (str): 'all' 或 'indoor'
        trial (int): 试验编号
    
    Returns:
        tuple: (gall_img, gall_label, gall_cam)
    """
    rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5'] if mode == 'all' else ['cam1', 'cam2']
    ids = load_ids(osp.join(data_path, 'exp/test_id.txt'))
    random.seed(trial)  # 确保随机选择可重现
    img_paths = process_images(data_path, rgb_cameras, ids, single=False)
    
    # SYSU格式：camID和ID从路径中提取
    cam_ids = [int(p[-15]) for p in img_paths]  # SYSU格式：camID在倒数第15位
    pids = [int(p[-13:-9]) for p in img_paths]  # SYSU格式：ID在倒数13到9位
    return img_paths, np.array(pids), np.array(cam_ids)

def process_query_llcm(data_path, mode=1):
    """
    处理 LLCM 数据集的查询集
    
    Args:
        data_path (str): 数据集根目录
        mode (int): 模式 (1 为 VIS,2 为 NIR)
    
    Returns:
        tuple: (query_img, query_label, query_cam)
    """
    prefix = 'test_vis' if mode == 1 else 'test_nir'
    cameras = [f'{prefix}/cam{i}' for i in ([1, 2, 3, 4, 5, 6, 7, 8, 9] if mode == 1 else [1, 2, 4, 5, 6, 7, 8, 9])]
    ids = load_ids(osp.join(data_path, 'idx/test_id.txt'))
    img_paths = process_images(data_path, cameras, ids, single=True)
    
    # LLCM格式：camID和ID从路径中的'camX_XXXX'提取
    cam_ids = [int(p.split('cam')[1][0]) for p in img_paths]
    pids = [int(p.split('cam')[1][2:6]) for p in img_paths]
    return img_paths, np.array(pids), np.array(cam_ids)

def process_gallery_llcm(data_path, mode=1, trial=0):
    """
    处理 LLCM 数据集的图库集
    
    Args:
        data_path (str): 数据集根目录
        mode (int): 模式 (1 为 VIS,2 为 NIR)
        trial (int): 试验编号
    
    Returns:
        tuple: (gall_img, gall_label, gall_cam)
    """
    prefix = 'test_nir' if mode == 1 else 'test_vis'  # 查询VIS时图库用NIR,反之亦然
    cameras = [f'{prefix}/cam{i}' for i in ([1, 2, 4, 5, 6, 7, 8, 9] if mode == 1 else [1, 2, 3, 4, 5, 6, 7, 8, 9])]
    ids = load_ids(osp.join(data_path, 'idx/test_id.txt'))
    random.seed(trial)
    img_paths = process_images(data_path, cameras, ids, single=False)
    
    # LLCM格式：camID和ID从路径中的'camX_XXXX'提取
    cam_ids = [int(p.split('cam')[1][0]) for p in img_paths]
    pids = [int(p.split('cam')[1][2:6]) for p in img_paths]
    return img_paths, np.array(pids), np.array(cam_ids)

def process_test_regdb(img_dir, trial=1, modal='visible'):
    """
    处理 RegDB 数据集的测试集
    
    Args:
        img_dir (str): 数据集根目录
        trial (int): 试验编号
        modal (str): 'visible' 或 'thermal'
    
    Returns:
        tuple: (test_img, test_label)
    """
    file_path = osp.join(img_dir, 'idx', f'test_{modal}_{trial}.txt')
    with open(file_path, 'rt') as f:
        lines = f.read().splitlines()
    img_paths = [osp.join(img_dir, line.split(' ')[0]) for line in lines]
    labels = [int(line.split(' ')[1]) for line in lines]
    return img_paths, np.array(labels)