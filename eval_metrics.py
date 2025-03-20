from __future__ import print_function, absolute_import
import numpy as np

def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20, same_cam_exclude=True):
    """
    通用的跨模态ReID评估函数，计算CMC、mAP和mINP指标
    
    Args:
        distmat (np.ndarray): 距离矩阵，形状为(num_q, num_g)，表示查询与图库间的距离
        q_pids (np.ndarray): 查询集身份ID数组，长度为num_q
        g_pids (np.ndarray): 图库集身份ID数组，长度为num_g
        q_camids (np.ndarray): 查询集相机ID数组，长度为num_q
        g_camids (np.ndarray): 图库集相机ID数组，长度为num_g
        max_rank (int): 计算CMC的最大排名，默认20
        same_cam_exclude (bool): 是否排除相同相机视图的图库样本，默认True
    
    Returns:
        tuple: (cmc, mAP, mINP)
            - cmc (np.ndarray): 平均CMC曲线，长度为max_rank
            - mAP (float): 平均精度均值
            - mINP (float): 平均归一化精度均值
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g  # 调整max_rank以适应较小的图库规模
        print(f"Note: gallery size ({num_g}) is smaller than max_rank, adjusted to {max_rank}")

    # 对距离矩阵按行排序，获取排序索引
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)  # 二进制匹配矩阵

    all_cmc, all_AP, all_INP = [], [], []
    num_valid_q = 0

    for q_idx in range(num_q):
        q_pid, q_camid = q_pids[q_idx], q_camids[q_idx]
        order = indices[q_idx]

        # 排除相同相机视图的样本（若启用）
        keep = np.ones(num_g, dtype=bool) if not same_cam_exclude else (g_camids[order] != q_camid)
        
        # 获取当前查询的匹配结果
        cmc = matches[q_idx][keep]
        if not np.any(cmc):  # 若无匹配，跳过此查询
            continue

        # 计算CMC曲线
        cmc = cmc.cumsum()
        cmc[cmc > 1] = 1  # 限制CMC值为0或1
        all_cmc.append(cmc[:max_rank])

        # 计算mINP（参考"Deep Learning for Person Re-identification"）
        pos_idx = np.where(cmc == 1)[0]
        inp = cmc[pos_idx[-1]] / (pos_idx[-1] + 1.0)  # 用最后一个正样本位置计算
        all_INP.append(inp)

        # 计算平均精度（参考Wikipedia的AP定义）
        num_rel = cmc.sum()
        tmp_cmc = cmc.cumsum() / (np.arange(len(cmc)) + 1.0)  # 每位置的精度
        AP = (tmp_cmc * cmc).sum() / num_rel  # 只对正样本位置求和
        all_AP.append(AP)

        num_valid_q += 1

    assert num_valid_q > 0, "Error: no valid queries found in gallery"

    # 计算平均指标
    cmc = np.mean(np.array(all_cmc, dtype=np.float32), axis=0)
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return cmc, mAP, mINP

def eval_llcm(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
    """
    评估LLCM数据集的跨模态ReID性能，排除相同相机视图
    
    Args:
        同evaluate函数
    
    Returns:
        tuple: (cmc, mAP, mINP)，具体含义同evaluate
    """
    return evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, same_cam_exclude=True)

def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
    """
    评估SYSU-MM01数据集的跨模态ReID性能，特殊处理cam3和cam2的排除
    
    Args:
        同evaluate函数
    
    Returns:
        tuple: (cmc, mAP, mINP)，具体含义同evaluate
    """
    # SYSU特殊规则：仅当查询相机为cam3时，排除图库中的cam2
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc, all_AP, all_INP = [], [], []
    num_valid_q = 0

    for q_idx in range(num_q):
        q_pid, q_camid = q_pids[q_idx], q_camids[q_idx]
        order = indices[q_idx]
        keep = ~(g_camids[order] == 2) if q_camid == 3 else np.ones(num_g, dtype=bool)

        cmc = matches[q_idx][keep]
        if not np.any(cmc):
            continue

        cmc = cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])

        pos_idx = np.where(cmc == 1)[0]
        inp = cmc[pos_idx[-1]] / (pos_idx[-1] + 1.0)
        all_INP.append(inp)

        num_rel = cmc.sum()
        tmp_cmc = cmc.cumsum() / (np.arange(len(cmc)) + 1.0)
        AP = (tmp_cmc * cmc).sum() / num_rel
        all_AP.append(AP)

        num_valid_q += 1

    assert num_valid_q > 0, "Error: no valid queries found in gallery"

    cmc = np.mean(np.array(all_cmc, dtype=np.float32), axis=0)
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return cmc, mAP, mINP

def eval_regdb(distmat, q_pids, g_pids, max_rank=20):
    """
    评估RegDB数据集的跨模态ReID性能，固定两相机设置
    
    Args:
        distmat, q_pids, g_pids, max_rank同evaluate；无相机ID输入
    
    Returns:
        tuple: (cmc, mAP, mINP)，具体含义同evaluate
    """
    # RegDB固定两相机：查询为cam1，图库为cam2
    num_q, num_g = distmat.shape
    q_camids = np.ones(num_q, dtype=np.int32)  # 假设查询全为cam1
    g_camids = 2 * np.ones(num_g, dtype=np.int32)  # 图库全为cam2
    return evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, same_cam_exclude=True)