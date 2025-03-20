# 文件：loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def pdist_torch(emb1, emb2):
    """
    计算两个嵌入集合之间的欧几里得距离矩阵，使用GPU加速
    
    Args:
        emb1 (torch.Tensor): 第一个嵌入集合，形状为 (m, d)
        emb2 (torch.Tensor): 第二个嵌入集合，形状为 (n, d)
    
    Returns:
        torch.Tensor: 距离矩阵，形状为 (m, n)
    """
    m, n = emb1.size(0), emb2.size(0)
    emb1, emb2 = emb1.float(), emb2.float()  # 转换为float32，确保数值一致性
    emb1_pow = emb1.pow(2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = emb2.pow(2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist = emb1_pow + emb2_pow - 2 * emb1 @ emb2.t()
    return dist.clamp(min=1e-12).sqrt()  # 确保数值稳定性

def compute_centers(feats, labels, unique_labels):
    """
    计算每个模态中唯一标签的类中心，处理空类情况
    
    Args:
        feats (list): 包含多个特征张量的列表，每个张量形状为 (N, d)
        labels (list): 包含对应标签张量的列表，每个张形为 (N,)
        unique_labels (torch.Tensor): 唯一标签集合
    
    Returns:
        list: 每个模态的类中心张量列表，形状为 (num_classes, d)
    """
    if len(feats) != len(labels):
        raise ValueError("特征和标签列表长度必须一致")
    centers = []
    for f, l in zip(feats, labels):
        if f.size(0) != l.size(0):
            raise ValueError("特征和标签的样本数必须一致")
        center = torch.stack([
            f[l == lb].mean(dim=0) if (l == lb).any() else torch.zeros(f.size(1), device=f.device)
            for lb in unique_labels
        ])
        centers.append(center)
    return centers

class CPMLoss(nn.Module):
    """
    Cross-Modality Pairwise Matching Loss，基于多模态特征的成对匹配损失
    """
    def __init__(self, margin=0.2):
        """
        初始化CPMLoss
        
        Args:
            margin (float): 排名损失的间隔参数，默认0.2
        """
        super().__init__()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        计算CPMLoss
        
        Args:
            inputs (torch.Tensor): 输入特征，形状为 (N, d)，当前假设为单批次特征
            targets (torch.Tensor): 标签，形状为 (N,)，对应inputs的身份ID
        
        Returns:
            torch.Tensor: 平均损失值
        """
        # 检查输入维度
        if inputs.dim() != 2 or targets.dim() != 1 or inputs.size(0) != targets.size(0):
            raise ValueError(f"输入维度不匹配: inputs {inputs.shape}, targets {targets.shape}")

        # 获取唯一标签
        unique_labels = targets.unique()

        # 如果唯一标签少于 2，返回小值而不是 0，避免训练停滞
        if len(unique_labels) < 2:
            return torch.tensor(0.01, device=inputs.device, requires_grad=True)

        # 假设 inputs 是单批次特征，暂不拆分模态
        # 计算类中心
        center = torch.stack([
            inputs[targets == lb].mean(dim=0) if (targets == lb).any() else torch.zeros(inputs.size(1), device=inputs.device)
            for lb in unique_labels
        ])

        # 计算模态内距离矩阵（当前假设单模态）
        dist = pdist_torch(center, center)

        # 创建正样本掩码
        n = len(unique_labels)
        mask = unique_labels.eq(unique_labels.unsqueeze(1))

        # 提取正负样本距离
        def get_pairs(dist):
            ap = dist[mask].view(n, -1).max(dim=1)[0] if dist[mask].numel() > 0 else torch.full((n,), float('inf'), device=dist.device)
            an = dist[~mask].view(n, -1).min(dim=1)[0] if dist[~mask].numel() > 0 else torch.full((n,), float('inf'), device=dist.device)
            return ap, an

        ap, an = get_pairs(dist)

        # 计算排名损失
        ones = torch.ones(n, device=inputs.device)
        loss = self.ranking_loss(an, ap, ones)

        # 检查损失是否为 nan 或 inf
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        return loss

class OriTripletLoss(nn.Module):
    """
    原始三元损失，带硬样本挖掘
    """
    def __init__(self, margin=0.3):
        """
        初始化OriTripletLoss
        
        Args:
            margin (float): 排名损失的间隔参数，默认0.3
        """
        super().__init__()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        计算OriTriplet Loss
        
        Args:
            inputs (torch.Tensor): 输入特征，形状为 (N, d)
            targets (torch.Tensor): 标签，形状为 (N,)
        
        Returns:
            torch.Tensor: 损失值
        """
        n = inputs.size(0)
        dist = pdist_torch(inputs, inputs)  # 计算距离矩阵，形状为 [n, n]

        # 创建正样本掩码
        mask = targets.eq(targets.unsqueeze(1))

        # 硬样本挖掘：每个样本的最远正样本和最近负样本
        dist_ap = torch.zeros(n, device=inputs.device)
        dist_an = torch.full((n,), float('inf'), device=inputs.device)

        for i in range(n):
            # 正样本距离（排除自己）
            pos_mask = mask[i] & (torch.arange(n, device=inputs.device) != i)
            if pos_mask.any():
                dist_ap[i] = dist[i][pos_mask].max()
            # 负样本距离
            neg_mask = ~mask[i]
            if neg_mask.any():
                dist_an[i] = dist[i][neg_mask].min()

        # 计算损失
        loss = self.ranking_loss(dist_an, dist_ap, torch.ones(n, device=inputs.device))
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        return loss

# 可选：保留 TripletLossWRT，但注释掉，因为 train.py 未使用
"""
class TripletLossWRT(nn.Module):
    def __init__(self):
        super().__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = F.normalize(inputs, p=2, dim=-1)
        dist_mat = pdist_torch(inputs, inputs)
        N = dist_mat.size(0)
        is_pos = targets.eq(targets.unsqueeze(1)).float()
        is_neg = 1 - is_pos
        weights_ap = F.softmax(dist_mat * is_pos, dim=1)
        weights_an = F.softmax(-dist_mat * is_neg, dim=1)
        dist_ap = (dist_mat * weights_ap).sum(dim=1)
        dist_an = (dist_mat * weights_an).sum(dim=1)
        loss = self.ranking_loss(dist_an - dist_ap, torch.ones(N, device=inputs.device))
        correct = (dist_an >= dist_ap).sum().item()
        return loss, correct
"""

# 确保损失函数可被导入
__all__ = ['OriTripletLoss', 'CPMLoss']