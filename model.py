import torch
import torch.nn as nn
import torch.nn.functional as F
from vit import DualViT, TransformerEncoder

def weights_init_kaiming(m):
    """
    使用Kaiming初始化卷积层、线性层和BatchNorm1d的权重
    """
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif 'Linear' in classname:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        nn.init.zeros_(m.bias.data)
    elif 'BatchNorm1d' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.01)
        nn.init.zeros_(m.bias.data)

def weights_init_classifier(m):
    """
    初始化分类器线性层的权重和偏置
    """
    if 'Linear' in m.__class__.__name__:
        nn.init.normal_(m.weight.data, 0, 0.001)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)

class Normalize(nn.Module):
    """
    对特征进行L2归一化
    """
    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        return x.div(norm)

class BaseViT(nn.Module):
    """
    基础ViT模型,提取特征层
    """
    def __init__(self, embed_dim=768, depth=12, num_heads=12, img_h=384, img_w=384):
        super().__init__()
        self.transformer = TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads)
        self.n_patches = (img_h // 16) ** 2

    def forward(self, x):
        x = self.transformer(x)  # (B, n_patches, embed_dim)
        return x

# 移除原DEEModule类
# class DEEModule(nn.Module):
#     def __init__(self, embed_dim, reduction=16):
#         ...
#     def forward(self, x):
#         ...

# 添加新的T_MHEE类
class T_MHEE(nn.Module):
    """
    Transformer-based Multi-Head Embedding Expansion
    """
    def __init__(self, embed_dim=768, num_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape  # (2B, 576, 768)
        # 多头注意力
        attn_output, _ = self.mha(x, x, x)  # (2B, 576, 768)
        attn_output = self.norm(attn_output + x)  # 残差连接
        attn_output = attn_output * self.scale  # 缩放
        return self.dropout(attn_output)

    def orth_loss(self, x):
        # 计算正交损失
        x_norm = F.normalize(x, p=2, dim=2)
        ortho_mat = torch.bmm(x_norm, x_norm.transpose(1, 2))
        ortho_mat.diagonal(dim1=1, dim2=2).zero_()
        return torch.clamp(ortho_mat.abs().sum() / (x.size(1) * (x.size(1) - 1)), min=0.0)

class CNL(nn.Module):
    """
    Cross-Modal Non-local模块,适配ViT的序列化特征
    """
    def __init__(self, high_dim, low_dim, flag=0):
        super().__init__()
        self.g = nn.Conv1d(high_dim, low_dim, kernel_size=1)
        self.theta = nn.Conv1d(high_dim, low_dim, kernel_size=1)
        self.phi = nn.Conv1d(high_dim, low_dim, kernel_size=1)
        self.W = nn.Sequential(
            nn.Conv1d(low_dim, high_dim, kernel_size=1),
            nn.BatchNorm1d(high_dim)
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        x_h = x_h.transpose(1, 2)  # (B, embed_dim, n_patches)
        x_l = x_l.transpose(1, 2)  # (B, embed_dim, n_patches)
        g_x = self.g(x_l)  # (B, low_dim, n_patches)
        theta_x = self.theta(x_h)  # (B, low_dim, n_patches)
        phi_x = self.phi(x_l)  # (B, low_dim, n_patches)
        energy = torch.bmm(theta_x.transpose(1, 2), phi_x)  # (B, n_patches, n_patches)
        attention = energy / energy.size(-1)
        y = torch.bmm(attention, g_x.transpose(1, 2))  # (B, n_patches, low_dim)
        y = y.transpose(1, 2)  # (B, low_dim, n_patches)
        y = self.W(y)  # (B, high_dim, n_patches)
        return (y + x_h).transpose(1, 2)  # (B, n_patches, high_dim)

class PNL(nn.Module):
    """
    Pyramid Non-local模块,适配ViT的序列化特征
    """
    def __init__(self, high_dim, low_dim, reduc_ratio=2):
        super().__init__()
        self.g = nn.Conv1d(high_dim, low_dim // reduc_ratio, kernel_size=1)
        self.theta = nn.Conv1d(high_dim, low_dim // reduc_ratio, kernel_size=1)
        self.phi = nn.Conv1d(high_dim, low_dim // reduc_ratio, kernel_size=1)
        self.W = nn.Sequential(
            nn.Conv1d(low_dim // reduc_ratio, high_dim, kernel_size=1),
            nn.BatchNorm1d(high_dim)
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        x_h = x_h.transpose(1, 2)  # (B, embed_dim, n_patches)
        x_l = x_l.transpose(1, 2)  # (B, embed_dim, n_patches)
        g_x = self.g(x_l).transpose(1, 2)  # (B, n_patches, low_dim // reduc_ratio)
        theta_x = self.theta(x_h).transpose(1, 2)  # (B, n_patches, low_dim // reduc_ratio)
        phi_x = self.phi(x_l)  # (B, low_dim // reduc_ratio, n_patches)
        energy = torch.bmm(theta_x, phi_x)  # (B, n_patches, n_patches)
        attention = energy / energy.size(-1)
        y = torch.bmm(attention, g_x).transpose(1, 2)  # (B, low_dim // reduc_ratio, n_patches)
        y = self.W(y)  # (B, high_dim, n_patches)
        return (y + x_h).transpose(1, 2)  # (B, n_patches, high_dim)

class MFA_Block(nn.Module):
    """
    Multi-Feature Alignment块,适配ViT
    """
    def __init__(self, high_dim, low_dim, flag):
        super().__init__()
        self.CNL = CNL(high_dim, low_dim, flag)
        self.PNL = PNL(high_dim, low_dim)

    def forward(self, x, x0):
        z = self.CNL(x, x0)
        return self.PNL(z, x0)

class EmbedNet(nn.Module):
    def __init__(self, n_class, dataset, embed_dim=768, depth=12, num_heads=12, img_h=384, img_w=384):
        super(EmbedNet, self).__init__()
        self.dataset = dataset
        assert img_h == img_w, "EmbedNet assumes square input images (img_h == img_w)"
        img_size = img_h
        self.n_patches = (img_size // 16) ** 2

        # 初始化 backbone (DualViT)
        self.backbone = DualViT(
            img_size=img_size,
            patch_size=16,
            in_channels=3,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            dropout=0.1
        )

        # 初始化 MFA 和 T_MHEE 模块
        self.MFA0 = MFA_Block(high_dim=embed_dim, low_dim=embed_dim // 2, flag=0)
        self.MFA1 = MFA_Block(high_dim=embed_dim, low_dim=embed_dim // 2, flag=0)
        self.MFA2 = MFA_Block(high_dim=embed_dim, low_dim=embed_dim // 2, flag=1)
        self.MFA3 = MFA_Block(high_dim=embed_dim, low_dim=embed_dim // 2, flag=2)
        self.T_MHEE = T_MHEE(embed_dim=embed_dim, num_heads=4)  # 替换DEE为T_MHEE

        # Stage-4 的额外Transformer层
        self.stage4_transformer = TransformerEncoder(embed_dim=embed_dim, depth=1, num_heads=num_heads)

        # 池化层
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Bottleneck 和分类器
        self.bottleneck = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )
        self.bottleneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(embed_dim, n_class)
        self.classifier.apply(weights_init_classifier)

        # L2 归一化
        self.l2norm = Normalize(2)

    def forward(self, x1, x2, modal=0):
        # Stage-0
        x = self.backbone(x1, x2, modal, stage=0)
        x_ = x
        x0 = x  # 保存初始特征作为低层次特征
        x_ = self.MFA0(x, x0)

        # Stage-1
        x = self.backbone(x1, x2, modal, stage=1)
        x_ = self.MFA1(x, x0)

        # Stage-2
        x = self.backbone(x1, x2, modal, stage=2)
        x_ = self.MFA2(x, x0)

        # Stage-3
        x = self.backbone(x1, x2, modal, stage=3)
        x_ = self.MFA3(x, x0)
        x_ = self.T_MHEE(x_)  # 替换为T_MHEE

        # Stage-4
        x_ = self.stage4_transformer(x_)
        x_ = F.normalize(x_, p=2, dim=2)
        xp = self.pool(x_.transpose(1, 2)).squeeze(-1)
        x_pool = xp
        feat = self.bottleneck(x_pool)

        if self.training:
            xps = x_.permute(0, 2, 1)
            batch_size = x_.size(0)
            n_patches = x_.size(1)
            if batch_size % 2 == 0:
                xp1, xp2 = torch.chunk(xps, 2, dim=0)
                xpss = torch.cat((xp2, xp1), dim=0)
                xpss_norm = F.normalize(xpss, p=2, dim=1)
                ortho_mat = torch.bmm(xpss_norm, xpss_norm.permute(0, 2, 1))
                ortho_mat.diagonal(dim1=1, dim2=2).zero_()
                # 更新正交损失，调用T_MHEE的orth_loss
                loss_ort = self.T_MHEE.orth_loss(x_)
            else:
                raise ValueError("Batch size must be even for current loss computation.")
            return x_pool, self.classifier(feat), loss_ort
        return self.l2norm(x_pool), self.l2norm(feat)

# 确保 embed_net 可被导入
embed_net = EmbedNet