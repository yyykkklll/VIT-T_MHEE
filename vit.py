# 文件：vit.py
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=384, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation="gelu"
            ) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class VisualTransformer(nn.Module):
    def __init__(self, img_size=384, patch_size=16, in_channels=3, embed_dim=768, depth=12, num_heads=12, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.transformer = TransformerEncoder(embed_dim, depth, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)

        # 初始化权重
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (B, n_patches + 1, embed_dim)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.norm(x)
        return x[:, 1:, :]  # 移除cls_token，仅返回patch嵌入 (B, n_patches, embed_dim)

class DualViT(nn.Module):
    def __init__(self, img_size=384, patch_size=16, in_channels=3, embed_dim=768, depth=12, num_heads=12, dropout=0.1):
        super().__init__()
        self.patch_embed_vis = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.patch_embed_ir = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        depth_per_stage = depth // 4  # 每阶段3层
        self.stages = nn.ModuleList([
            TransformerEncoder(embed_dim, depth=depth_per_stage, num_heads=num_heads, dropout=dropout)
            for _ in range(4)
        ])
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed_vis.n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)  # 添加Dropout
        self.norm = nn.LayerNorm(embed_dim)  # 添加LayerNorm
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x1, x2, modal=0, stage=0):
        if modal == 0:
            x1 = self.patch_embed_vis(x1) + self.pos_embed
            x1 = self.dropout(x1)
            x2 = self.patch_embed_ir(x2) + self.pos_embed
            x2 = self.dropout(x2)
            x1 = self.stages[stage](x1)
            x1 = self.norm(x1)
            x2 = self.stages[stage](x2)
            x2 = self.norm(x2)
            return torch.cat((x1, x2), dim=0)
        elif modal == 1:
            x1 = self.patch_embed_vis(x1) + self.pos_embed
            x1 = self.dropout(x1)
            x1 = self.stages[stage](x1)
            x1 = self.norm(x1)
            return x1
        else:
            x2 = self.patch_embed_ir(x2) + self.pos_embed
            x2 = self.dropout(x2)
            x2 = self.stages[stage](x2)
            x2 = self.norm(x2)
            return x2

def vit_base_patch16_384(pretrained=False, **kwargs):
    model = VisualTransformer(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, dropout=0.1, **kwargs)
    if pretrained:
        raise NotImplementedError("Pretrained weights loading not implemented. Please use timm or HuggingFace models.")
    return model