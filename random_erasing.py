import random
import math
import torch

class RandomErasing:
    """
    随机擦除数据增强类，基于Zhong et al.的"Random Erasing Data Augmentation"
    在图像中随机选择一个矩形区域并将其像素值替换为指定均值，增强模型对遮挡的鲁棒性
    
    Args:
        probability (float): 执行随机擦除的概率，默认0.5
        sl (float): 擦除区域面积占图像面积的最小比例，默认0.2
        sh (float): 擦除区域面积占图像面积的最大比例，默认0.8
        r1 (float): 擦除区域的最小宽高比，默认0.3
        mean (list): 擦除区域填充的像素均值，默认ImageNet均值[0.4914, 0.4822, 0.4465]
    """
    def __init__(self, probability=0.5, sl=0.2, sh=0.8, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean = torch.tensor(mean).view(-1, 1, 1)  # 转换为张量并调整形状为(C, 1, 1)

    def __call__(self, img):
        """
        对输入图像执行随机擦除操作
        
        Args:
            img (torch.Tensor): 输入图像，形状为(C, H, W)
        
        Returns:
            torch.Tensor: 擦除后的图像
        """
        if random.random() > self.probability:
            return img

        img = img.clone()  # 避免修改原始图像
        C, H, W = img.shape
        area = H * W

        # 最多尝试100次以找到合适的擦除区域
        for _ in range(100):
            # 计算目标擦除面积和宽高比
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            # 计算擦除区域的高度和宽度
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            # 确保擦除区域在图像范围内
            if w < W and h < H:
                x1 = random.randint(0, H - h)
                y1 = random.randint(0, W - w)
                
                # 确保mean的通道数与图像一致
                mean = self.mean.to(device=img.device, dtype=img.dtype)
                if C == 3:
                    img[:, x1:x1+h, y1:y1+w] = mean
                else:  # 灰度图像，C=1
                    img[:, x1:x1+h, y1:y1+w] = mean[0:1]  # 只取第一个通道值
                return img

        return img  # 如果尝试100次仍未找到合适区域，返回原图