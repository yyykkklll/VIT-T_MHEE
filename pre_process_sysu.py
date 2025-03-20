import numpy as np  # 导入数值计算库
from PIL import Image  # 导入图像处理库
import pdb  # 导入Python调试器
import os  # 导入操作系统接口模块
import h5py  # 导入HDF5文件格式处理库

# 定义数据集路径
data_path = 'data/SYSU-MM01'

# 定义RGB相机和红外相机ID
rgb_cameras = ['cam1','cam2','cam4','cam5']  # RGB相机编号
ir_cameras = ['cam3','cam6']  # 红外相机编号

# 加载训练集ID信息
file_path_train = os.path.join(data_path,'exp/train_id.txt')
file_path_val   = os.path.join(data_path,'exp/val_id.txt')
with open(file_path_train, 'r') as file:
    ids = file.read().splitlines()  # 读取所有行
    ids = [int(y) for y in ids[0].split(',')]  # 将字符串转换为整数列表
    id_train = ["%04d" % x for x in ids]  # 格式化ID为4位数字符串
    
# 加载验证集ID信息
with open(file_path_val, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_val = ["%04d" % x for x in ids]
    
# 合并训练集和验证集ID   
id_train.extend(id_val) 

# 初始化RGB和红外图像文件路径列表
files_rgb = []
files_ir = []

# 遍历所有ID，收集对应的图像文件路径
for id in sorted(id_train):
    # 收集RGB图像
    for cam in rgb_cameras:
        img_dir = os.path.join(data_path,cam,id)
        if os.path.isdir(img_dir):  # 检查目录是否存在
            new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])  # 获取目录下所有文件
            files_rgb.extend(new_files)  # 添加到RGB文件列表
            
    # 收集红外图像
    for cam in ir_cameras:
        img_dir = os.path.join(data_path,cam,id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
            files_ir.extend(new_files)  # 添加到红外文件列表

# 重新标记ID：将原始ID映射到连续的标签
pid_container = set()
for img_path in files_ir:
    pid = int(img_path[-13:-9])  # 从文件路径提取人物ID
    pid_container.add(pid)  # 添加到集合中去重
pid2label = {pid:label for label, pid in enumerate(pid_container)}  # 创建ID到标签的映射字典

# 定义图像调整大小的目标尺寸
fix_image_width = 144
fix_image_height = 384

def process_images_in_batches(image_files, output_img_path, output_label_path, batch_size=100):
    """
    分批处理图像以避免内存错误
    
    参数:
    image_files: 图像文件路径列表
    output_img_path: 输出图像数据的HDF5文件路径
    output_label_path: 输出标签数据的HDF5文件路径
    batch_size: 每批处理的图像数量
    """
    total_images = len(image_files)
    print(f"总共需要处理 {total_images} 张图像")
    
    # 创建HDF5文件来存储数据
    with h5py.File(output_img_path, 'w') as img_file, h5py.File(output_label_path, 'w') as label_file:
        # 创建可扩展的数据集
        img_dataset = img_file.create_dataset(
            'images', 
            shape=(total_images, fix_image_height, fix_image_width, 3),  # 设置数据集形状
            dtype=np.uint8,  # 设置数据类型为无符号8位整数
            chunks=(1, fix_image_height, fix_image_width, 3)  # 每个图像单独作为一个块，优化读取性能
        )
        
        label_dataset = label_file.create_dataset(
            'labels', 
            shape=(total_images,),  # 标签是一维数组
            dtype=np.int32  # 标签使用32位整数
        )
        
        # 分批处理图像
        for i in range(0, total_images, batch_size):
            end_idx = min(i + batch_size, total_images)  # 计算当前批次的结束索引
            batch_files = image_files[i:end_idx]  # 获取当前批次的文件路径
            print(f"处理批次 {i//batch_size + 1}/{(total_images-1)//batch_size + 1}，图像 {i+1} 到 {end_idx}")
            
            batch_imgs = []  # 存储当前批次的图像数据
            batch_labels = []  # 存储当前批次的标签数据
            
            for img_path in batch_files:
                # 处理图像：打开、调整大小并转换为数组
                img = Image.open(img_path)
                img = img.resize((fix_image_width, fix_image_height), Image.Resampling.LANCZOS)  # 使用LANCZOS算法进行高质量缩放
                pix_array = np.array(img)  # 转换为NumPy数组
                
                # 处理标签：从文件路径提取ID并映射到标签
                pid = int(img_path[-13:-9])
                pid = pid2label[pid]  # 使用映射字典转换为连续标签
                
                batch_imgs.append(pix_array)
                batch_labels.append(pid)
            
            # 保存批次数据到HDF5文件
            img_dataset[i:end_idx] = np.array(batch_imgs)
            label_dataset[i:end_idx] = np.array(batch_labels)
            
            # 清空批次数据释放内存
            batch_imgs = []
            batch_labels = []
    
    print(f"已完成处理并保存到 {output_img_path} 和 {output_label_path}")

# 处理RGB图像并保存
rgb_img_path = os.path.join(data_path, 'train_rgb_resized_img.h5')
rgb_label_path = os.path.join(data_path, 'train_rgb_resized_label.h5')
process_images_in_batches(files_rgb, rgb_img_path, rgb_label_path)

# 处理红外图像并保存
ir_img_path = os.path.join(data_path, 'train_ir_resized_img.h5')
ir_label_path = os.path.join(data_path, 'train_ir_resized_label.h5')
process_images_in_batches(files_ir, ir_img_path, ir_label_path)

print("所有数据处理完成！")
