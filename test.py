# 文件：test.py
import argparse
import time
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import os

from data_loader import SYSUData, RegDBData, LLCMData, TestData
from data_manager import process_query_sysu, process_gallery_sysu, process_query_llcm, process_gallery_llcm, process_test_regdb
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from model import embed_net

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Testing')
    parser.add_argument('--dataset', default='sysu', choices=['sysu', 'regdb', 'llcm'], help='dataset name: sysu, regdb, or llcm')
    parser.add_argument('--arch', default='vit', type=str, help='network baseline: vit')  # 更新为ViT
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint path')
    parser.add_argument('--model_path', default='save_model/', type=str, help='model save path')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers')
    parser.add_argument('--img_w', default=384, type=int, help='image width')  # 调整为384
    parser.add_argument('--img_h', default=384, type=int, help='image height (384 for ViT)')  # 调整为384
    parser.add_argument('--test_batch', default=32, type=int, help='testing batch size')
    parser.add_argument('--trial', default=1, type=int, help='trial number for RegDB dataset')
    parser.add_argument('--gpu', default='1', type=str, help='GPU device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--mode', default='all', type=str, help='all or indoor for SYSU dataset')
    parser.add_argument('--tvsearch', action='store_true', default=True, help='thermal to visible search for RegDB')
    return parser.parse_args()

def setup_dataset(args):
    """
    根据数据集设置参数
    """
    if args.dataset == 'sysu':
        data_path = '/home/s-sunxc/LLCM-main/data/SYSU-MM01/'
        n_class, test_mode, pool_dim = 395, [1, 2], 768  # 调整pool_dim为768
        args.img_h = 384
    elif args.dataset == 'regdb':
        data_path = '/home/s-sunxc/LLCM-main/RegDB/'
        n_class, test_mode, pool_dim = 206, [2, 1], 768  # 调整pool_dim为768
    else:  # llcm
        data_path = '/home/s-sunxc/LLCM-main/data/LLCM/'
        n_class, test_mode, pool_dim = 713, [2, 1], 768  # 调整pool_dim为768
        args.img_h = 384
    return data_path, n_class, test_mode, pool_dim

def fliplr(img):
    """
    水平翻转图像
    """
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
    return img.index_select(3, inv_idx)

def extract_features(loader, net, device, mode, pool_dim, n_samples):
    """
    提取特征
    """
    net.eval()
    start = time.time()
    ptr = 0
    feats = [np.zeros((n_samples, pool_dim)) for _ in range(6)]
    with torch.no_grad():
        for input, _ in loader:
            batch_num = input.size(0)
            input1 = input.to(device)
            input2 = fliplr(input).to(device)
            feat_pool1, feat_fc1 = net(input1, input1, mode)
            feat_pool2, feat_fc2 = net(input2, input2, mode)
            feat, feat_att = feat_pool1 + feat_pool2, feat_fc1 + feat_fc2
            for i, feat_slice in enumerate([feat[:batch_num], feat_att[:batch_num],
                                          feat[batch_num:batch_num*2], feat_att[batch_num:batch_num*2],
                                          feat[batch_num*2:], feat_att[batch_num*2:]]):
                feats[i][ptr:ptr + batch_num] = feat_slice.detach().cpu().numpy()
            ptr += batch_num
    print(f'Extracting Time:\t {time.time() - start:.3f} s')
    return tuple(feats)

def compute_dist_and_metrics(query_feats, gall_feats, query_labels, gall_labels, query_cams, gall_cams, test_mode, eval_fn):
    """
    计算距离矩阵并评估指标
    """
    a = 0.1
    distmats = [np.matmul(qf, gf.T) for qf, gf in zip(query_feats, gall_feats)]
    distmat7 = sum(distmats)
    distmat8 = a * sum(distmats[:3]) + (1 - a) * sum(distmats[3:])
    
    if test_mode == [1, 2]:
        cmc7, mAP7, mINP7 = eval_fn(-distmat7, query_labels, gall_labels, query_cams, gall_cams)
        cmc8, mAP8, mINP8 = eval_fn(-distmat8, query_labels, gall_labels, query_cams, gall_cams)
    else:
        cmc7, mAP7, mINP7 = eval_fn(-distmat7, gall_labels, query_labels, gall_cams, query_cams)
        cmc8, mAP8, mINP8 = eval_fn(-distmat8, gall_labels, query_labels, gall_cams, query_cams)
    return (cmc7, mAP7, mINP7), (cmc8, mAP8, mINP8)

def print_metrics(cmc, mAP, mINP, prefix='POOL'):
    """
    打印评估指标
    """
    print(f'{prefix}: Rank-1: {cmc[0]:.2%} | Rank-5: {cmc[4]:.2%} | Rank-10: {cmc[9]:.2%} | '
          f'Rank-20: {cmc[19]:.2%} | mAP: {mAP:.2%} | mINP: {mINP:.2%}')

def main():
    """
    主函数，执行跨模态测试
    """
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 设置数据集参数
    data_path, n_class, test_mode, pool_dim = setup_dataset(args)

    # 初始化模型
    print('==> Building model..')
    net = embed_net(n_class, args.dataset, embed_dim=768, depth=12, num_heads=12).to(device)  # 适配ViT参数
    torch.cuda.set_device(0)  # 确保使用第一个GPU

    # 数据预处理
    transform_test = transforms.Compose([
        transforms.Resize((args.img_h, args.img_w)),  # 调整为384×384
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载检查点
    def load_checkpoint(model_path):
        if os.path.isfile(model_path):
            print(f'==> Loading checkpoint {model_path}')
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])
            print(f'==> Loaded checkpoint (epoch {checkpoint["epoch"]})')
        else:
            print(f'==> No checkpoint found at {model_path}')
            return False
        return True

    # 测试流程
    end = time.time()
    all_cmc7, all_mAP7, all_mINP7 = None, 0, 0
    all_cmc8, all_mAP8, all_mINP8 = None, 0, 0

    if args.dataset in ['sysu', 'llcm']:
        # 加载数据
        process_query = process_query_sysu if args.dataset == 'sysu' else process_query_llcm
        process_gallery = process_gallery_sysu if args.dataset == 'sysu' else process_gallery_llcm
        eval_fn = eval_sysu if args.dataset == 'sysu' else eval_llcm

        query_img, query_label, query_cam = process_query(data_path, mode=args.mode if args.dataset == 'sysu' else test_mode[1])
        gall_img, gall_label, gall_cam = process_gallery(data_path, mode=args.mode if args.dataset == 'sysu' else test_mode[0], trial=0)

        nquery, ngall = len(query_label), len(gall_label)
        print("Dataset statistics:")
        print("  ------------------------------")
        print(f"  query    | {len(np.unique(query_label)):5d} | {nquery:8d}")
        print(f"  gallery  | {len(np.unique(gall_label)):5d} | {ngall:8d}")
        print("  ------------------------------")

        # 数据加载器
        query_loader = data.DataLoader(
            TestData(query_img, query_label, transform=transform_test),
            batch_size=args.test_batch, shuffle=False, num_workers=args.workers
        )
        print(f'Data Loading Time:\t {time.time() - end:.3f} s')

        # 加载检查点
        model_path = args.resume if args.resume.startswith('/') or args.resume.startswith('save_model/') else os.path.join(args.model_path, args.resume)
        load_checkpoint(model_path)

        # 提取特征
        query_feats = extract_features(query_loader, net, device, test_mode[1], pool_dim, nquery)

        # 多次试验
        for trial in range(10):
            gall_img, gall_label, gall_cam = process_gallery(data_path, mode=args.mode if args.dataset == 'sysu' else test_mode[0], trial=trial)
            gall_loader = data.DataLoader(
                TestData(gall_img, gall_label, transform=transform_test),
                batch_size=args.test_batch, shuffle=False, num_workers=args.workers
            )
            gall_feats = extract_features(gall_loader, net, device, test_mode[0], pool_dim, len(gall_label))

            # 计算距离矩阵并评估
            (cmc7, mAP7, mINP7), (cmc8, mAP8, mINP8) = compute_dist_and_metrics(
                query_feats, gall_feats, query_label, gall_label, query_cam, gall_cam, test_mode, eval_fn
            )

            # 累加结果
            if trial == 0:
                all_cmc7, all_mAP7, all_mINP7 = cmc7, mAP7, mINP7
                all_cmc8, all_mAP8, all_mINP8 = cmc8, mAP8, mINP8
            else:
                all_cmc7 += cmc7
                all_mAP7 += mAP7
                all_mINP7 += mINP7
                all_cmc8 += cmc8
                all_mAP8 += mAP8
                all_mINP8 += mINP8

            print(f'Test Trial: {trial}')
            print_metrics(cmc7, mAP7, mINP7)
            print_metrics(cmc8, mAP8, mINP8)

    else:  # regdb
        eval_fn = eval_regdb
        for trial in range(10):
            test_trial = trial + 1
            model_path = args.resume if args.resume else os.path.join(args.model_path, f'regdb_agw_p4_n6_lr_0.1_seed_0_trial_{test_trial}_best.t')
            model_path = model_path if model_path.startswith('/') or model_path.startswith('save_model/') else os.path.join(args.model_path, model_path)
            if not load_checkpoint(model_path):
                continue

            # 加载数据
            query_img, query_label = process_test_regdb(data_path, trial=test_trial, modal='thermal')
            gall_img, gall_label = process_test_regdb(data_path, trial=test_trial, modal='visible')
            nquery, ngall = len(query_label), len(gall_label)

            query_loader = data.DataLoader(
                TestData(query_img, query_label, transform=transform_test),
                batch_size=args.test_batch, shuffle=False, num_workers=args.workers
            )
            gall_loader = data.DataLoader(
                TestData(gall_img, gall_label, transform=transform_test),
                batch_size=args.test_batch, shuffle=False, num_workers=args.workers
            )
            print(f'Data Loading Time:\t {time.time() - end:.3f} s')

            # 提取特征
            query_feats = extract_features(query_loader, net, device, test_mode[1], pool_dim, nquery)
            gall_feats = extract_features(gall_loader, net, device, test_mode[0], pool_dim, ngall)

            # 计算距离矩阵并评估
            (cmc7, mAP7, mINP7), (cmc8, mAP8, mINP8) = compute_dist_and_metrics(
                query_feats, gall_feats, query_label, gall_label, None, None, test_mode, eval_fn
            )

            # 累加结果
            if trial == 0:
                all_cmc7, all_mAP7, all_mINP7 = cmc7, mAP7, mINP7
                all_cmc8, all_mAP8, all_mINP8 = cmc8, mAP8, mINP8
            else:
                all_cmc7 += cmc7
                all_mAP7 += mAP7
                all_mINP7 += mINP7
                all_cmc8 += cmc8
                all_mAP8 += mAP8
                all_mINP8 += mINP8

            print(f'Test Trial: {trial}')
            print_metrics(cmc7, mAP7, mINP7)
            print_metrics(cmc8, mAP8, mINP8)

    # 计算平均结果
    cmc7, mAP7, mINP7 = all_cmc7 / 10, all_mAP7 / 10, all_mINP7 / 10
    cmc8, mAP8, mINP8 = all_cmc8 / 10, all_mAP8 / 10, all_mINP8 / 10
    print('All Average:')
    print_metrics(cmc7, mAP7, mINP7)
    print_metrics(cmc8, mAP8, mINP8)

if __name__ == '__main__':
    main()