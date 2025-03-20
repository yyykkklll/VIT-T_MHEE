import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch import amp
from tensorboardX import SummaryWriter
from data_loader import LLCMData, SYSUData, RegDBData, TestData
from data_manager import GenIdx, process_query_llcm, process_gallery_llcm, process_query_sysu, process_gallery_sysu, process_test_regdb
from eval_metrics import eval_llcm, eval_sysu, eval_regdb
from model import embed_net
from loss import OriTripletLoss, CPMLoss
from random_erasing import RandomErasing
from utils import AverageMeter, IdentitySampler, set_seed, Logger
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import warnings

# 抑制重复警告
warnings.filterwarnings("ignore", message="Expected feature size")

def parse_args():
    """解析命令行参数并返回配置对象"""
    parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
    parser.add_argument('--dataset', default='llcm', choices=['sysu', 'regdb', 'llcm'], help='Dataset name (default: llcm)')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate (default: 0.001)')
    parser.add_argument('--optim', default='adamw', type=str, help='Optimizer (default: adamw)')
    parser.add_argument('--arch', default='vit', type=str, help='Network architecture (default: vit)')
    parser.add_argument('--resume', default='', type=str, help='Path to checkpoint file for resuming training')
    parser.add_argument('--test-only', action='store_true', help='Run test only without training')
    parser.add_argument('--model-path', default='save_model/', type=str, help='Directory to save model checkpoints')
    parser.add_argument('--save-epoch', default=5, type=int, help='Save model every N epochs (default: 5)')
    parser.add_argument('--log-path', default='log/', type=str, help='Directory to save training logs')
    parser.add_argument('--vis-log-path', default='log/vis_log/', type=str, help='Directory to save TensorBoard logs')
    parser.add_argument('--workers', default=16, type=int, help='Number of data loading workers (default: 16)')
    parser.add_argument('--img-w', default=224, type=int, help='Image width (default: 224)')
    parser.add_argument('--img-h', default=224, type=int, help='Image height (default: 224)')
    parser.add_argument('--batch-size', default=4, type=int, help='Number of identities per batch (default: 4)')
    parser.add_argument('--test-batch', default=4, type=int, help='Testing batch size (default: 4)')
    parser.add_argument('--margin', default=0.3, type=float, help='Triplet loss margin (default: 0.3)')
    parser.add_argument('--erasing-p', default=0.5, type=float, help='Random erasing probability (default: 0.5)')
    parser.add_argument('--num-pos', default=4, type=int, help='Number of positive samples per identity (default: 4)')
    parser.add_argument('--trial', default=2, type=int, help='Trial number for RegDB dataset (default: 2)')
    parser.add_argument('--seed', default=0, type=int, help='Random seed for reproducibility (default: 0)')
    parser.add_argument('--gpu', type=int, help='GPU device ID (e.g., 6), defaults to 0 if not specified')
    parser.add_argument('--mode', default='all', type=str, help='Mode for SYSU dataset (all or indoor, default: all)')
    parser.add_argument('--lambda-1', default=1.0, type=float, help='Weight for CPM loss (default: 1.0)')
    parser.add_argument('--lambda-2', default=0.00001, type=float, help='Weight for orthogonality loss (default: 0.00001)')
    parser.add_argument('--epochs', default=20, type=int, help='Number of training epochs (default: 20)')
    return parser.parse_args()

def setup_dataset(dataset, log_path):
    """根据数据集返回数据路径、日志路径、测试模式和池化维度"""
    if dataset == 'sysu':
        return '/home/s-sunxc/LLCM-main/data/SYSU-MM01/', log_path + 'sysu_log/', [1, 2], 768
    elif dataset == 'regdb':
        return '/home/s-sunxc/LLCM-main/data/RegDB/', log_path + 'regdb_log/', [2, 1], 768
    else:  # llcm
        return '/home/s-sunxc/LLCM-main/data/LLCM/', log_path + 'llcm_log/', [1, 2], 768

def setup_transforms(args):
    """设置数据预处理变换"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = {
        'sysu': transforms.Compose([
            lambda x: transforms.ToPILImage()(x) if isinstance(x, np.ndarray) else x,
            transforms.Resize((args.img_h, args.img_w)),
            transforms.RandomGrayscale(p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            RandomErasing(args.erasing_p, 0.2, 0.8, 0.3, [0.485, 0.456, 0.406])
        ]),
        'regdb': transforms.Compose([
            transforms.Resize((args.img_h, args.img_w)),
            transforms.RandomGrayscale(p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            RandomErasing(args.erasing_p, 0.02, 0.4, 0.3, [0.485, 0.456, 0.406])
        ]),
        'llcm': transforms.Compose([
            lambda x: transforms.ToPILImage()(x) if isinstance(x, np.ndarray) else x,
            transforms.Resize((args.img_h, args.img_w)),
            transforms.RandomGrayscale(p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            RandomErasing(args.erasing_p, 0.02, 0.4, 0.3, [0.485, 0.456, 0.406])
        ])
    }
    transform_test = transforms.Compose([
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalize
    ])
    return transform_train[args.dataset], transform_test

def extract_features(loader, net, mode, pool_dim, n_samples, device):
    """提取特征向量，修复特征大小不匹配问题"""
    net.eval()
    feats = [torch.zeros(n_samples, pool_dim, device=device) for _ in range(4)]
    start = time.time()
    ptr = 0
    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(loader):
            batch_num = input.size(0)
            if batch_num == 0:
                continue
            input = input.to(device)
            feat, feat_att = net(input, input, mode)
            if feat.size(0) != batch_num * 2:
                print(f"Warning: Expected {batch_num * 2} features, got {feat.size(0)} at batch {batch_idx}. Adjusting...")
                continue
            for i, feat_slice in enumerate([feat[:batch_num], feat_att[:batch_num], feat[batch_num:], feat_att[batch_num:]]):
                feats[i][ptr:ptr + batch_num] = feat_slice
            ptr += batch_num
    print(f'Extracting Time:\t {time.time() - start:.3f} s')
    return tuple(f.cpu().numpy() for f in feats)

def adjust_learning_rate(optimizer, args):
    """返回当前学习率"""
    return optimizer.param_groups[0]['lr']

def train(epoch, net, trainloader, optimizer, criterion_id, criterion_tri, criterion_cpm, scaler, args, writer, device, scheduler):
    """训练一个 epoch，调整日志输出为每 50 个批次打印一次"""
    net.train()
    losses = {'train': AverageMeter(), 'id': AverageMeter(), 'tri': AverageMeter(), 'cpm': AverageMeter(), 'ort': AverageMeter()}
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    lr = adjust_learning_rate(optimizer, args)
    for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):
        labs = torch.cat((label1, label2), 0)
        labels = labs
        input1, input2, labs, labels = input1.to(device), input2.to(device), labs.to(device), labels.to(device)
        data_time.update(time.time() - end)

        with amp.autocast('cuda'):
            feat1, out1, loss_ort = net(input1, input2)
            loss_id = criterion_id(out1, labels)
            loss_tri = criterion_tri(feat1, labels)
            ft1, ft2 = torch.chunk(feat1, 2, dim=0)
            cpm_inputs = torch.cat((ft1, ft2), dim=0)
            loss_cpm = criterion_cpm(cpm_inputs, labs) * args.lambda_1
            loss = loss_id + loss_tri + loss_cpm + loss_ort * args.lambda_2

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"Warning: NaN or Inf detected in loss at Epoch {epoch}, Batch {batch_idx}, skipping...")
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        for key, meter in losses.items():
            meter.update(loss.item() if key == 'train' else locals()[f'loss_{key}'].item(), input1.size(0) * 2)
        batch_time.update(time.time() - end)
        end = time.time()

        # 修改为每 50 个批次打印一次日志
        if batch_idx % 50 == 0 or batch_idx == len(trainloader) - 1:
            print(f'Epoch: [{epoch}][{batch_idx}/{len(trainloader)}] '
                  f'Time: {batch_time.avg:.3f}s '
                  f'Data: {data_time.avg:.3f}s '
                  f'Loss: {losses["train"].avg:.3f} '
                  f'iLoss: {losses["id"].avg:.3f} '
                  f'TLoss: {losses["tri"].avg:.3f} '
                  f'CLoss: {losses["cpm"].avg:.3f} '
                  f'OLoss: {losses["ort"].avg:.3f}')

    scheduler.step()
    for key in losses:
        writer.add_scalar(f'{key}_loss', losses[key].avg, epoch)
    writer.add_scalar('lr', lr, epoch)

def test(epoch, net, query_loader, gall_loader, query_label, gall_label, query_cam, gall_cam, test_mode, dataset, pool_dim, checkpoint_path, suffix, best_acc, device):
    """测试模型性能并保存最佳模型"""
    net.eval()
    query_feats = extract_features(query_loader, net, test_mode[1], pool_dim, len(query_label), device)
    gall_feats = extract_features(gall_loader, net, test_mode[0], pool_dim, len(gall_label), device)

    distmats = [torch.matmul(torch.tensor(q, device=device), torch.tensor(g, device=device).T) 
                for q, g in zip(query_feats, gall_feats)]
    distmat = sum(distmats).cpu().numpy()
    eval_fn = {'sysu': eval_sysu, 'regdb': eval_regdb, 'llcm': eval_llcm}[dataset]
    cmc, mAP, mINP = eval_fn(-distmat, query_label, gall_label, query_cam, gall_cam)

    if cmc[0] > best_acc:
        best_acc = cmc[0]
        state = {'net': net.state_dict(), 'cmc': cmc, 'mAP': mAP, 'mINP': mINP, 'epoch': epoch}
        torch.save(state, f'{checkpoint_path}{suffix}_best.t')
        print(f'Saved best model checkpoint at epoch {epoch + 1} with Rank-1 {cmc[0]:.2%}')

    print(f'Test Epoch: {epoch + 1}')
    print(f'Rank-1: {cmc[0]:.2%} | Rank-5: {cmc[4]:.2%} | Rank-10: {cmc[9]:.2%} | '
          f'Rank-20: {cmc[19]:.2%} | mAP: {mAP:.2%} | mINP: {mINP:.2%}')
    return best_acc, epoch + 1, mAP

def main():
    """主训练和测试流程"""
    args = parse_args()

    # 配置 GPU 设备
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please use a GPU-enabled machine.")
    device = torch.device(f'cuda:{args.gpu if args.gpu is not None else 0}')
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    print(f"Using device: {device}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(device)}")

    # 设置随机种子
    set_seed(args.seed)

    # 创建目录
    data_path, log_path, test_mode, pool_dim = setup_dataset(args.dataset, args.log_path)
    for path in [args.model_path, args.vis_log_path, log_path]:
        os.makedirs(path, exist_ok=True)

    # 配置日志和文件名
    suffix = f'{args.dataset}_p{args.num_pos}_n{args.batch_size}_lr_{args.lr}_seed_{args.seed}_{args.optim}'
    sys.stdout = Logger(os.path.join(log_path, f'{suffix}_log.txt'))
    writer = SummaryWriter(os.path.join(args.vis_log_path, suffix))
    print(f"Args: {args}")

    # 设置数据变换
    transform_train, transform_test = setup_transforms(args)

    # 加载数据集
    start_time = time.time()
    if args.dataset == 'llcm':
        trainset = LLCMData(transform=transform_train)
        query_img, query_label, query_cam = process_query_llcm(data_path, mode=test_mode[1])
        gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0])
    else:
        trainset = SYSUData(transform=transform_train) if args.dataset == 'sysu' else RegDBData(args.trial, transform=transform_train)
        query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode) if args.dataset == 'sysu' else process_test_regdb(data_path, args.trial, 'visible')
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode) if args.dataset == 'sysu' else process_test_regdb(data_path, args.trial, 'thermal')

    # 数据统计
    n_classes = len(np.unique(trainset.color_labels))
    print(f'Dataset {args.dataset} Statistics:')
    print(f'  Visible: {n_classes} classes | {len(trainset.color_labels)} samples')
    print(f'  Thermal: {n_classes} classes | {len(trainset.thermal_labels)} samples')
    print(f'  Query: {len(np.unique(query_label))} IDs | {len(query_label)} samples')
    print(f'  Gallery: {len(np.unique(gall_label))} IDs | {len(gall_label)} samples')
    print(f'Data Loading Time: {time.time() - start_time:.3f} s')

    # 构建模型
    net = embed_net(n_classes, args.dataset, embed_dim=768, depth=6, num_heads=6,
                    img_h=args.img_h, img_w=args.img_w).to(device)
    if args.resume and os.path.isfile(os.path.join(args.model_path, args.resume)):
        checkpoint = torch.load(os.path.join(args.model_path, args.resume))
        net.load_state_dict(checkpoint['net'])
        print(f'Loaded checkpoint {args.resume} (epoch {checkpoint["epoch"]})')

    # 定义损失函数
    criterion_id = nn.CrossEntropyLoss().to(device)
    criterion_tri = OriTripletLoss(args.batch_size * args.num_pos).to(device)
    criterion_cpm = CPMLoss(margin=0.2).to(device)

    # 构建优化器
    try:
        bottleneck_params = set(id(p) for p in net.bottleneck.parameters())
        classifier_params = set(id(p) for p in net.classifier.parameters())
        base_params = [p for p in net.parameters() if id(p) not in bottleneck_params and id(p) not in classifier_params]
        optimizer = optim.AdamW([
            {'params': base_params, 'lr': 0.1 * args.lr},
            {'params': net.bottleneck.parameters(), 'lr': args.lr},
            {'params': net.classifier.parameters(), 'lr': args.lr}
        ], weight_decay=5e-4)
    except AttributeError as e:
        print(f"Warning: net.bottleneck or net.classifier not found in model. Using uniform learning rate.")
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=5e-4)

    scaler = amp.GradScaler()
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 生成身份索引
    color_pos, thermal_pos = GenIdx(trainset.color_labels, trainset.thermal_labels)
    print(f'Generated color_pos with {len(color_pos)} entries, thermal_pos with {len(thermal_pos)} entries')
    for label, indices in list(color_pos.items())[:5]:
        print(f'Color label {label}: {len(indices)} indices')
    for label, indices in list(thermal_pos.items())[:5]:
        print(f'Thermal label {label}: {len(indices)} indices')

    # 训练和测试循环
    print('Starting Training...')
    best_acc = 0
    best_mAP = 0
    early_stop_counter = 0
    early_stop_patience = 5

    try:
        for epoch in range(args.epochs):
            print(f'Preparing Data Loader for Epoch {epoch + 1}...')
            sampler = IdentitySampler(trainset.color_labels, trainset.thermal_labels, color_pos, thermal_pos, args.num_pos, args.batch_size, args.epochs)
            trainloader = data.DataLoader(trainset, batch_size=args.batch_size * args.num_pos * 2,
                                        sampler=sampler, num_workers=args.workers, drop_last=True, pin_memory=True)
            gallset, queryset = TestData(gall_img, gall_label, transform=transform_test), TestData(query_img, query_label, transform=transform_test)
            gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, drop_last=False, pin_memory=True)
            query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, drop_last=False, pin_memory=True)

            train(epoch, net, trainloader, optimizer, criterion_id, criterion_tri, criterion_cpm, scaler, args, writer, device, scheduler)
            if (epoch + 1) % args.save_epoch == 0:
                state = {'net': net.state_dict(), 'epoch': epoch}
                torch.save(state, f'{args.model_path}{suffix}_epoch_{epoch + 1}.t')
                print(f'Saved checkpoint at epoch {epoch + 1}')

            if (epoch + 1) % 5 == 0:
                best_acc, _, mAP = test(epoch, net, query_loader, gall_loader, query_label, gall_label, query_cam, gall_cam, test_mode, args.dataset, pool_dim, args.model_path, suffix, best_acc, device)
                if mAP > best_mAP:
                    best_mAP = mAP
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= early_stop_patience:
                        print(f'Early stopping at epoch {epoch + 1} due to no improvement in mAP.')
                        break

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected, saving current model...")
        state = {'net': net.state_dict(), 'epoch': epoch}
        torch.save(state, f'{args.model_path}{suffix}_interrupted_epoch_{epoch + 1}.t')
        print(f'Saved interrupted checkpoint at epoch {epoch + 1}')
        writer.close()
        sys.exit(0)

    print(f'Training finished. Best Rank-1: {best_acc:.2%} at epoch {_}')
    writer.close()

class IdentitySampler(data.Sampler):
    def __init__(self, color_labels, thermal_labels, color_pos, thermal_pos, num_pos, batch_size, epochs):
        self.color_labels = color_labels
        self.thermal_labels = thermal_labels
        self.color_pos = color_pos
        self.thermal_pos = thermal_pos
        self.num_pos = num_pos
        self.batch_size = batch_size
        self.n_classes = len(set(color_labels))
        self.num_batches = min(len(color_labels), len(thermal_labels)) // (batch_size * num_pos * 2)

    def __iter__(self):
        indices = []
        sampled_pids = set()
        for _ in range(self.num_batches):
            available_pids = [pid for pid in range(self.n_classes) if pid not in sampled_pids]
            if len(available_pids) < self.batch_size:
                sampled_pids.clear()
                available_pids = list(range(self.n_classes))
            pids = np.random.choice(available_pids, self.batch_size, replace=False)
            sampled_pids.update(pids)
            
            for pid in pids:
                color_idx = np.random.choice(self.color_pos[pid], self.num_pos, replace=len(self.color_pos[pid]) < self.num_pos)
                thermal_idx = np.random.choice(self.thermal_pos[pid], self.num_pos, replace=len(self.thermal_pos[pid]) < self.num_pos)
                indices.extend(color_idx)
                indices.extend(thermal_idx)
        return iter(indices)

    def __len__(self):
        return self.num_batches * self.batch_size * self.num_pos * 2

if __name__ == '__main__':
    main()