import json
import numpy as np
import time
from torch.amp import GradScaler
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from data_potential import CloudsDataset, ActiveLearningSampler
from model import RandLANet
from datetime import datetime
import os
from utils.metrics import compute_metrics


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def evaluate(model, loader, criterion, device, num_classes, cfg):
    losses = []
    accuracies = []
    ious = []

    model.eval()

    with torch.no_grad():
        for batch_idx, points in enumerate(loader):
            points = {k: v.to(device, non_blocking=True) for k, v in points.items()}
            feature = points['xyz']
            labels = points['class'].long()
            if cfg["HasRGB"]:
                rgb = points['rgb']
                feature = torch.cat((feature, rgb), dim=2)
            if cfg["HasIntensity"]:
                intensity = points['intensity'].unsqueeze(-1)
                feature = torch.cat((feature, intensity), dim=2)

            scores = model(feature).to(device)
            loss = criterion(scores, labels).to(device)

            acc, miou = compute_metrics(scores, labels, num_classes)
            losses.append(loss.cpu().item())
            accuracies.append(acc)
            ious.append(miou)
    return np.mean(losses), np.nanmean(np.array(accuracies), axis=0), np.nanmean(np.array(ious), axis=0)

def main(cfg):
    now = datetime.now()
    logs_dir = cfg.train.loggers
    logs_dir = os.path.join(logs_dir, f"{now.year}-{now.month}-{now.day}")
    os.makedirs(logs_dir, exist_ok=True)
    checkpoint_dir = os.path.join(cfg.train.checkpoints, f"{now.year}-{now.month}-{now.day}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    path = os.path.join(cfg.data.root, f'processed-{cfg.data.grid_size:.2f}', 'metadata.json')
    with open(path, 'rb') as f:
        data_raw = json.load(f)
    data = torch.tensor(data_raw['class_weights'])

    device = torch.device('cuda')
    torch.cuda.set_device(int(cfg.train.gpu))

    dataset_train = CloudsDataset(cfg, split='train', transform=True)
    train_sampler = ActiveLearningSampler(
        config=cfg,
        dataset=dataset_train,
        batch_size=cfg.train.batch_size,
        step_size=cfg.train.train_steps,
        split='train',
        transform=True
    )
    train_loader = DataLoader(
        train_sampler,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True
    )

    dataset_val = CloudsDataset(cfg, split='val')
    val_sampler = ActiveLearningSampler(
        config=cfg,
        dataset=dataset_val,
        batch_size=cfg.train.batch_size,
        step_size=cfg.train.val_steps,
        split='val'
    )
    val_loader = DataLoader(
        val_sampler,
        batch_size=9,  # cfg.train.batch_size
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True
    )

    num_classes = cfg.data.num_classes
    d_in = 3
    if cfg.data.HasRGB:
        d_in += 3
    if cfg.data.HasIntensity:
        d_in += 1

    model = RandLANet(
        d_in,
        num_classes,
        num_neighbors=cfg.network.num_neighbors,
        decimation=cfg.network.decimation,
        device=device
    ).to(device)

    print(f"模型设备: {next(model.parameters()).device}")  # 输出应为 cuda:x

    print('Computing weights...', end='\t')

    frequency = data / torch.sum(data)
    weights = 1.0 / torch.sqrt(frequency)
    weights = weights.to(torch.float).to(device)

    print('Done.')

    criterion = nn.CrossEntropyLoss(weight=weights).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, cfg.train.scheduler_gamma)

    # if cfg.train.load:
    #     load_path = Path(cfg.trian.load)
    #     path = max(list(load_path.glob('*.pth')))
    #     print(f'Loading {path}...')
    #     checkpoint = torch.load(path)
    #     first_epoch = checkpoint['epoch'] + 1
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"当前设备: {torch.cuda.current_device()} → {torch.cuda.get_device_name()}")
    top_checkpoints = []
    epochs = int(cfg.train.num_epochs)
    for epoch in range(1, epochs + 1):
        print(f'====== EPOCH {epoch:d}/{epochs:d} ======')
        t0 = time.time()
        # Train
        model.train()
        # metrics
        losses = []
        accuracies = []
        ious = []

        scaler = GradScaler('cuda')  # 或 torch.amp.GradScaler('cuda')
        # iterate over dataset
        for batch_idx, points in enumerate(train_loader):
            points = {k: v.to(device, non_blocking=True) for k, v in points.items()}
            feature = points['xyz']
            labels = points['class'].long()

            if cfg.data.HasRGB:
                rgb = points['rgb']
                feature = torch.cat((feature, rgb), dim=2)
            if cfg.data.HasIntensity:
                intensity = points['intensity'].unsqueeze(-1)
                feature = torch.cat((feature, intensity), dim=2)

            optimizer.zero_grad()

            scores = model(feature)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()

            # with autocast('cuda'):  # 指定设备类型
            #     scores = model(feature)
            #     loss = criterion(scores, labels)
            # scaler.scale(loss).backward()  # 缩放梯度
            # scaler.step(optimizer)  # 更新参数
            # scaler.update()  # 调整缩放因子[1,2](@ref)

            losses.append(loss.cpu().item())
            if batch_idx % 10 == 0:  # 每10batch计算一次指标
                acc, miou = compute_metrics(scores, labels, num_classes)
                accuracies.append(acc)
                ious.append(miou)

        scheduler.step()
        accs = np.nanmean(np.array(accuracies), axis=0)
        ious = np.nanmean(np.array(ious), axis=0)

        val_loss, val_accs, val_ious = evaluate(
            model,
            val_loader,
            criterion,
            device,
            num_classes,
            cfg
        )

        loss_dict = {
            'Training loss': np.mean(losses),
            'Validation loss': val_loss
        }
        acc_dicts = {
            'Training accuracy': accs,
            'Validation accuracy': val_accs
        }
        iou_dicts = {
            'Training accuracy': ious,
            'Validation accuracy': val_ious
        }

        t1 = time.time()
        d = t1 - t0
        # Display results

        print('Accuracy     ', '   OA', sep=' | ')
        print('Training:    ', *[f'{accs:.3f}' if not np.isnan(accs) else '  nan'], sep=' | ')
        print('Validation:  ', *[f'{val_accs:.3f}' if not np.isnan(accs) else '  nan'], sep=' | ')
        print('----------------------')
        print('MIoU         ', ' mIoU', sep=' | ')
        print('Training:    ', *[f'{ious:.3f}' if not np.isnan(ious) else '  nan'], sep=' | ')
        print('Validation:  ', *[f'{val_ious:.3f}' if not np.isnan(ious) else '  nan'], sep=' | ')
        for k, v in loss_dict.items():
            print(f'{k}: {v:.7f}', end='\n')
        print('Time elapsed:', '{:.0f} s'.format(d) if d < 60 else '{:.0f} min {:02.0f} s'.format(*divmod(d, 60)))
        print('')
        with SummaryWriter(logs_dir) as writer:
            # send results to tensorboard
            writer.add_scalars('Loss', loss_dict, epoch)

            writer.add_scalars('Per-class accuracy/Overall', acc_dicts, epoch)
            writer.add_scalars('Per-class IoU/Mean IoU', iou_dicts, epoch)

        checkpoint_path = os.path.join(checkpoint_dir,
                                           f'checkpoint_{epoch:02d}_ACC_{val_accs:.2f}_MIOU_{val_ious:.2f}.pth')

        # 更新Top-5 Checkpoints
        if len(top_checkpoints) < 5 or val_ious > min(top_checkpoints, key=lambda x: x[0])[0]:
            path = checkpoint_path
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mIoU': val_ious
            }, path)

            top_checkpoints.append((val_ious, path))
            top_checkpoints.sort(reverse=True, key=lambda x: x[0])
            # 清理旧checkpoint
            while len(top_checkpoints) > 5:
                _, old_path = top_checkpoints.pop()
                os.remove(old_path)

            top_checkpoints = top_checkpoints[:5]


