"""Unified training script for FIIF-Net.

Two-stage training procedure:
Stage 1: Individual optimization of Alignment-Net and Focus-Net
Stage 2: End-to-end joint training with Fusion-Net

Usage:
    # Full pipeline (both stages)
    python train.py --stage both --train_dir ./dataset/train --val_dir ./dataset/test

    # Only Stage 1
    python train.py --stage 1 --train_dir ./dataset/train

    # Only Stage 2 (requires pre-trained models from Stage 1)
    python train.py --stage 2 --train_dir ./dataset/train --val_dir ./dataset/test
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os
import logging
import argparse

from model.fiif_net import FIIFNet
from model.alignment_net import AlignmentNet
from model.focus_net import FocusNet
from model.fusion_net import FusionNet
from dataset import FocusNetDataset, FusionNetDataset, AlignmentNetDataset
from loss import FocusNetLoss, FusionNetLoss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_stage1_focus(args):
    """Stage 1a: Train Focus-Net."""
    logger.info('=' * 50)
    logger.info('Stage 1a: Training Focus-Net')
    logger.info('=' * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = FocusNetDataset(
        root_dir=args.train_dir, transform=train_transform, num_frames=6
    )
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)

    model = FocusNet(in_channels=3, out_channels=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = FocusNetLoss()
    scaler = GradScaler(enabled=True)

    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(1, 51):
        model.train()
        epoch_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f'FocusNet Epoch {epoch}/50'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs, aux1, aux2, aux3 = model(inputs)
                labels_aux1 = nn.functional.interpolate(labels, size=aux1.shape[2:],
                                                        mode='bicubic', align_corners=False)
                labels_aux2 = nn.functional.interpolate(labels, size=aux2.shape[2:],
                                                        mode='bicubic', align_corners=False)
                labels_aux3 = nn.functional.interpolate(labels, size=aux3.shape[2:],
                                                        mode='bicubic', align_corners=False)
                total_loss, _, _, _, _ = criterion(
                    outputs, labels, aux1, aux2, aux3,
                    labels_aux1, labels_aux2, labels_aux3
                )

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        logger.info(f'Epoch {epoch}/50 - Loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'focus_net_best.pth'))

    torch.save(model.state_dict(), os.path.join(args.save_dir, 'focus_net_final.pth'))
    logger.info('Focus-Net training completed.')


def train_stage1_alignment(args):
    """Stage 1b: Train Alignment-Net."""
    logger.info('=' * 50)
    logger.info('Stage 1b: Training Alignment-Net')
    logger.info('=' * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = AlignmentNetDataset(
        root_dir=args.train_dir, transform=train_transform, num_frames=6
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)

    model = AlignmentNet(in_channels=3, hidden_dim=64).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    l1_loss = nn.L1Loss()
    scaler = GradScaler(enabled=True)

    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(1, 101):
        model.train()
        epoch_loss = 0.0

        for ref, src, flow_gt in tqdm(train_loader, desc=f'AlignNet Epoch {epoch}/100'):
            ref, src = ref.to(device), src.to(device)
            optimizer.zero_grad()

            with autocast():
                aligned_imgs, pred_flows = model(ref, src)
                B, N, C, H, W = aligned_imgs.shape
                ref_expand = ref.unsqueeze(1).expand_as(aligned_imgs)
                photo_loss = l1_loss(aligned_imgs, ref_expand)

                # Flow smoothness
                diff_h = pred_flows[:, :, :, 1:, :] - pred_flows[:, :, :, :-1, :]
                diff_w = pred_flows[:, :, :, :, 1:] - pred_flows[:, :, :, :, :-1]
                smooth_loss = (diff_h.abs().mean() + diff_w.abs().mean())

                total_loss = photo_loss + 0.1 * smooth_loss

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        logger.info(f'Epoch {epoch}/100 - Loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'alignment_net_best.pth'))

    torch.save(model.state_dict(), os.path.join(args.save_dir, 'alignment_net_final.pth'))
    logger.info('Alignment-Net training completed.')


def train_stage2_joint(args):
    """Stage 2: Joint end-to-end training."""
    logger.info('=' * 50)
    logger.info('Stage 2: Joint Training with Fusion-Net')
    logger.info('=' * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = FusionNetDataset(
        root_dir=args.train_dir, transform=train_transform,
        num_frames=6, crop_size=(128, 128)
    )
    val_dataset = FusionNetDataset(
        root_dir=args.val_dir, transform=train_transform,
        num_frames=6, crop_size=None
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Initialize FIIF-Net
    model = FIIFNet(
        num_frames=6, in_channels=3, use_zssr=False,
        align_hidden=64, focus_out=1, fusion_base=64
    ).to(device)

    # Load pre-trained sub-networks
    align_pth = os.path.join(args.save_dir, 'alignment_net_best.pth')
    focus_pth = os.path.join(args.save_dir, 'focus_net_best.pth')

    if os.path.exists(align_pth):
        model.alignment_net.load_state_dict(torch.load(align_pth, map_location=device))
        logger.info(f'Loaded Alignment-Net from {align_pth}')
    if os.path.exists(focus_pth):
        model.focus_net.load_state_dict(torch.load(focus_pth, map_location=device))
        logger.info(f'Loaded Focus-Net from {focus_pth}')

    # Differential learning rates
    optimizer = optim.AdamW([
        {'params': model.alignment_net.parameters(), 'lr': 0.00001},
        {'params': model.focus_net.parameters(), 'lr': 0.00001},
        {'params': model.fusion_net.parameters(), 'lr': 0.0001},
    ], lr=0.0001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = FusionNetLoss()
    scaler = GradScaler(enabled=True)

    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(1, 51):
        model.train()
        epoch_loss = 0.0

        for focal_stack, focus_maps, targets in tqdm(train_loader, desc=f'Joint Epoch {epoch}/50'):
            focal_stack = focal_stack.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            with autocast():
                B, C_total, H, W = focal_stack.shape
                focal_stack_reshaped = focal_stack.view(B, 6, 3, H, W)

                # Full forward
                ref_img = focal_stack_reshaped[:, 0]
                src_imgs = focal_stack_reshaped[:, 1:]
                aligned_src, flows = model.alignment_net(ref_img, src_imgs)
                aligned_imgs = torch.cat([ref_img.unsqueeze(1), aligned_src], dim=1)

                focus_maps_pred = []
                for i in range(6):
                    fm, _, _, _ = model.focus_net(aligned_imgs[:, i])
                    focus_maps_pred.append(fm)
                focus_maps_pred = torch.cat(focus_maps_pred, dim=1)

                img_stack = aligned_imgs.view(B, 18, H, W)
                fused_img = model.fusion_net(img_stack, focus_maps_pred)

                total_loss, _, _, _ = criterion(fused_img, targets)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        logger.info(f'Epoch {epoch}/50 - Loss: {avg_loss:.4f}')

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for focal_stack, focus_maps, targets in val_loader:
                focal_stack = focal_stack.to(device)
                targets = targets.to(device)
                B = focal_stack.shape[0]
                fs = focal_stack.view(B, 6, 3, focal_stack.shape[2], focal_stack.shape[3])
                fused, _, _, _ = model(fs)
                loss, _, _, _ = criterion(fused, targets)
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)
        logger.info(f'Val Loss: {avg_val:.4f}')

        if avg_val < best_loss:
            best_loss = avg_val
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
            }, os.path.join(args.save_dir, 'fiif_net_best.pth'))
            logger.info(f'Best model saved at epoch {epoch}')

    torch.save(model.state_dict(), os.path.join(args.save_dir, 'fiif_net_final.pth'))
    logger.info('FIIF-Net joint training completed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FIIF-Net Training')
    parser.add_argument('--stage', type=str, default='both', choices=['1', '2', 'both'],
                        help='Training stage: 1 (individual), 2 (joint), or both')
    parser.add_argument('--train_dir', type=str, default='./dataset/train',
                        help='Training data directory')
    parser.add_argument('--val_dir', type=str, default='./dataset/test',
                        help='Validation data directory')
    parser.add_argument('--save_dir', type=str, default='./model_pth',
                        help='Model save directory')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for joint training')
    args = parser.parse_args()

    if args.stage in ['1', 'both']:
        train_stage1_focus(args)
        train_stage1_alignment(args)

    if args.stage in ['2', 'both']:
        train_stage2_joint(args)
