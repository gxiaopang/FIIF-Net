"""Stage 1: Alignment-Net Pre-training

Paper settings:
- Trained on FlyingThings dataset
- AdamW optimizer, lr=0.0001
- Batch size 18
- 100 epochs
- Random parameter initialization
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

from model.alignment_net import AlignmentNet
from dataset import AlignmentNetDataset
from loss import FocusNetLoss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_alignment_net(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Data transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Dataset and DataLoader
    train_dataset = AlignmentNetDataset(
        root_dir=args.train_dir,
        transform=train_transform,
        num_frames=6
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)

    # Model
    model = AlignmentNet(in_channels=3, hidden_dim=64, num_levels=4, num_iters=6).to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Loss for optical flow (L1 + smoothness)
    l1_loss = nn.L1Loss()

    # AMP scaler
    scaler = GradScaler(enabled=args.amp)

    # Save directory
    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for ref, src, flow_gt in pbar:
            ref = ref.to(device)
            src = src.to(device)

            optimizer.zero_grad()

            with autocast(enabled=args.amp):
                # Forward pass
                aligned_imgs, pred_flows = model(ref, src)

                if flow_gt is not None:
                    # Supervised flow loss
                    flow_gt = flow_gt.to(device)
                    flow_loss = l1_loss(pred_flows, flow_gt)
                else:
                    # Self-supervised: photometric consistency loss
                    # Aligned images should be similar to reference
                    B, N, C, H, W = aligned_imgs.shape
                    ref_expand = ref.unsqueeze(1).expand_as(aligned_imgs)
                    photo_loss = l1_loss(aligned_imgs, ref_expand)

                    # Smoothness loss on flow
                    smooth_loss = flow_smoothness_loss(pred_flows)

                    flow_loss = photo_loss + 0.1 * smooth_loss

            scaler.scale(flow_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += flow_loss.item()
            pbar.set_postfix({'loss': flow_loss.item()})

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f'Epoch [{epoch}/{args.epochs}] Loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'alignment_net_best.pth'))
            logger.info(f'Best model saved at epoch {epoch}, Loss: {best_loss:.4f}')

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'alignment_net_final.pth'))
    logger.info('Alignment-Net training completed.')


def flow_smoothness_loss(flow):
    """Smoothness loss for optical flow."""
    diff_h = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
    diff_w = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]
    return (diff_h.abs().mean() + diff_w.abs().mean())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Alignment-Net Training (Stage 1)')
    parser.add_argument('--train_dir', type=str, default='./dataset/train',
                        help='Training data directory (FlyingThings or MFM-CDP)')
    parser.add_argument('--save_dir', type=str, default='./model_pth',
                        help='Model save directory')
    parser.add_argument('--batch_size', type=int, default=18, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision')
    args = parser.parse_args()

    train_alignment_net(args)
