"""Stage 1: Focus-Net Pre-training

Paper settings:
- 100 focal stack images
- AdamW optimizer, lr=0.0001
- Batch size 2
- 50 epochs
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

from model.focus_net import FocusNet
from dataset import FocusNetDataset
from loss import FocusNetLoss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_focus_net(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Data transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Dataset and DataLoader
    train_dataset = FocusNetDataset(
        root_dir=args.train_dir,
        transform=train_transform,
        num_frames=6
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)

    # Model
    model = FocusNet(in_channels=3, out_channels=1).to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Loss
    criterion = FocusNetLoss()

    # AMP scaler
    scaler = GradScaler(enabled=args.amp)

    # Save directory
    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        content_loss_sum = 0.0
        aux_loss_sum = 0.0
        sobel_loss_sum = 0.0
        tv_loss_sum = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast(enabled=args.amp):
                outputs, aux1, aux2, aux3 = model(inputs)

                # Downsample labels for auxiliary outputs
                labels_aux1 = nn.functional.interpolate(labels, size=aux1.shape[2:],
                                                        mode='bicubic', align_corners=False)
                labels_aux2 = nn.functional.interpolate(labels, size=aux2.shape[2:],
                                                        mode='bicubic', align_corners=False)
                labels_aux3 = nn.functional.interpolate(labels, size=aux3.shape[2:],
                                                        mode='bicubic', align_corners=False)

                total_loss, content_loss, aux_loss, sobel_loss, tv = criterion(
                    outputs, labels, aux1, aux2, aux3,
                    labels_aux1, labels_aux2, labels_aux3
                )

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()
            content_loss_sum += content_loss.item()
            aux_loss_sum += aux_loss.item()
            sobel_loss_sum += sobel_loss.item()
            tv_loss_sum += tv.item()

            pbar.set_postfix({
                'loss': total_loss.item(),
                'content': content_loss.item(),
                'aux': aux_loss.item()
            })

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        logger.info(
            f'Epoch [{epoch}/{args.epochs}] '
            f'Total: {avg_loss:.4f} '
            f'Content: {content_loss_sum / len(train_loader):.4f} '
            f'Aux: {aux_loss_sum / len(train_loader):.4f} '
            f'Sobel: {sobel_loss_sum / len(train_loader):.4f} '
            f'TV: {tv_loss_sum / len(train_loader):.6f}'
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'focus_net_best.pth'))
            logger.info(f'Best model saved at epoch {epoch}, Loss: {best_loss:.4f}')

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'focus_net_final.pth'))
    logger.info('Focus-Net training completed.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Focus-Net Training (Stage 1)')
    parser.add_argument('--train_dir', type=str, default='./dataset/train',
                        help='Training data directory')
    parser.add_argument('--save_dir', type=str, default='./model_pth',
                        help='Model save directory')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision')
    args = parser.parse_args()

    train_focus_net(args)
