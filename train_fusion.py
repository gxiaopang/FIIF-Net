"""Stage 2: Joint End-to-End Training with Fusion-Net

Paper settings:
- 20,000 randomly cropped 128x128 patches
- AdamW optimizer
- lr=0.00001 for Alignment-Net and Focus-Net
- lr=0.0001 for Fusion-Net
- Batch size 64
- 50 epochs
- Pre-trained Alignment-Net and Focus-Net loaded first
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

from model.fiif_net import FIIFNet
from dataset import FusionNetDataset
from loss import FusionNetLoss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_fiif_net(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Data transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Dataset with random cropping to 128x128
    train_dataset = FusionNetDataset(
        root_dir=args.train_dir,
        transform=train_transform,
        num_frames=6,
        crop_size=(128, 128)
    )
    val_dataset = FusionNetDataset(
        root_dir=args.val_dir,
        transform=train_transform,
        num_frames=6,
        crop_size=None
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=args.num_workers)

    # Initialize FIIF-Net
    model = FIIFNet(
        num_frames=6,
        in_channels=3,
        use_zssr=False,  # ZSSR is applied during inference, not training
        align_hidden=64,
        focus_out=1,
        fusion_base=64
    ).to(device)

    # Load pre-trained sub-networks
    if args.alignment_pth and os.path.exists(args.alignment_pth):
        logger.info(f'Loading pre-trained Alignment-Net from {args.alignment_pth}')
        model.alignment_net.load_state_dict(
            torch.load(args.alignment_pth, map_location=device)
        )

    if args.focus_pth and os.path.exists(args.focus_pth):
        logger.info(f'Loading pre-trained Focus-Net from {args.focus_pth}')
        model.focus_net.load_state_dict(
            torch.load(args.focus_pth, map_location=device)
        )

    # Differential learning rates
    optimizer = optim.AdamW([
        {'params': model.alignment_net.parameters(), 'lr': args.lr_align},
        {'params': model.focus_net.parameters(), 'lr': args.lr_focus},
        {'params': model.fusion_net.parameters(), 'lr': args.lr_fusion},
    ], lr=args.lr_fusion)

    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Loss
    criterion = FusionNetLoss()

    # AMP scaler
    scaler = GradScaler(enabled=args.amp)

    # Save directory
    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        epoch_loss = 0.0
        content_loss_sum = 0.0
        perceptual_loss_sum = 0.0
        color_loss_sum = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for focal_stack, focus_maps, targets in pbar:
            focal_stack = focal_stack.to(device)
            focus_maps = focus_maps.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            with autocast(enabled=args.amp):
                # Forward through full pipeline
                B, C_total, H, W = focal_stack.shape
                N = 6
                C = 3
                focal_stack_reshaped = focal_stack.view(B, N, C, H, W)

                # Step 1: Alignment
                ref_img = focal_stack_reshaped[:, 0]
                src_imgs = focal_stack_reshaped[:, 1:]
                aligned_src, flows = model.alignment_net(ref_img, src_imgs)
                aligned_imgs = torch.cat([ref_img.unsqueeze(1), aligned_src], dim=1)

                # Step 2: Focus prediction
                focus_maps_pred = []
                for i in range(N):
                    fm, _, _, _ = model.focus_net(aligned_imgs[:, i])
                    focus_maps_pred.append(fm)
                focus_maps_pred = torch.cat(focus_maps_pred, dim=1)

                # Step 3: Fusion
                img_stack = aligned_imgs.view(B, N * C, H, W)
                fused_img = model.fusion_net(img_stack, focus_maps_pred)

                # Compute loss
                total_loss, content_loss, perceptual_loss, color_loss = criterion(
                    fused_img, targets
                )

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()
            content_loss_sum += content_loss.item()
            perceptual_loss_sum += perceptual_loss.item()
            color_loss_sum += color_loss.item()

            pbar.set_postfix({
                'loss': total_loss.item(),
                'content': content_loss.item(),
                'perceptual': perceptual_loss.item(),
                'color': color_loss.item()
            })

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        logger.info(
            f'Epoch [{epoch}/{args.epochs}] '
            f'Total: {avg_loss:.4f} '
            f'Content: {content_loss_sum / len(train_loader):.4f} '
            f'Perceptual: {perceptual_loss_sum / len(train_loader):.4f} '
            f'Color: {color_loss_sum / len(train_loader):.4f}'
        )

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for focal_stack, focus_maps, targets in tqdm(val_loader, desc='Validation'):
                focal_stack = focal_stack.to(device)
                targets = targets.to(device)

                B = focal_stack.shape[0]
                focal_stack_reshaped = focal_stack.view(B, 6, 3,
                                                         focal_stack.shape[2],
                                                         focal_stack.shape[3])
                fused_img, _, _, _ = model(focal_stack_reshaped)
                total_loss, _, _, _ = criterion(fused_img, targets)
                val_loss += total_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        logger.info(f'Validation Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(args.save_dir, 'fiif_net_best.pth'))
            logger.info(f'Best model saved at epoch {epoch}, Loss: {best_loss:.4f}')

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(args.save_dir, 'fiif_net_final.pth'))
    logger.info('FIIF-Net joint training completed.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='FIIF-Net Joint Training (Stage 2)')
    parser.add_argument('--train_dir', type=str, default='./dataset/train',
                        help='Training data directory')
    parser.add_argument('--val_dir', type=str, default='./dataset/test',
                        help='Validation data directory')
    parser.add_argument('--save_dir', type=str, default='./model_pth',
                        help='Model save directory')
    parser.add_argument('--alignment_pth', type=str, default='./model_pth/alignment_net_best.pth',
                        help='Pre-trained Alignment-Net weights')
    parser.add_argument('--focus_pth', type=str, default='./model_pth/focus_net_best.pth',
                        help='Pre-trained Focus-Net weights')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr_align', type=float, default=0.00001,
                        help='Learning rate for Alignment-Net')
    parser.add_argument('--lr_focus', type=float, default=0.00001,
                        help='Learning rate for Focus-Net')
    parser.add_argument('--lr_fusion', type=float, default=0.0001,
                        help='Learning rate for Fusion-Net')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision')
    args = parser.parse_args()

    train_fiif_net(args)
