"""FIIF-Net Inference Pipeline

Full pipeline:
1. Alignment-Net aligns focal stack images
2. Focus-Net predicts focus maps
3. ZSSR performs super-resolution
4. Fusion-Net produces all-in-focus image
"""
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import argparse
from tqdm import tqdm

from model.fiif_net import FIIFNet
from model.zssr import apply_zssr_batch
from dataset import InferenceDataset


def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Dataset
    test_dataset = InferenceDataset(
        root_dir=args.test_dir,
        transform=transform,
        num_frames=6
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    # Load model
    model = FIIFNet(
        num_frames=6,
        in_channels=3,
        use_zssr=args.use_zssr,
        align_hidden=64,
        focus_out=1,
        fusion_base=64,
        zssr_iters=args.zssr_iters,
        zssr_scale=args.zssr_scale
    ).to(device)

    # Load weights
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f'Loaded model from {args.checkpoint}')

    model.eval()

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for focal_stack, focus_maps_gt, gt, scene_name in tqdm(test_loader, desc='Inference'):
            focal_stack = focal_stack.to(device)  # [1, N, 3, H, W]

            # Full pipeline forward
            fused_img, aligned_imgs, focus_maps, flows = model(focal_stack)

            # Save fused result
            output_img = fused_img.cpu().squeeze(0)  # [3, H, W]
            output_img = output_img.permute(1, 2, 0).numpy()
            output_img = (output_img * 255).clip(0, 255).astype(np.uint8)
            output_img = Image.fromarray(output_img)

            scene_dir = os.path.join(args.output_dir, scene_name[0])
            os.makedirs(scene_dir, exist_ok=True)
            output_img.save(os.path.join(scene_dir, 'fused_result.tif'))

            # Save aligned images if requested
            if args.save_aligned:
                for i in range(aligned_imgs.shape[1]):
                    ali_img = aligned_imgs[0, i].cpu().permute(1, 2, 0).numpy()
                    ali_img = (ali_img * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(ali_img).save(
                        os.path.join(scene_dir, f'aligned_{i+1}.tif')
                    )

            # Save focus maps if requested
            if args.save_focus:
                for i in range(focus_maps.shape[1]):
                    fm = focus_maps[0, i].cpu().numpy()
                    fm = (fm * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(fm).save(
                        os.path.join(scene_dir, f'focusmap_{i+1}.png')
                    )

            # Compute metrics if GT available
            if gt is not None:
                gt_np = gt.numpy().squeeze(0).transpose(1, 2, 0)
                fused_np = fused_img.cpu().squeeze(0).permute(1, 2, 0).numpy()

                # Normalize to [0, 1]
                gt_np = gt_np.clip(0, 1)
                fused_np = fused_np.clip(0, 1)

                metrics = compute_metrics(fused_np, gt_np)
                print(f'Scene: {scene_name[0]} | '
                      f'PSNR: {metrics["psnr"]:.2f} | '
                      f'SSIM: {metrics["ssim"]:.4f} | '
                      f'RMSE: {metrics["rmse"]:.4f}')

    print(f'Results saved to {args.output_dir}')


def compute_metrics(pred, gt):
    """Compute reference-based metrics."""
    # PSNR
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))

    # SSIM (simplified)
    from scipy.ndimage import uniform_filter
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    mu_pred = uniform_filter(pred, size=11)
    mu_gt = uniform_filter(gt, size=11)
    mu_pred_sq = mu_pred ** 2
    mu_gt_sq = mu_gt ** 2
    mu_pred_gt = mu_pred * mu_gt
    sigma_pred_sq = uniform_filter(pred ** 2, size=11) - mu_pred_sq
    sigma_gt_sq = uniform_filter(gt ** 2, size=11) - mu_gt_sq
    sigma_pred_gt = uniform_filter(pred * gt, size=11) - mu_pred_gt
    ssim_map = ((2 * mu_pred_gt + C1) * (2 * sigma_pred_gt + C2)) / \
               ((mu_pred_sq + mu_gt_sq + C1) * (sigma_pred_sq + sigma_gt_sq + C2))
    ssim = ssim_map.mean()

    # RMSE
    rmse = np.sqrt(mse)

    return {'psnr': psnr, 'ssim': ssim, 'rmse': rmse}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FIIF-Net Inference')
    parser.add_argument('--test_dir', type=str, default='./dataset/test',
                        help='Test data directory')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory')
    parser.add_argument('--checkpoint', type=str, default='./model_pth/fiif_net_best.pth',
                        help='Model checkpoint path')
    parser.add_argument('--use_zssr', action='store_true', help='Use ZSSR super-resolution')
    parser.add_argument('--zssr_iters', type=int, default=500, help='ZSSR iterations')
    parser.add_argument('--zssr_scale', type=float, default=2.0, help='ZSSR scale factor')
    parser.add_argument('--save_aligned', action='store_true', help='Save aligned images')
    parser.add_argument('--save_focus', action='store_true', help='Save focus maps')
    args = parser.parse_args()

    inference(args)
