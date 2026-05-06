"""Test forward propagation of FIIF-Net."""
import torch
from model.fiif_net import FIIFNet

def test_forward(H, W, mode='train'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== Testing {mode} mode, input {H}x{W} ===")

    model = FIIFNet(num_frames=6, in_channels=3, use_zssr=False).to(device)
    if mode == 'train':
        model.train()
    else:
        model.eval()

    B, N, C = 1, 6, 3
    focal_stack = torch.randn(B, N, C, H, W, device=device)

    try:
        if mode == 'train':
            # Test forward + backward
            fused_img, aligned_imgs, focus_maps, flows = model(focal_stack)
            loss = fused_img.mean()
            loss.backward()
            print(f"Forward + backward success!")
        else:
            with torch.no_grad():
                fused_img, aligned_imgs, focus_maps, flows = model(focal_stack)
            print(f"Forward (eval) success!")

        print(f"  fused_img:    {fused_img.shape}")
        print(f"  aligned_imgs: {aligned_imgs.shape}")
        print(f"  focus_maps:   {focus_maps.shape}")
        print(f"  flows:        {flows.shape}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_forward(256, 256, mode='train')
    test_forward(256, 256, mode='eval')
    test_forward(600, 600, mode='train')
    test_forward(600, 600, mode='eval')
