import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np


class FocusNetDataset(Dataset):
    """Dataset for Focus-Net training.

    Loads individual focal stack images and their corresponding focus maps.
    """
    def __init__(self, root_dir, transform=None, num_frames=6):
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames

        self.samples = []
        images_dir = os.path.join(root_dir, 'images')
        if os.path.exists(images_dir):
            for scene in sorted(os.listdir(images_dir)):
                if os.path.isdir(os.path.join(images_dir, scene)):
                    self.samples.append(scene)

    def __len__(self):
        return len(self.samples) * self.num_frames

    def __getitem__(self, idx):
        scene_idx = idx // self.num_frames
        frame_idx = (idx % self.num_frames) + 1
        scene = self.samples[scene_idx]

        # Load image
        img_path = os.path.join(self.root_dir, 'images', scene, f'defocus_{frame_idx}.tif')
        img = Image.open(img_path).convert('RGB')

        # Load focus map label
        label_path = os.path.join(self.root_dir, 'focusmap', scene, f'focusmap_{frame_idx}.png')
        label = Image.open(label_path).convert('L')

        if self.transform:
            img = self.transform(img)
        label = transforms.ToTensor()(label)

        return img, label


class FusionNetDataset(Dataset):
    """Dataset for Fusion-Net training.

    Loads complete focal stacks (6 images), focus maps, and ground truth all-in-focus images.
    Supports random cropping to 128x128 patches as specified in the paper.
    """
    def __init__(self, root_dir, transform=None, num_frames=6, crop_size=None):
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.crop_size = crop_size

        self.samples = []
        images_dir = os.path.join(root_dir, 'images')
        if os.path.exists(images_dir):
            for scene in sorted(os.listdir(images_dir)):
                if os.path.isdir(os.path.join(images_dir, scene)):
                    self.samples.append(scene)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene = self.samples[idx]

        # Load focal stack images
        focal_stack = []
        for i in range(1, self.num_frames + 1):
            img_path = os.path.join(self.root_dir, 'images', scene, f'defocus_{i}.tif')
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            focal_stack.append(img)
        focal_stack = torch.cat(focal_stack, dim=0)  # [18, H, W]

        # Load focus maps
        focus_maps = []
        for i in range(1, self.num_frames + 1):
            map_path = os.path.join(self.root_dir, 'focusmap', scene, f'focusmap_{i}.png')
            fmap = Image.open(map_path).convert('L')
            if self.transform:
                fmap = transforms.ToTensor()(fmap)
            focus_maps.append(fmap)
        focus_maps = torch.cat(focus_maps, dim=0)  # [6, H, W]

        # Load ground truth
        gt_path = os.path.join(self.root_dir, 'labels', scene, 'all_in_focus_1.tif')
        gt = Image.open(gt_path).convert('RGB')
        if self.transform:
            gt = self.transform(gt)

        # Random crop if specified
        if self.crop_size is not None:
            _, H, W = focal_stack.shape
            ch, cw = self.crop_size
            if H > ch and W > cw:
                h_start = torch.randint(0, H - ch, (1,)).item()
                w_start = torch.randint(0, W - cw, (1,)).item()
                focal_stack = focal_stack[:, h_start:h_start + ch, w_start:w_start + cw]
                focus_maps = focus_maps[:, h_start:h_start + ch, w_start:w_start + cw]
                gt = gt[:, h_start:h_start + ch, w_start:w_start + cw]

        return focal_stack, focus_maps, gt


class AlignmentNetDataset(Dataset):
    """Dataset for Alignment-Net training.

    Loads image pairs for optical flow estimation.
    Can be used with FlyingThings dataset or MFM-CDP misaligned pairs.
    """
    def __init__(self, root_dir, transform=None, num_frames=6):
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames

        self.samples = []
        images_dir = os.path.join(root_dir, 'images')
        if os.path.exists(images_dir):
            for scene in sorted(os.listdir(images_dir)):
                if os.path.isdir(os.path.join(images_dir, scene)):
                    self.samples.append(scene)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene = self.samples[idx]

        # Load all frames
        frames = []
        for i in range(1, self.num_frames + 1):
            img_path = os.path.join(self.root_dir, 'images', scene, f'defocus_{i}.tif')
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        # Reference frame (first frame)
        ref = frames[0]  # [3, H, W]

        # Source frames (remaining)
        src = torch.stack(frames[1:], dim=0)  # [N-1, 3, H, W]

        # For training with flow supervision, load flow ground truth if available
        # Otherwise, use self-supervised approach
        flow_dir = os.path.join(self.root_dir, 'flow', scene)
        flows = None
        if os.path.exists(flow_dir):
            flow_list = []
            for i in range(1, self.num_frames):
                flow_path = os.path.join(flow_dir, f'flow_{i}.flo')
                if os.path.exists(flow_path):
                    flow = self._read_flow(flow_path)
                    flow_list.append(flow)
            if flow_list:
                flows = torch.stack(flow_list, dim=0)

        return ref, src, flows

    @staticmethod
    def _read_flow(path):
        """Read .flo optical flow file (Middlebury format)."""
        with open(path, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)[0]
            if magic != 202021.25:
                raise ValueError('Invalid .flo file')
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2 * w * h)
            flow = np.reshape(data, (h, w, 2))
        return torch.from_numpy(flow).permute(2, 0, 1)


class InferenceDataset(Dataset):
    """Dataset for inference on real-world or rendered test data."""
    def __init__(self, root_dir, transform=None, num_frames=6):
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames

        self.samples = []
        images_dir = os.path.join(root_dir, 'images')
        if os.path.exists(images_dir):
            for scene in sorted(os.listdir(images_dir)):
                if os.path.isdir(os.path.join(images_dir, scene)):
                    self.samples.append(scene)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene = self.samples[idx]

        # Load focal stack
        frames = []
        for i in range(1, self.num_frames + 1):
            img_path = os.path.join(self.root_dir, 'images', scene, f'defocus_{i}.tif')
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        focal_stack = torch.stack(frames, dim=0)  # [N, 3, H, W]

        # Load focus maps if available
        focus_maps = None
        fm_dir = os.path.join(self.root_dir, 'focusmap', scene)
        if os.path.exists(fm_dir):
            fm_list = []
            for i in range(1, self.num_frames + 1):
                map_path = os.path.join(fm_dir, f'focusmap_{i}.png')
                fmap = Image.open(map_path).convert('L')
                if self.transform:
                    fmap = transforms.ToTensor()(fmap)
                fm_list.append(fmap)
            focus_maps = torch.cat(fm_list, dim=0)

        # Load GT if available
        gt = None
        gt_path = os.path.join(self.root_dir, 'labels', scene, 'all_in_focus_1.tif')
        if os.path.exists(gt_path):
            gt_img = Image.open(gt_path).convert('RGB')
            if self.transform:
                gt = self.transform(gt_img)

        return focal_stack, focus_maps, gt, scene
