import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torchvision.models


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                          for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).double().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class MEFSSIM(nn.Module):
    """Multi-Exposure Fusion SSIM loss."""
    def __init__(self, window_size=11, channel=3, sigma_g=0.2, sigma_l=0.2,
                 c1=0.01, c2=0.03, is_lum=False):
        super(MEFSSIM, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = create_window(window_size, self.channel)
        self.denom_g = 2 * sigma_g ** 2
        self.denom_l = 2 * sigma_l ** 2
        self.C1 = c1 ** 2
        self.C2 = c2 ** 2
        self.is_lum = is_lum

    def forward(self, X, Ys):
        (_, channel, _, _) = Ys.size()
        if channel == self.channel and self.window.data.type() == Ys.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if Ys.is_cuda:
                window = window.cuda(Ys.get_device())
            window = window.type_as(Ys)
            self.window = window
            self.channel = channel

        K, C, H, W = list(Ys.size())
        ws = self.window_size

        muY_seq = F.conv2d(Ys, window, padding=ws // 2, groups=C).view(K, C, H, W)
        muY_sq_seq = muY_seq * muY_seq
        sigmaY_sq_seq = F.conv2d(Ys * Ys, window, padding=ws // 2, groups=C).view(K, C, H, W) \
            - muY_sq_seq
        sigmaY_sq, patch_index = torch.max(sigmaY_sq_seq, dim=0)

        muX = F.conv2d(X, window, padding=ws // 2, groups=C).view(C, H, W)
        muX_sq = muX * muX
        sigmaX_sq = F.conv2d(X * X, window, padding=ws // 2, groups=C).view(C, H, W) - muX_sq

        sigmaXY = F.conv2d(X.expand_as(Ys) * Ys, window, padding=ws // 2, groups=C) \
            .view(K, C, H, W) - muX.expand_as(muY_seq) * muY_seq

        cs_seq = (2 * sigmaXY + self.C2) / (sigmaX_sq + sigmaY_sq_seq + self.C2)
        cs_map = torch.gather(cs_seq.view(K, -1), 0, patch_index.view(1, -1)).view(C, H, W)

        if self.is_lum:
            lY = torch.mean(muY_seq.view(K, -1), dim=1)
            lL = torch.exp(-((muY_seq - 0.5) ** 2) / self.denom_l)
            lG = torch.exp(-((lY - 0.5) ** 2) / self.denom_g)[:, None, None].expand_as(lL)
            LY = lG * lL
            muY = torch.sum((LY * muY_seq), dim=0) / torch.sum(LY, dim=0)
            muY_sq = muY * muY
            l_map = (2 * muX * muY + self.C1) / (muX_sq + muY_sq + self.C1)
        else:
            l_map = torch.Tensor([1.0])
            if Ys.is_cuda:
                l_map = l_map.cuda(Ys.get_device())

        qmap = l_map * cs_map
        q = qmap.mean()
        return q


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_model = torchvision.models.vgg19(pretrained=True)
        vgg_pretrained_features = vgg_model.features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.weights = [10.0, 1.0, 1.0, 1.0, 1.0]
        self.vgg = Vgg19(requires_grad=False)

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class FocusNetLoss(nn.Module):
    """Loss function for Focus-Net training.

    Combines:
    - Content loss (SSIM + L1)
    - Auxiliary losses at intermediate scales
    - Sobel edge loss
    - Total variation loss
    """
    def __init__(self, alpha=0.11, content_weight=10, aux_weight=1,
                 sobel_weight=200, tv_weight=0.0001):
        super(FocusNetLoss, self).__init__()
        self.alpha = alpha
        self.content_weight = content_weight
        self.aux_weight = aux_weight
        self.sobel_weight = sobel_weight
        self.tv_weight = tv_weight
        self.l1_loss = nn.L1Loss()

    def ssim_loss(self, pred, target, window_size=11):
        """Compute 1 - SSIM loss."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        window = create_window(window_size, pred.shape[1]).to(pred.device).type_as(pred)
        mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=pred.shape[1])
        mu2 = F.conv2d(target, window, padding=window_size // 2, groups=target.shape[1])
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu12 = mu1 * mu2
        sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2,
                             groups=pred.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2,
                             groups=target.shape[1]) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=window_size // 2,
                           groups=pred.shape[1]) - mu12
        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()

    def sobel_loss(self, pred, target):
        """Sobel edge loss."""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        pred_x = F.conv2d(pred, sobel_x, padding=1)
        pred_y = F.conv2d(pred, sobel_y, padding=1)
        target_x = F.conv2d(target, sobel_x, padding=1)
        target_y = F.conv2d(target, sobel_y, padding=1)
        return self.l1_loss(pred_x, target_x) + self.l1_loss(pred_y, target_y)

    def tv_loss(self, pred):
        """Total variation loss."""
        diff_h = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        diff_w = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return (diff_h.abs().mean() + diff_w.abs().mean())

    def forward(self, outputs, labels, aux1, aux2, aux3, labels_aux1, labels_aux2, labels_aux3):
        """
        Args:
            outputs: Main output [B, 1, H, W]
            labels: Ground truth [B, 1, H, W]
            aux1, aux2, aux3: Auxiliary outputs at different scales
            labels_aux1, labels_aux2, labels_aux3: Downsampled GT for auxiliary outputs
        """
        # Content loss
        l1_loss = self.l1_loss(outputs, labels)
        ssim_loss = self.ssim_loss(outputs, labels)
        content_loss = self.alpha * ssim_loss / 2 + (1 - self.alpha) * l1_loss

        # Auxiliary losses
        loss_aux1 = self.l1_loss(aux1, labels_aux1)
        loss_aux2 = self.l1_loss(aux2, labels_aux2)
        loss_aux3 = self.l1_loss(aux3, labels_aux3)
        aux_loss = loss_aux1 + loss_aux2 + loss_aux3

        # Sobel edge loss
        sobel_loss = self.sobel_loss(outputs, labels)

        # TV loss
        tv = self.tv_loss(outputs)

        total_loss = (self.content_weight * content_loss +
                      self.aux_weight * aux_loss +
                      self.sobel_weight * sobel_loss +
                      self.tv_weight * tv)

        return total_loss, content_loss, aux_loss, sobel_loss, tv


class FusionNetLoss(nn.Module):
    """Loss function for Fusion-Net training.

    Combines:
    - Content loss (L1 + MEF-SSIM)
    - Perceptual loss (VGG19 + TV)
    - Color loss (angle-based)
    """
    def __init__(self, content_weight=10, color_weight=1.1, perceptual_weight=1,
                 mefssim_weight=0.1, tv_weight=0.0001):
        super(FusionNetLoss, self).__init__()
        self.content_weight = content_weight
        self.color_weight = color_weight
        self.perceptual_weight = perceptual_weight
        self.mefssim_weight = mefssim_weight
        self.tv_weight = tv_weight
        self.l1_loss = nn.L1Loss()
        self.mefssim = MEFSSIM()
        self.vgg_loss = VGGLoss()

    @staticmethod
    def angle_loss(a, b):
        """Angle-based color loss."""
        up = torch.sum(a * b)
        norm_a = torch.norm(a, p=2)
        norm_b = torch.norm(b, p=2)
        down = norm_a * norm_b
        cos_theta = torch.clamp(up / (down + 1e-8), -1, 1.0)
        theta = torch.acos(cos_theta)
        return theta

    def tv_loss(self, pred):
        diff_h = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        diff_w = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return (diff_h.abs().mean() + diff_w.abs().mean())

    def forward(self, outputs, targets):
        # Content loss
        l1_loss = self.l1_loss(outputs, targets)
        mefssim_loss = 1 - self.mefssim(outputs, targets)
        content_loss = l1_loss + self.mefssim_weight * mefssim_loss

        # Perceptual loss
        tv = self.tv_loss(outputs)
        vgg = self.vgg_loss(outputs, targets)
        perceptual_loss = self.tv_weight * tv + vgg

        # Color loss
        color_loss = (self.angle_loss(outputs[:, 0], targets[:, 0]) +
                      self.angle_loss(outputs[:, 1], targets[:, 1]) +
                      self.angle_loss(outputs[:, 2], targets[:, 2]))

        total_loss = (self.content_weight * content_loss +
                      self.perceptual_weight * perceptual_loss +
                      self.color_weight * color_loss)

        return total_loss, content_loss, perceptual_loss, color_loss
