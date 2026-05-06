import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvPositionalEmbedding(nn.Module):
    """Convolutional Positional Embedding (CPE).

    Injects absolute positional information via 3x3 depth-wise convolution with zero padding.
    """
    def __init__(self, dim, kernel_size=3):
        super(ConvPositionalEmbedding, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1,
                                padding=kernel_size // 2, padding_mode='zeros',
                                groups=dim)

    def forward(self, x):
        return self.dwconv(x)


class SuperTokenExtraction(nn.Module):
    """Super Token Extraction (STE).

    Soft K-means based algorithm adapted from pixel space to token space.
    Iteratively computes mapping matrix Q (soft assignment) and super tokens S.
    """
    def __init__(self, grid_size=(2, 2), num_iters=3):
        super(SuperTokenExtraction, self).__init__()
        self.grid_size = grid_size
        self.num_iters = num_iters

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            S: Super tokens [B, m, C]
            Q: Mapping matrix [B, N, m]
        """
        B, C, H, W = x.shape
        N = H * W
        h, w = self.grid_size
        m = (H // h) * (W // w)
        d = C

        # Reshape to tokens [B, N, C]
        X = x.view(B, C, N).permute(0, 2, 1)  # [B, N, C]

        # Initialize super tokens by averaging within regular grid regions
        S = x.view(B, C, H // h, h, W // w, w)
        S = S.mean(dim=(3, 5))  # [B, C, H//h, W//w]
        S = S.view(B, C, -1).permute(0, 2, 1)  # [B, m, C]

        for t in range(self.num_iters):
            # Token correlation: compute soft assignment matrix Q
            Q = torch.bmm(X, S.transpose(1, 2)) / (d ** 0.5)  # [B, N, m]
            Q = F.softmax(Q, dim=-1)

            # Normalize columns of Q
            Q_norm = Q / (Q.sum(dim=1, keepdim=True) + 1e-6)

            # Super token update: weighted sum
            S = torch.bmm(Q_norm.transpose(1, 2), X)  # [B, m, C]

        return S, Q


class TokenUpsampling(nn.Module):
    """Token Upsampling (TU).

    Reallocates information from super tokens back to original visual token resolution
    using the mapping matrix Q.
    TU(Attn(S)) = Attn(S) * Q^T
    """
    def __init__(self):
        super(TokenUpsampling, self).__init__()

    def forward(self, attn_S, Q):
        """
        Args:
            attn_S: Attention-enhanced super tokens [B, m, C]
            Q: Mapping matrix [B, N, m]
        Returns:
            Reconstructed visual tokens [B, N, C]
        """
        return torch.bmm(Q, attn_S)  # [B, N, m] @ [B, m, C] = [B, N, C]


class STAModule(nn.Module):
    """Super Token Attention (STA) Module.

    Four components: CPE, STE, MHSA, TU.
    Residual block structure:
        X = CPE(Input) + Input
        Output = STA(LN(X)) + X
    """
    def __init__(self, dim, num_heads=8, grid_size=(2, 2), ste_iters=3, mlp_ratio=4.0):
        super(STAModule, self).__init__()
        self.dim = dim
        self.cpe = ConvPositionalEmbedding(dim)
        self.ln = nn.LayerNorm(dim)
        self.ste = SuperTokenExtraction(grid_size=grid_size, num_iters=ste_iters)
        self.tu = TokenUpsampling()

        # MHSA on super tokens
        self.mhsa = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # FFN after attention
        mlp_hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )
        self.ln_ffn = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Output tensor [B, C, H, W]
        """
        B, C, H, W = x.shape
        N = H * W

        # CPE + residual
        x = x + self.cpe(x)

        # Reshape for token operations
        X = x.view(B, C, N).permute(0, 2, 1)  # [B, N, C]

        # Layer norm
        X_norm = self.ln(X)

        # STE: extract super tokens
        S, Q = self.ste(x)  # S: [B, m, C], Q: [B, N, m]

        # MHSA on super tokens
        S_norm = self.ln(S)
        attn_out, _ = self.mhsa(S_norm, S_norm, S_norm)
        attn_out = attn_out + S  # residual on super tokens

        # TU: project back to visual tokens
        X_reconstructed = self.tu(attn_out, Q)  # [B, N, C]

        # Residual connection
        X = X + X_reconstructed

        # FFN with residual
        X = X + self.ffn(self.ln_ffn(X))

        # Reshape back to spatial
        out = X.permute(0, 2, 1).view(B, C, H, W)

        return out
