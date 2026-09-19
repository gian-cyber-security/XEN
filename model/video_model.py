from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        emb = t.float()[:, None] * freq[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class VideoResBlock(nn.Module):
    """Compact factorized 3D residual block.

    Spatial and temporal convolutions are separated to reduce parameters and
    memory compared with a full 3D convolution.
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, cond_dim: int):
        super().__init__()
        groups_in = 8 if in_ch >= 8 else 1
        groups_out = 8 if out_ch >= 8 else 1
        self.norm1 = nn.GroupNorm(groups_in, in_ch)
        self.spatial1 = nn.Conv3d(in_ch, out_ch, (1, 3, 3), padding=(0, 1, 1))
        self.temporal1 = nn.Conv3d(out_ch, out_ch, (3, 1, 1), padding=(1, 0, 0))
        self.norm2 = nn.GroupNorm(groups_out, out_ch)
        self.spatial2 = nn.Conv3d(out_ch, out_ch, (1, 3, 3), padding=(0, 1, 1))
        self.temporal2 = nn.Conv3d(out_ch, out_ch, (3, 1, 1), padding=(1, 0, 0))
        self.time = nn.Linear(time_dim, out_ch)
        self.cond = nn.Linear(cond_dim, out_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.spatial1(h)
        h = self.temporal1(h)
        h = h + self.time(t_emb)[:, :, None, None, None]
        h = h + self.cond(c_emb)[:, :, None, None, None]
        h = self.spatial2(F.silu(self.norm2(h)))
        h = self.temporal2(h)
        return h + self.skip(x)


class XENVideoModel(nn.Module):
    """Small from-scratch conditional video diffusion U-Net.

    Input/output shape: [batch, 3, frames, height, width].
    The temporal dimension is preserved, so the same checkpoint can be used
    with different clip lengths as memory allows.
    """

    def __init__(self, cond_dim: int = 384, base: int = 32):
        super().__init__()
        time_dim = base * 4
        self.time = SinusoidalTimeEmbedding(time_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.in_conv = nn.Conv3d(3, base, 3, padding=1)
        self.down1 = VideoResBlock(base, base, time_dim, cond_dim)
        self.down2 = VideoResBlock(base, base * 2, time_dim, cond_dim)
        self.down3 = VideoResBlock(base * 2, base * 4, time_dim, cond_dim)
        self.mid1 = VideoResBlock(base * 4, base * 4, time_dim, cond_dim)
        self.mid2 = VideoResBlock(base * 4, base * 4, time_dim, cond_dim)
        self.up3 = VideoResBlock(base * 8, base * 2, time_dim, cond_dim)
        self.up2 = VideoResBlock(base * 4, base, time_dim, cond_dim)
        self.up1 = VideoResBlock(base * 2, base, time_dim, cond_dim)
        self.out = nn.Conv3d(base, 3, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        c = self.cond_proj(text_embedding)
        t = self.time(timestep)

        x0 = self.in_conv(x)
        d1 = self.down1(x0, t, c)
        d2 = self.down2(F.avg_pool3d(d1, (1, 2, 2)), t, c)
        d3 = self.down3(F.avg_pool3d(d2, (1, 2, 2)), t, c)

        m = self.mid1(d3, t, c)
        m = self.mid2(m, t, c)

        u3 = F.interpolate(m, size=d2.shape[-3:], mode="trilinear", align_corners=False)
        u3 = self.up3(torch.cat([u3, d2], dim=1), t, c)
        u2 = F.interpolate(u3, size=d1.shape[-3:], mode="trilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, d1], dim=1), t, c)
        u1 = self.up1(torch.cat([u2, x0], dim=1), t, c)
        return self.out(F.silu(u1))
