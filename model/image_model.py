from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(
                half, device=t.device, dtype=torch.float32
            ) / max(half - 1, 1)
        )
        emb = t.float()[:, None] * freq[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, cond_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(time_dim, out_ch)
        self.cond = nn.Linear(cond_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, c_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(t_emb)[:, :, None, None] + self.cond(c_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class XENImageModel(nn.Module):
    """Small conditional diffusion U-Net for XEN-GEN1-I."""

    def __init__(self, cond_dim=256, base=64):
        super().__init__()
        time_dim = base * 4
        self.time = SinusoidalTimeEmbedding(time_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.in_conv = nn.Conv2d(3, base, 3, padding=1)
        self.down1 = ResBlock(base, base, time_dim, cond_dim)
        self.down2 = ResBlock(base, base * 2, time_dim, cond_dim)
        self.down3 = ResBlock(base * 2, base * 4, time_dim, cond_dim)
        self.mid1 = ResBlock(base * 4, base * 4, time_dim, cond_dim)
        self.mid2 = ResBlock(base * 4, base * 4, time_dim, cond_dim)
        self.up3 = ResBlock(base * 8, base * 2, time_dim, cond_dim)
        self.up2 = ResBlock(base * 4, base, time_dim, cond_dim)
        self.up1 = ResBlock(base * 2, base, time_dim, cond_dim)
        self.out = nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, x, timestep, text_embedding):
        c = self.text_proj(text_embedding)
        t = self.time(timestep)

        x0 = self.in_conv(x)
        d1 = self.down1(x0, t, c)
        d2 = self.down2(F.avg_pool2d(d1, 2), t, c)
        d3 = self.down3(F.avg_pool2d(d2, 2), t, c)

        m = self.mid1(d3, t, c)
        m = self.mid2(m, t, c)

        u3 = F.interpolate(m, size=d2.shape[-2:], mode="nearest")
        u3 = self.up3(torch.cat([u3, d2], dim=1), t, c)
        u2 = F.interpolate(u3, size=d1.shape[-2:], mode="nearest")
        u2 = self.up2(torch.cat([u2, d1], dim=1), t, c)
        u1 = self.up1(torch.cat([u2, x0], dim=1), t, c)
        return self.out(F.silu(u1))
