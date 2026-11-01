"""G1 training bridge: vae3d-style encoder and Hi3D PTv3 backbone.

The existing inference decoder is left untouched. Its discrete occupancy prune is
replaced here by supervised child selection while training; each occupancy head
still receives BCE gradients. Only this module owns the new continuous head.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from vae3d.sparse import Sparse, Residual, Downsample, WindowAttention, lookup

from sparc3d.models.autoencoders.blocks import PointGroupNorm, Swish
from sparc3d.models.autoencoders.ptv3 import PTv3MeshDecoder
from sparc3d.ops import sparse_backend
from sparc3d.structures.point import Point
from sparc3d.structures.point_modules import PointSequential


ENCODER_LADDER = {
    # Widening only: the latent width stays 32 because the decoder is loaded
    # strictly from the released Hi3D weights and expects that interface.
    "E0": SimpleNamespace(widths=(32, 64, 128, 256), blocks_per_level=2),
    "E1": SimpleNamespace(widths=(64, 128, 256, 512), blocks_per_level=2),
    "E2": SimpleNamespace(widths=(96, 192, 384, 768), blocks_per_level=3),
    "E3": SimpleNamespace(widths=(128, 256, 512, 1024), blocks_per_level=2),
    # E4 matches the 94 M decoder: same depth, widths scaled by 1.5 (params scale ~k^2).
    "E4": SimpleNamespace(widths=(192, 384, 768, 1536), blocks_per_level=2),
}


def encoder_config(backend: str = "spconv", scale: str = "E0") -> SimpleNamespace:
    """Explicit factor-8/latent-32 extension of vae3d T1's encoder contract."""
    ladder = ENCODER_LADDER[scale]
    return SimpleNamespace(
        widths=ladder.widths, latent_channels=32, factor=8,
        backend=backend, blocks_per_level=ladder.blocks_per_level,
        attention_shifts=(0, 2),
        encoder_kernel_size=3, encoder_dilations=(1,), checkpoint_blocks=True,
        max_sites=2_000_000, sparsity_mode="full-band", halo=0, scale=scale,
    )


class G1Encoder(nn.Module):
    """Adapt vae3d's four-channel sparse Encoder to factor 8 and latent 32.

    Derived from vae3d/model.py at 422e79988ccda06ef0995bf7768cd05a8894a124.
    The G1 full-band support has no requested latent halo or checkpoint reuse.
    """

    def __init__(self, scale: str = "E0"):
        super().__init__()
        cfg = encoder_config(scale=scale)
        self.input = nn.Linear(4, cfg.widths[0])
        self.blocks = nn.ModuleList([
            nn.ModuleList([Residual(c, cfg.backend) for _ in range(cfg.blocks_per_level)])
            for c in cfg.widths
        ])
        self.down = nn.ModuleList([Downsample(a, b) for a, b in zip(cfg.widths, cfg.widths[1:])])
        self.attn = nn.ModuleList([WindowAttention(cfg.widths[-1], shift=s)
                                   for s in cfg.attention_shifts])
        self.mean = nn.Linear(cfg.widths[-1], cfg.latent_channels)
        self.logvar = nn.Linear(cfg.widths[-1], cfg.latent_channels)

    def forward(self, x: Sparse, requested_cells: torch.Tensor):
        if x.feats.shape[1] != 4 or x.resolution % 8 or not len(x.coords):
            raise ValueError("G1 encoder expects nonempty four-channel R divisible by 8")
        x = x.replace(self.input(x.feats))
        for level, blocks in enumerate(self.blocks):
            for block in blocks:
                x = _run_encoder_block(block, x, self.training)
            if level < len(self.down):
                x = self.down[level](x)
        for block in self.attn:
            x = _run_encoder_block(block, x, self.training)
        requested = torch.unique(requested_cells // 8, dim=0).int()
        query = torch.cat((torch.zeros((len(requested), 1), device=requested.device,
                                        dtype=torch.int32), requested), dim=1)
        rows = lookup(x.coords, query, x.resolution)
        if bool((rows < 0).any()):
            raise ValueError("G1 encoder lacks requested decoder latent support")
        x = Sparse(query, x.feats[rows], x.resolution)
        mean = self.mean(x.feats)
        logvar = self.logvar(x.feats).clamp(-12, 8)
        return x.replace(mean), mean, logvar


def _run_encoder_block(block, x, training):
    if training and torch.is_grad_enabled():
        return x.replace(checkpoint(lambda feats: block(x.replace(feats)).feats,
                                    x.feats, use_reentrant=False))
    return block(x)


def _keys(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    c = coords.long()
    return (c[:, 0] * resolution + c[:, 1]) * resolution + c[:, 2]


def _rows(source: torch.Tensor, query: torch.Tensor, resolution: int) -> torch.Tensor:
    keys, order = _keys(source, resolution).sort()
    pos = torch.searchsorted(keys, _keys(query, resolution)).clamp(max=len(keys) - 1)
    if not torch.equal(keys[pos], _keys(query, resolution)):
        raise ValueError("G1 target support is not reachable from parent support")
    return order[pos]


class TrainableHi3DDecoder(nn.Module):
    """Keep pretrained PTv3 modules, with GT child support and a fresh 40-channel head."""

    def __init__(self, backbone: PTv3MeshDecoder):
        super().__init__()
        self.backbone = backbone
        self.head = PointSequential(
            PointGroupNorm(backbone.conv_out.conv._modules["0"].channels),
            Swish(),
            sparse_backend.Conv3d(
                backbone.conv_out.conv._modules["0"].channels, 40,
                kernel_size=3, stride=1, padding=1, bias=True,
            ),
        )

    def forward(self, latent, fine_cells: torch.Tensor, resolution: int,
                *, predicted_support: bool = False, max_sites: int = 2_000_000,
                support_threshold: float = 0.0):
        """Decode a latent into per-corner field and deformation.

        support_threshold only affects the self-predicted support: a child cell
        is kept when its occupancy logit exceeds this value. Hi3D's inference
        path hard-codes the equivalent of 0.0, and 0.0 reproduces that behaviour
        exactly. The threshold is an inference operating point, never used in
        training, and must be reported alongside any metric it produced.
        """
        if resolution % 8 or len(fine_cells) == 0:
            raise ValueError("G1 requires nonempty cells at a multiple-of-8 resolution")
        levels = [torch.unique(fine_cells // (2 ** power), dim=0)
                  for power in (3, 2, 1, 0)]
        latent_grid = latent.coords[:, 1:].int()
        if not torch.equal(torch.sort(_keys(latent_grid, resolution // 8)).values,
                           torch.sort(_keys(levels[0], resolution // 8)).values):
            raise ValueError("Encoder and decoder latent supports differ")
        point = Point({
            "grid_coord": latent_grid, "feat": latent.feats, "resolution": resolution // 8,
            "offset": torch.tensor([len(latent_grid)], device=latent.feats.device),
            "sdf": None, "deform": None,
        })
        point.initialization()
        point.sparsify()
        point = self.backbone.z_to_f(point)
        point.serialization(order=self.backbone.order)
        point = self.backbone.mid(point)
        occupancy_losses = []
        for stage in self.backbone.dec._modules.values():
            for module in stage._modules.values():
                if not hasattr(module, "upsample"):
                    point = module(point)
                    continue
                unpool = module.upsample
                logits = unpool.cls_head(point).feat.reshape(-1)
                if predicted_support:
                    flat = torch.nonzero(logits > support_threshold, as_tuple=True)[0]
                    if len(flat) > max_sites:
                        raise RuntimeError("Predicted child support exceeds max_sites")
                    if len(flat) == 0:
                        empty = point.feat.new_empty((0, 8))
                        return (point.grid_coord.new_empty((0, 3)), empty,
                                point.feat.new_empty((0, 8, 3)), occupancy_losses)
                    parent_rows = torch.div(flat, 8, rounding_mode="floor")
                    child = point.grid_coord[parent_rows] * 2 + unpool.inv_grid_coord[flat % 8]
                else:
                    child = levels[len(occupancy_losses) + 1]
                    parent_rows = _rows(point.grid_coord, child // 2, point.resolution)
                    parity = child.long() % 2
                    slot = parity[:, 0] * 4 + parity[:, 1] * 2 + parity[:, 2]
                    flat = parent_rows * 8 + slot
                    labels = point.feat.new_zeros((len(point.feat) * 8,))
                    labels[flat] = 1
                    occupancy_losses.append(F.binary_cross_entropy_with_logits(logits, labels))
                projected = unpool.proj(point.feat).reshape(-1, point.feat.shape[1])
                feat = projected[flat]
                point = point.clone(
                    feat=feat, grid_coord=child.int(), resolution=point.resolution * 2,
                    batch=torch.zeros(len(child), device=child.device, dtype=torch.long),
                    offset=torch.tensor([len(child)], device=child.device),
                )
                point.sparsify()
        if point.feat.shape[0] > 12_000_000:
            raise RuntimeError("Training would silently skip a residual block")
        point = self.head(point)
        raw = point.feat.reshape(-1, 8, 5)
        signed = -torch.tanh(raw[..., 0]) * (F.softplus(raw[..., 1]) + 1e-5)
        delta = 0.5 * torch.tanh(raw[..., 2:])
        return point.grid_coord, signed, delta, occupancy_losses


class G1VAE(nn.Module):
    def __init__(self, backbone: PTv3MeshDecoder, encoder_scale: str = "E0"):
        super().__init__()
        self.encoder = G1Encoder(encoder_scale)
        self.decoder = TrainableHi3DDecoder(backbone)

    def forward(self, unique_coords, features, fine_cells, resolution: int,
                *, predicted_support: bool = False, max_sites: int = 2_000_000,
                support_threshold: float = 0.0):
        x = Sparse(torch.cat((torch.zeros((len(unique_coords), 1), device=unique_coords.device,
                                          dtype=torch.int32), unique_coords.int()), 1),
                   features, resolution, sparsity_mode="full-band")
        latent, mean, logvar = self.encoder(x, fine_cells)
        cells, signed, delta, occupancy = self.decoder(
            latent, fine_cells, resolution, predicted_support=predicted_support,
            max_sites=max_sites, support_threshold=support_threshold)
        return cells, signed, delta, occupancy, mean, logvar
