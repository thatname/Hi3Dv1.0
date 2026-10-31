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


def encoder_config(backend: str = "spconv") -> SimpleNamespace:
    """Explicit factor-8/latent-32 extension of vae3d T1's encoder contract."""
    return SimpleNamespace(
        widths=(32, 64, 128, 256), latent_channels=32, factor=8,
        backend=backend, blocks_per_level=2, attention_shifts=(0, 2),
        encoder_kernel_size=3, encoder_dilations=(1,), checkpoint_blocks=True,
        max_sites=2_000_000, sparsity_mode="full-band", halo=0,
    )


class G1Encoder(nn.Module):
    """Adapt vae3d's four-channel sparse Encoder to factor 8 and latent 32.

    Derived from vae3d/model.py at 422e79988ccda06ef0995bf7768cd05a8894a124.
    The G1 full-band support has no requested latent halo or checkpoint reuse.
    """

    def __init__(self):
        super().__init__()
        cfg = encoder_config()
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
                support_dilation: int = 0):
        """Decode a latent into per-corner field and deformation.

        support_dilation only touches the self-predicted support and only the
        finest level: 1 keeps every sibling of a selected cell (its parent's whole
        2x2x2 block), 2 also includes children of face-adjacent parents. Either
        way the added cells are produced by the decoder, never invented, and 0
        preserves the trained behaviour exactly.
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
                    flat = torch.nonzero(logits > 0, as_tuple=True)[0]
                    if support_dilation:
                        if not 0 <= support_dilation <= 2:
                            raise ValueError("support_dilation must be 0, 1 or 2")
                        # Applied at every upsampling level so the extra margin
                        # propagates to the finer grids; the last level alone cannot
                        # grow past the parent set it inherited.
                        parents = torch.unique(torch.div(flat, 8, rounding_mode="floor"))
                        if support_dilation == 2:
                            parents = _dilate_parents(parents, point.grid_coord, point.resolution)
                        flat = (parents[:, None] * 8
                                + torch.arange(8, device=parents.device)).reshape(-1)
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


def _dilate_parents(parents: torch.Tensor, grid: torch.Tensor, resolution: int) -> torch.Tensor:
    """Rows of the face-adjacent parents of the given parent rows, within the grid."""
    coords = grid[parents.long()]
    offsets = torch.tensor(((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                            (0, 0, 1), (0, 0, -1)), device=coords.device, dtype=coords.dtype)
    neighbours = (coords[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    inside = ((neighbours >= 0) & (neighbours < int(resolution))).all(-1)
    keys = _keys(torch.cat((parents.new_zeros((len(parents), 1)), coords), 1), int(resolution))
    neighbour_keys = _keys(torch.cat((parents.new_zeros((len(neighbours), 1)), neighbours), 1),
                           int(resolution))
    known, order = keys.sort()
    position = torch.searchsorted(known, neighbour_keys).clamp(max=len(known) - 1)
    existing = inside & (known[position] == neighbour_keys)
    return torch.unique(torch.cat((parents.long(), order[position][existing])))


class G1VAE(nn.Module):
    def __init__(self, backbone: PTv3MeshDecoder):
        super().__init__()
        self.encoder = G1Encoder()
        self.decoder = TrainableHi3DDecoder(backbone)

    def forward(self, unique_coords, features, fine_cells, resolution: int,
                *, predicted_support: bool = False, max_sites: int = 2_000_000,
                support_dilation: int = 0):
        x = Sparse(torch.cat((torch.zeros((len(unique_coords), 1), device=unique_coords.device,
                                          dtype=torch.int32), unique_coords.int()), 1),
                   features, resolution, sparsity_mode="full-band")
        latent, mean, logvar = self.encoder(x, fine_cells)
        cells, signed, delta, occupancy = self.decoder(
            latent, fine_cells, resolution, predicted_support=predicted_support,
            max_sites=max_sites, support_dilation=support_dilation)
        return cells, signed, delta, occupancy, mean, logvar
