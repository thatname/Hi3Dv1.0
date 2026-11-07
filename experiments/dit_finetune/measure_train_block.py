"""Measure VRAM of Hi3D DiT training pieces on a synthetic 256^3 surface shell."""
import json, sys, time
import torch

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from safetensors.torch import load_file
from sparc3d.models.transformers.sparc3d_transformer import Sparc3DTransformer
from sparc3d.nn.sparse import SparseTensor

CK = str(ROOT / "ckpt" / "Hi3Dv1.0" / "transformer_1024")
DEV = "cuda:0"
GiB = 2**30
L_COND = 1374


def shell(n_target, res=256):
    r, t = res * 0.39, 1.0
    g = torch.arange(res, dtype=torch.float32) - res / 2
    x, y, z = torch.meshgrid(g, g, g, indexing="ij")
    d = (x * x + y * y + z * z).sqrt() - r
    while True:
        m = d.abs() < t / 2
        if m.sum() >= n_target:
            break
        t *= 1.05
    c = m.nonzero().int()
    return c


def sp(feats, c):
    b = torch.zeros((c.shape[0], 1), dtype=torch.int32, device=c.device)
    return SparseTensor(feats=feats, coords=torch.cat((b, c), 1))


def reset():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated()


def rep(tag, base):
    print(f"{tag}: peak_over_base={(torch.cuda.max_memory_allocated()-base)/GiB:.3f} GiB "
          f"now_over_base={(torch.cuda.memory_allocated()-base)/GiB:.3f}", flush=True)


cfg = json.load(open(Path(CK) / "config.json"))
m = Sparc3DTransformer(**cfg)
print("load", m.load_state_dict(load_file(str(Path(CK) / "model.safetensors")), strict=False), flush=True)
total = sum(p.numel() for p in m.parameters())
bp = sum(p.numel() for p in m.blocks[0].parameters())
print(f"params total={total/1e9:.3f}B block={bp/1e6:.1f}M", flush=True)
m.to(DEV)
print(f"resident={torch.cuda.memory_allocated()/GiB:.3f} GiB", flush=True)

coords = shell(185365).to(DEV)
N = coords.shape[0]
feats = torch.randn(N, 32, device=DEV)
cond = torch.randn(1, L_COND, 2048, device=DEV)
ts = torch.tensor([500.0], device=DEV)
print(f"voxels={N}", flush=True)

# 1. inference calibration against the measured 11.95 GiB run
b = reset()
t0 = time.time()
with torch.no_grad():
    m(sp(feats, coords), ts, cond)
rep(f"no_grad full forward ({time.time()-t0:.0f}s)", b)

# 2. input path with grad, to obtain real tokens
b = reset()
h = m.input_layer(sp(feats, coords)).type(m.dtype)
emb = m.t_embedder(ts).to(m.dtype)
skips = []
for blk in m.input_blocks:
    h = blk(h, emb)
    skips.append(h.feats)
rep("input path forward with grad", b)
T = h.feats.shape[0]
print(f"tokens={T}", flush=True)
real = h.coords[:, 1:].detach()
scale = h._scale if hasattr(h, "_scale") else None
del h, skips, emb
torch.cuda.empty_cache()

# 3. one transformer block forward+backward at several token counts
mod = torch.randn(1, 6 * cfg["model_channels"], device=DEV, dtype=m.dtype)
ctx = cond.to(m.dtype)
blk = m.blocks[0]
for frac in [1.0, 1.5, 2.0]:
    n = int(T * frac)
    if frac <= 1:
        c = real[:n]
    else:
        extra = real[torch.randint(0, T, (n - T,), device=DEV)] + torch.tensor([0, 0, 1], device=DEV) * 2000
        c = torch.cat((real, extra))
        c = torch.unique(c, dim=0)
        n = c.shape[0]
    x = sp(torch.randn(n, cfg["model_channels"], device=DEV, dtype=m.dtype), c)
    x_size = x.feats.numel() * 2 / GiB
    m.zero_grad(set_to_none=True)
    b = reset()
    t0 = time.time()
    out = blk(x, mod, ctx)
    rep(f"T={n} block forward", b)
    loss = (out.feats.float() ** 2).mean()
    loss.backward()
    rep(f"T={n} block fwd+bwd ({time.time()-t0:.1f}s) one_x={x_size:.3f} GiB", b)
    del x, out, loss
    torch.cuda.empty_cache()
print("done", flush=True)
