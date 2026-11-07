"""Whole-block vs per-segment backward peaks under a simulated 16 GB limit."""
import sys
import torch

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sparc3d.nn.sparse import SparseTensor
from sparc3d.nn.sparse.transformer import ModulatedSparseTransformerCrossBlock as Block

DEV = "cuda:0"
GiB = 2**30
D = 2112
LIMIT = 15.5  # usable GiB assumed on a headless 16 GB card
torch.cuda.set_per_process_memory_fraction(LIMIT * GiB / torch.cuda.get_device_properties(0).total_memory, 0)

g = torch.arange(256, dtype=torch.float32) - 128
x_, y_, z_ = torch.meshgrid(g, g, g, indexing="ij")
d = (x_ * x_ + y_ * y_ + z_ * z_).sqrt() - 256 * 0.39
t = 1.0
while (d.abs() < t / 2).sum() < 185365:
    t *= 1.05
base = torch.unique((d.abs() < t / 2).nonzero().int() // 2, dim=0)
del x_, y_, z_, d, g
print("base tokens", base.shape[0], flush=True)

from sparc3d.nn.utils import convert_module_to_f16
blk = Block(D, 2048, 22, mlp_ratio=4.0)
blk.apply(convert_module_to_f16)
blk.to(DEV)
mod = torch.randn(1, 6 * D, device=DEV, dtype=torch.float16)
ctx = torch.randn(1, 1374, 2048, device=DEV, dtype=torch.float16)


def coords_for(n):
    c = base.to(DEV)
    layers = [c]
    k = 1
    while sum(len(l) for l in layers) < n:
        layers.append(c + torch.tensor([0, 0, 1000 * k], device=DEV))
        k += 1
    return torch.cat(layers)[:n]


def sp(f, c):
    b = torch.zeros((c.shape[0], 1), dtype=torch.int32, device=DEV)
    return SparseTensor(feats=f, coords=torch.cat((b, c), 1))


def segments():
    """Split the block's forward into self-attn, cross-attn, and MLP."""
    def s1(f, c):
        x = sp(f, c)
        sh, sc, ga = (blk.modulation + mod).chunk(6, dim=1)[:3]
        h = x.replace(blk.norm1(x.feats))
        h = h * (1 + sc) + sh
        return (x + blk.self_attn(h) * ga).feats

    def s2(f, c):
        x = sp(f, c)
        return (x + blk.cross_attn(x.replace(blk.norm2(x.feats)), ctx)).feats

    def s3(f, c):
        x = sp(f, c)
        sh, sc, ga = (blk.modulation + mod).chunk(6, dim=1)[3:]
        h = x.replace(blk.norm3(x.feats))
        h = h * (1 + sc) + sh
        return (x + blk.mlp(h) * ga).feats

    return [s1, s2, s3]


def stats(b):
    return (torch.cuda.max_memory_allocated() - b) / GiB, torch.cuda.max_memory_reserved() / GiB


def whole(n):
    c = coords_for(n)
    f = torch.randn(n, D, device=DEV, dtype=torch.float16, requires_grad=True)
    blk.zero_grad(set_to_none=True)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    b = torch.cuda.memory_allocated()
    out = blk(sp(f, c), mod, ctx)
    (out.feats.float() ** 2).mean().backward()
    return stats(b)


def seg(n):
    c = coords_for(n)
    f = torch.randn(n, D, device=DEV, dtype=torch.float16)
    blk.zero_grad(set_to_none=True)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    b = torch.cuda.memory_allocated()
    ss = segments()
    inputs = [f]
    with torch.no_grad():
        for s in ss:
            inputs.append(s(inputs[-1], c))
    g_out = 2 * inputs[-1].float().to(torch.float16) / n
    inputs.pop()
    for s in reversed(ss):
        i = inputs.pop().detach().requires_grad_(True)
        o = s(i, c)
        o.backward(g_out)
        g_out = i.grad.detach()
        del o, i
    return stats(b)


for name, fn in [("whole-block", whole), ("per-segment", seg)]:
    for n in [47765, 60000, 70000, 80000, 90000, 100000, 120000, 140000, 160000]:
        try:
            a, r = fn(n)
            print(f"{name} T={n}: peak_alloc={a:.2f} peak_reserved={r:.2f} GiB OK", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"{name} T={n}: OOM at {LIMIT} GiB", flush=True)
            torch.cuda.empty_cache()
            break
print("done", flush=True)
