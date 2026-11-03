# G3 encoder training and data pipeline: review fix list

> Date: 2026-11-04. Reviewed at commit `8fb1921` (branch `finetune`).
> Source: code review of `experiments/vae_sdf/` written by codely. **All findings are [源码核查] (source review only)**: nothing here was run or measured. Costs quoted from codely's own report `docs/g3-stage-and-resource-map-2026-11-02.md` are labelled as theirs.
> Audience: the Claude session on the training rig that will implement the fixes. Read `AGENTS.md`, `docs/HANDOFF.md` and `docs/experiment-plan-v0.3.md` first; this file assumes those rules.

## 0. Before touching anything

1. **The G3 target stream may still be running** (`run_g3_stream.py`, ETA codely gave was 11-04 08:35). `run_g3_a4_batch.py` starts a fresh `python -m experiments.vae_sdf.run_four_route_asset` **per asset**, and producers re-import modules on restart. Editing files in the checkout the stream runs from changes its behaviour mid-run. Check `remote_g3_status` / `logs/stream.log` first. If it is still running, do the pipeline work (Part B) in a separate checkout or wait until it finishes.
2. **Cache keys depend on code hashes.** `run_four_route_asset.py` puts `code_sha256` of `run_four_route_asset.py`, `four_route_a3.py`, `four_route_a4.py`, `four_route_normals.py` and `g0b_field_adapter.py` into the hashed config that names the target directory. Any edit to those files makes new targets land in a new `flexicubes_smooth-<key>` directory. Loaders glob `flexicubes_smooth-*`, so they still work, but the pool then mixes code versions. Record the switch point (commit and time) in HANDOFF, and keep A4 maths unchanged in Part B.
3. **GPU whitelist conflict, ask the user.** CODELY.md (2026-11-01 feedback) says only GPUs 0/1/2/3/7 may be used, because 4/5/6 run other people's processes. But `train_g1_ddp.py:37` and `run_four_route_asset.py:18` whitelist 0–7, `run_g3_stream.py:104` defaults to all eight, and the 11-02 resource map shows all 8 H20 cards in use. Get the current rule from the user before launching anything.
4. Follow the existing conventions: new run directory per run, outputs outside preparation directories, `resolved_config.json` records every new parameter, resume checks include every new parameter, label all evidence [源码核查] / [静态检查] / [GPU 实测]. Commit locally only; the user pushes.

## Part A: model and training (`g1_model.py`, `train_g1.py`, `train_g1_ddp.py`, `streaming_targets.py`)

Do A0 first: every other A item would otherwise be written twice.

### A0. Share one training-step implementation

Loss code is duplicated: `train_g1.py:348-375` and `train_g1_ddp.py:212-237`. The two trainers have also drifted apart: DDP has no `--max-skipped` (aborts on the first bad batch), and only single-GPU has `--a4-summary`.

- Move the loss computation (field, delta, support, validity, KL), the gradient-group check and the trace row into a shared function, e.g. a new `experiments/vae_sdf/g1_step.py`.
- Give both trainers one shared argument-builder for the loss and optimiser arguments, so new flags cannot exist in only one of them.
- Decide with the user whether DDP should also skip non-finite batches. If it does, every rank must skip together: all-reduce a flag before `optimizer.step()`, the same way `abort_if_any_rank_fails` already works.

### A1. Make it an actual VAE (sampled latent + KL with warm-up)

**Problem.** Both trainers hard-code `"beta": 0` (`train_g1.py:289,314`, `train_g1_ddp.py:145,176`). The encoder returns `x.replace(mean)` (`g1_model.py:91`), so the decoder sees μ only. `logvar` (`g1_model.py:69,90`) gets no gradient, which is also why DDP needs `find_unused_parameters=True` (`train_g1_ddp.py:130`). The plan only allowed β=0 as the G1 debug stage (`experiment-plan-v0.3.md:202`): "先 β=0 … 之后 KL warm-up，候选 β=1e-6/1e-5/1e-4 仅作起点。训练 sample、主验证用 μ".

**Fix.**
- Encoder: in training, use `z = mean + exp(0.5*logvar) * randn_like(mean)`. In eval (and when a `sample=False` argument is passed), use `mean`. Keep the existing `clamp(-12, 8)`.
- Loss: KL as in plan line 197, `0.5*mean(mu²+exp(logvar)-1-logvar)`, normalised within each asset and then averaged over the batch/ranks. The `kl_diagnostic` already logged (`train_g1.py:410`) is that formula. Make it the loss term.
- New args in both trainers: `--beta` (default `1e-6`), `--kl-warmup-steps` (linear ramp from 0 to `--beta`; suggest default 1000). Optional `--sample-latent/--no-sample-latent` for the β=0 debug mode.
- Record `beta`, `kl_warmup_steps` and `sample_latent` in `resolved_config.json` and in the resume `expected` dicts. Old checkpoints with `"beta": 0` must still resume only when the new run also uses β=0 and no sampling.
- Once `logvar` is used, try `find_unused_parameters=False` in DDP. Keep it only if a smoke run shows no unused-parameter error; the code comment at `train_g1_ddp.py:127-128` says some steps leave a head unused, so verify that rather than assume it.
- Evaluation (`evaluate_g1.py`): main metrics decode from μ. Also report a sampled-z reconstruction and latent μ/σ/KL statistics, plus a posterior-collapse indicator such as the fraction of latent channels with mean KL < 1e-3 (plan line 273).

**Note.** This does not make latents compatible with the old Hi3D DiT. Plan §9 says this explicitly; the decoder is fine-tuned regardless.

### A2. Optimiser: decoder learning rate, warm-up, schedule, clipping

**Problem.** A single AdamW at `lr=1e-4` (`train_g1.py:275`, `train_g1_ddp.py:131`) updates the randomly initialised encoder, the new random head and the pretrained Hi3D decoder together. There is no warm-up, no schedule, no gradient clipping and no AMP. Early random-encoder gradients go straight into the pretrained decoder.

**Fix.**
- Parameter groups: `encoder`, `continuous_head`, `decoder_backbone` (including `cls_head` support heads). Add `--decoder-lr-scale` (suggest 0.1) for the backbone group.
- `--decoder-freeze-steps` (suggest 500–1000): backbone `requires_grad=False`, or LR 0 for that group, for the first N steps. With DDP, LR 0 is simpler, because changing `requires_grad` mid-run breaks DDP's reducer.
- `--warmup-steps` (linear) followed by cosine decay to `--min-lr-ratio` (suggest 0.1) over the total steps.
- `--grad-clip` (suggest 1.0), using `clip_grad_norm_` after `backward()`. Log the pre-clip norm.
- Save and restore a scheduler state in checkpoints (plan line 328 already requires it).
- AMP (bf16) is optional and must be opt-in (`--amp bf16`, default off). Spconv/PTv3 behaviour under autocast was not checked; it needs a GPU comparison of loss and peak memory before anyone relies on it.

### A3. Shared corners: training scores per-cell copies, extraction uses merged values

**Correction (2026-11-04, after Codex review).** The first version of this item said that independent per-cell corner predictions could leave cracks. That is wrong for the G3 route. `evaluate_g1.py:_raw_mesh` builds a `RouteSurface` (`four_route_a3.py:19-33`) on vae3d's `DynamicSurface`, which works per unique sample point (`base.corners`, `base.indices[:, HI3D_FROM_VAE]`), so FlexiCubes gets one value per shared corner and the mesh is welded. Codex states that `DynamicSurface` averages the copies. That fits the structure, but it was not verified here because vae3d is not in the reviewing checkout. **Confirm the reduction (mean vs. first/last) in `vae3d/render_refine.py` before implementing.**

**Problem (what remains).** The head predicts 8 corners × 5 channels per cell (`g1_model.py:120-126,196`), so a sample point gets up to 8 independent predictions.
1. Train/eval mismatch: the field/delta loss scores each copy separately (re-weighted by `1/count`, `train_g1.py:107-109`), while extraction uses the merged value. Where copies disagree, the merged sign/magnitude and delta move the zero surface and the vertices to a place no copy was trained to produce. Together with missing support (A4), this shows up as boundary damage or geometric distortion, not as open seams.
2. Validity is checked on the wrong deltas: `cell_validity` scores each cell's own unmerged deltas, but the extracted geometry uses merged deltas. A cell can pass the validity loss and still be inverted after merging.

**Fix.**
- Merge the per-corner predictions (signed and delta) to unique sample points with **the same reduction the extractor uses** (confirm first; `mean` is expected). The mapping already exists: `hi_indices = indices[:, HI3D_FROM_VAE]`, the per-cell corner → unique-point index built in `load_asset`. Index it with `target_rows`, `torch_scatter` or `index_reduce_(..., "mean")`.
- Compute field/delta loss once per unique point against `features[:, 0]` / `features[:, 1:]` (no `1/count` weight needed).
- Gather the merged deltas back per cell and compute the validity loss on those, so it constrains the geometry that is actually extracted.
- Extraction needs no change if it already merges the same way. Add a [静态检查] test: merged training values == values `DynamicSurface` produces from the same per-cell tensors.
- Gate behind `--corner-merge {none,mean}` (default `mean` for new runs) so G2 results remain reproducible.
- Optional, separate experiment: a per-point output head instead of 8 corners per cell. Not requested; only if the merged variant still shows inconsistency.

### A4. Support (occupancy) head: class imbalance and train/inference mismatch

**Problem.** Training uses ground-truth child support (`teacher_forced_support: True`); inference uses predicted support with threshold 0. The occupancy loss is plain BCE over all 8 children of every parent (`g1_model.py:184`), averaged over levels with equal weight (`train_g1.py:369`). Codely's own records found predicted support "比真实窄一圈" (one ring narrower than the truth), which cuts zero surfaces at the support edge and hurts watertightness (`docs/reports/g2-r256-and-resolution-limits-2026-10-31.md`). Support dilation was tried and reverted (`docs/reports/g2-pretest-status-2026-10-31.md`).

**Fix.** This is the least certain item; treat it as an experiment, not a guaranteed fix.
- Add `--support-pos-weight {none,auto,<float>}`: `auto` uses negatives/positives per level per batch, clamped (e.g. ≤ 10), as `pos_weight` in `binary_cross_entropy_with_logits`.
- Add per-level support precision/recall to the trace (plan line 273 already asks for them).
- Optionally `--support-loss-weight` (default 1.0) to scale the support term against field/delta.
- Do not switch training to predicted support: predicted cells outside the ground-truth support have no targets. If exposure bias remains after weighting, report it and ask the user rather than inventing a scheme.

### A5. Encoder size E4

**Problem.** E4 (`g1_model.py:34`) is 98.9 M parameters, about the size of the 94.3 M decoder, and the G2 R256 result was obtained with E0. E4 is a new arm with no evidence yet.

**Correction (2026-11-04, after Codex review).** The first version called the encoder's task "close to identity". That overstated it. The input is the full 4-channel target `[s, dx, dy, dz]`, so this is field autoencoding with no missing information to infer. But it still passes a 32-channel latent at R/8 and the decoder, so it is real compression, roughly `4·fine_points / (32·latent_sites)`. Both counts are already in the trace (`fine_cells`, `latent_sites`), so report the actual ratio rather than assume it. How much encoder capacity that compression needs is unknown; that is exactly what the E0/E4 comparison has to measure.

**Fix.** No code change beyond keeping `--encoder-scale` explicit. In the run plan, run the E0 arm first (or alongside), and only keep E4 if it beats E0 on the agreed metrics. Say so in HANDOFF. If the user prefers, change the trainers' default back to E0 and require E4 to be passed explicitly.

### A6. No shuffling

**Problem.** `LazyTargets` sorts indices (`streaming_targets.py:58`) and accepts `seed` (`:43`) but never uses it. Training walks the same fixed order every epoch: `step % len(assets)` (`train_g1.py:348`), `(step*world+rank) % len` (`train_g1_ddp.py:212`). If manifest order correlates with category or face count, training drifts.

**Fix.**
- Epoch-wise permutation derived from `(seed, epoch)` only, e.g. `numpy.random.default_rng([seed, epoch]).permutation(n)`, so resume reproduces it without saving extra state. Position `p` maps to `epoch = p // n` and `perm[p % n]`.
- DDP: build the permutation identically on all ranks and slice `[step*world + rank]`, so each epoch still visits every asset once.
- Record `shuffle: "epoch_permutation_v1"` in the config and resume check. Keep the old order available (`--order fixed`) for reproducing G2.

## Part B: GPU/CPU balance (data preparation and loading)

Context from codely's 11-02 snapshot (their numbers, not mine): CPU is the only bottleneck. A1/A2 (CPU) had 43.1 h remaining vs 26.6 h for A3/A4 (GPU). The load average was 122.7 on 128 threads, while GPU utilisation was 35–65%. A4 takes 41 s wall per asset, of which 12.8 s is GPU. **Every CPU-second saved anywhere goes to A1/A2.**

If the current G3 stream has finished by the time you read this, B1–B4 matter for the next pool (or a resolution change). B5 and B6 matter for G3 training now. **Do B5 and B6 first.**

### B5. Training data loading blocks the GPU (do first)

**Problem.** With `--stream-resident 1`, every step calls `load_asset` synchronously on the training thread (`streaming_targets.py`, `__getitem__`). It:
- reads a large npz;
- computes the SHA-256 of the whole `target.npz` (`train_g1.py:118`);
- rebuilds `cell_indices` on CPU (`train_g1.py:102`);
- parses `audit.json`.

The GPU idles meanwhile.

**Fix.**
- Background prefetch: a thread (or a small process pool) that loads the next K assets in the visit order (A6 makes the order known in advance) into pinned CPU tensors. `upload()` then copies with `non_blocking=True`. `--prefetch K` (default 2). Bound memory: R256 assets are up to hundreds of MB in RAM.
- Hash once, not per load: when the pool is frozen at launch, write a sidecar `pool_hashes.json` (or compute lazily the first time an asset is seen and cache it in the run directory), and record that in the config. Alternatively use `(size, mtime)` plus the A4 `audit.json` identity per load, and the full hash only in the frozen sidecar.
- Consider having the compaction daemon (`compact_g3_targets.py`) store `cell_indices` again if rebuilding it turns out to be a measurable cost. Measure first.
- Acceptance: [GPU 实测] step time and GPU utilisation before/after on the same 50 steps.

### B6. DDP waits on the largest asset

**Problem.** Asset sizes vary 4–7× in fine-cell count (codely's R128→R256 measurements). Each DDP step waits for the slowest rank.

**Fix.** Within each epoch permutation (A6), group assets into chunks of `world` with similar `support_cells` (available in A4 `resolved_config.json`/`audit.json`, so no npz load is needed). Shuffle the chunk order, not the members. Keep it deterministic from `(seed, epoch)`. Flag: `--balance-by-size`. Report per-step max/min rank time in the trace.

### B1. A1/A2 run on CPU only

**Problem.** Producers hard-code `raycast_backend="cpu", distance_backend="cpu"` (`run_g3_producer_worker.py:136`). `initial_fields.py` / `prepare_initial_targets.py` also offer `dxr`/`vulkan` raycasting and `cuda`/`rtx` distance.
- `dxr` is Windows-only.
- H20 (Hopper) has no RT cores, so `vulkan`/`rtx` ray-tracing paths are probably unavailable on the rig.
- The `cuda` distance backend might work.
- The vae3d sources were not in the reviewing checkout, so none of this was checked.

**Fix.**
- Probe on one rig GPU: can `Queries(mesh, "cpu", "cuda")` and `Queries(mesh, "vulkan", ...)` be constructed, and do they produce results identical (sign) or within tolerance (distance) to CPU on ~10 assets at R256? Record [GPU 实测].
- If the raycast count must stay on CPU, consider a CUDA BVH ray caster that does not need RT cores (e.g. NVIDIA Warp `mesh_query_ray`) as a new backend. Exactness requirement: the 98-direction vote counts (`counts > 49`, `counts > 24.5`) must match CPU on validation assets, or any differences must be listed and accepted by the user. Different counts mean different targets.
- Expose `--raycast-backend` / `--distance-backend` on `run_g3_producer_worker.py` and `run_g3_stream.py`, recorded in the A1/A2 config (it already records them, see `prepare_initial_targets.py:142`).

### B2. A1 is dense R³ while targets are a narrow band

**Problem.** `initial_fields.py:49` loops over all R slices and casts 98 rays at every one of the R³ points (≈1.6 G rays at R256). Only points near the surface end up in `target.npz` (feature values are clipped at ±4 fine voxels, `prepare_initial_targets.py:156`). Cost scales with R³ instead of the band size.

**Fix (changes the target recipe: a separate experiment, not a drop-in).**
- Compute UDF first (cheap). Cast the 98-direction votes only for points with `udf < cap` plus a safety margin. Fill the sign elsewhere by connected-component flood fill from the band, or don't compute it at all if those points never enter the target.
- A2's gate mesh is marching cubes on the dense vote field. Either compute votes on a coarser grid for the gate, or restrict it to the band as well.
- Acceptance: on ≥ 50 assets, the band-only A1/A2 must give identical `target.npz` signs and fields (or an explained, user-approved difference list) to the dense version, and a measured speed-up. Until accepted, it must not write into the G3 pool.

### B3. A4 per-asset CPU overhead

**Problem.** Per asset, `run_g3_a4_batch.py:105` spawns a new interpreter (torch + CUDA init each time). `run_four_route_asset.py` then:
- re-reads and SHA-256-hashes the full manifest, A1/A2 summary and A0 audit JSONs (~35 k rows each);
- re-hashes the source mesh (`:115`) and the A2 target (`:151`);
- runs CPU `mesh_metrics` for A3 and A4 (`:213`);
- writes A3's `target.npz` with `savez_compressed` (single-threaded zlib) plus `raw_mesh.ply`, all of which `prune()` (`run_g3_a4_batch.py:36`) deletes right after.

**Fix.** Must not change the A4 maths or outputs. Verify a byte-identical A4 `features` array on a few assets before/after.
- Persistent worker: refactor `run_four_route_asset.main` into a function that takes already-loaded manifest/summary/audit objects and their precomputed hashes. `run_g3_a4_batch.py` calls it in-process for each asset in its slice, with per-asset try/except and `torch.cuda.empty_cache()` between assets. Keep a subprocess fallback flag for pathological assets (CUDA faults poison the process; the stream already tracks failures via `--max-a4-failures`).
- `--skip-a3-artifacts` (on for G3): keep A3 in memory, don't write its npz/ply or compute its metrics. Note the existing `--resume` path reloads A3 from disk; with this flag a crashed asset simply redoes A3.
- Store A4 `target.npz` uncompressed or with a faster compressor only if disk allows. Codely's compaction daemon and the 400 GB budget depend on size, so measure the size/CPU trade-off and ask the user before changing the format. Loaders use `np.load`, so uncompressed `savez` is read-compatible.
- Because these files feed `code_sha256` (see §0.2), do this between runs, not during one.

### B4. Producers compute A1/A2 metrics for every asset

**Problem.** `run_g3_producer_worker.py:142` runs `evaluate_pair` (CPU mesh metrics, 4096 samples) on every asset. It's diagnostic only; A4 does not need it.

**Fix.** `--metrics-every N` (or a deterministic hash-based sample, e.g. 5% of assets). Keep `summarize()` tolerant of rows without `metrics`. Check `summarize_g3_funnel.py` and `remote_g3_status.py` for consumers of those fields.

## Part C: resolution (for reference, no fix requested)

Resolution is not a single switch:
- A1/A2: any R ≥ 4, but dense R³ arrays and rays (see B2).
- A3/A4: `SUPPORTED_RESOLUTIONS = (128, 256)` (`run_four_route_asset.py:17`).
- Legacy `prepare_a4_targets.py`: hard-coded to 128.
- Trainers: `--resolution choices=(128, 256)`.
- Model: any multiple of 8.

Mesh normalization is `(R-2)/R`, so pools at different R cannot be mixed. Do not widen these lists as part of this work.

## Suggested order and acceptance

1. §0 checks. Ask the user about the GPU whitelist.
2. A0 → A6 → A1 → A2 → A3 → A4 (A5 is a run-plan note). One commit per item.
3. B5 → B6 (training-side, needed for G3 training).
4. B3 → B4 → B1 probe → B2 (pipeline; between runs only).
5. For each item: [静态检查] (`python -m py_compile`, argument parsing, resume `expected` dict updated), then [GPU 实测] on a small smoke (e.g. 2 GPUs × 50 steps, plus one resume from a mid-run checkpoint). Record commands and evidence paths in `docs/HANDOFF.md`.
6. Do not claim quality, speed or memory improvements without a [GPU 实测] before/after on the same assets and steps.
