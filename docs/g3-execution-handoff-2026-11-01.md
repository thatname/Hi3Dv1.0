# G3 执行交接（2026-11-01 15:40）

本文件是"重启 session 前后的续接卡"。实验全部落在 `/data/hi3d/g3/exp-35k-r256-e4-v1/`，本文件只写状态、待决策、命令与坑；详细设计见 `docs/experiment-records-g3-encoder-scale-v0.1.md` 与实验目录里的 `README.md`。

## 0. 一句话现状

转换已完成（20,693/21,911），**全量清单/split 正在后台构建**；网络（E4）已实测可跑；训练排法待你选 A/B/C；开跑前还差"容量策略 + 8 卡 E4 冒烟"。

## 1. 正在运行的作业

### 本地查看页（已用 `os.setsid` 独立会话，重启 CLI 不受影响）

| 端口 | 内容 | URL |
|---|---|---|
| 8765 | 网格浏览器（balanced50 四路线） | http://127.0.0.1:8765/ |
| 8767 | 旧 G2 预实验训练监控 | http://127.0.0.1:8767/ |
| 8768 | 场切片诊断（step12000，0/7/54） | http://127.0.0.1:8768/ |
| 8769 | R256 v5（78 件 12000 步，已完成） | http://127.0.0.1:8769/ |
| 8770 | **本实验**（`runs/train-v1`，开训前显示 waiting） | http://127.0.0.1:8770/ |

启动方式（macOS 无 `setsid(1)`，`nohup/disown` 会被工具进程组清理杀掉）：

```bash
cd /Users/changle/code/3dgen/hi3d-vae-field
python3 .local-cache/detach.py 8770 python3 -m experiments.vae_sdf.view_g1 \
  --port 8770 --no-browser --run /data/hi3d/g3/exp-35k-r256-e4-v1/runs/train-v1
# 日志：~/.codely-cli/viewer-logs/8770.log
```

### 远端后台作业（`setsid nohup`，与 ssh 会话解耦）

| 作业 | 状态 | 日志 |
|---|---|---|
| 全量清单 + load audit + 分层 split | 运行中（33,621 候选、48 线程、CPU ≈743%） | `/data/hi3d/g3/exp-35k-r256-e4-v1/logs/manifest-build.log` |

检查命令：

```bash
ssh tc-hk-h20-02-ide 'tail -5 /data/hi3d/g3/exp-35k-r256-e4-v1/logs/manifest-build.log; ls -l /data/hi3d/g3/exp-35k-r256-e4-v1/manifest/*.json'
```

完成标志：`manifest/manifest-g3-full.json`、`load-audit-g3-full.json`、`split-g3-full-v1.json`、`build-summary.json` 四个文件出现（`build-summary.json` 里有最终 loaded 数与面数 P10/中位/P90）。

## 2. 已完成

| 项目 | 结果 |
|---|---|
| 池格式审计 | 34,847 文件；可加载 33%（obj/glb/stl/ply 全通、gltf 52.5%、fbx/blend/dae 0%） |
| 格式转换 | 21,911 请求 → **20,693 成功（94.4%）**、失败 1,209、超时 9；26.4 GB；62 分钟；产物 `/data/hi3d/g3/converted/glb-v1/`（几何 GLB，无材质） |
| 网络 E4 | 编码器 **98.89 M** + 解码器 94.29 M = **193.18 M**；单步中位 2.07 s / 均值 2.51 s；峰值 84.6 GB（2.7M cell）；≈27 kB/cell；checkpoint ≈2.3 GB |
| 代码 | `--encoder-scale E0..E4` 已接入 `train_g1`、`train_g1_ddp`、`probe_r256_memory`；新增 `audit_mesh_pool.py`、`convert_mesh_pool.py`、`build_g3_inputs.py`、`run_g3_a4_batch.py` |
| 提交 | `a336c70`(E4) → `477f376`(独立目录) → `c4d8c64`(builder + A4 驱动 + 文档)；本地 `main` 领先 origin 若干，push 由用户执行 |
| 容量相关实测 | 100 件池最大 4.42M cell；E4 在 2.7M cell 峰值 84.6 GB |
| 压缩可行性 | `cell_indices` 可由 `cells + coords` 用 `VAE_CORNERS` **精确重建**（对现有 cache 0 mismatch）→ 缓存可省 42%；`raw_mesh.ply`（评测）与 `resolved_config.json`（provenance）必须保留 |

## 3. 待你决策

**训练排法（三选一）**

| 方案 | 安排 | 墙钟 | 归因 |
|---|---|---|---|
| **A（推荐）** | E4 先跑（8 卡、batch 8、12000 步），随后 E0 同配置补对照 | ≈16–18 h + 14–16 h | 可分离"数据规模"与"编码器容量" |
| B | E4 与 E0 各 4 卡、batch 4 并行 | ≈15 h | 两臂同口径，但每件更新次数减半 |
| C | 只跑 E4 | ≈16–18 h | 省时间，失去归因 |

E0 一臂的价值：测"31k 件数据单独能买到多少"（已知数据 8→80 件约 3.1×、R128→R256 约 2.6×）；若 E0 在 31k 上已接近 2.5e-4，则容量不是约束。

## 4. 开跑前两个阻塞项

1. **容量策略（必须先做）**：E4 ≈27 kB/cell → 4.4M cell ≈119 GB、5.0M ≈135 GB 触卡（143 GB 卡）。做法：用清单里的面数与包围盒预估 R256 cell 数（≈表面积/R²×厚度 2–3），冻结"预估 ≤ **4.0M** 才进训练集"的确定性口径，**E0/E4 两臂同口径**；超限资产只进 val/test 或直接排除，并写入 split 的排除清单。
2. **8 卡 E4 冒烟（约 20 分钟、200 步）**：验证 NCCL、每 rank 峰值、续训等价（193 M 参数每步 allreduce ≈770 MB）。

## 5. 接下来的执行顺序（可直接照抄）

```bash
# 环境
export PYTHONPATH=/data/hi3d/hi3d-vae-field:/data/hi3d/g0-prep/vendor-full:/data/hi3d/g0-prep/vendor-a3:/data/hi3d/g0-prep/vendor-a4:/data/hi3d/g0-prep/vae3d
export LD_LIBRARY_PATH=/data/hi3d/hi3d-validation/.venv/lib/python3.10/site-packages/torch/lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/data/hi3d/hi3d-validation/.venv/bin/python
EXP=/data/hi3d/g3/exp-35k-r256-e4-v1
```

1. **等清单完成** → 读 `$EXP/manifest/build-summary.json`，确认 loaded 数、面数分位、split 比例。
2. **容量审计**：按清单面数/bbox 预估 cell 尾巴，产出 `$EXP/manifest/capacity-policy-v1.json`（阈值 4.0M + 排除清单）。
3. **A1/A2 目标**（CPU、分块+剪枝）：`run_initial_batch`（`--resolution 256 --max-sites 6000000`）按块产出，随后立即用 `run_g3_a4_batch … --keep-a1a2`? 注意：A4 驱动默认会删 A1/A2，**A1/A2 必须先于 A4 且成块进行**，否则 6 TB 峰值。
4. **A4 目标（含剪枝）**：每卡一个 shard：
   ```bash
   CUDA_VISIBLE_DEVICES=1 $PY -B -m experiments.vae_sdf.run_g3_a4_batch \
     --manifest $EXP/manifest/manifest-g3-full.json \
     --a1a2-summary <A1/A2 summary> --a0-audit <A0 audit（只需 id 字段）> \
     --vae3d-root /data/hi3d/g0-prep/vae3d \
     --a1a2-root <A1/A2 root> --output-root $EXP/targets \
     --records $EXP/logs/a4-records --physical-gpu 1 --indices <分片>
   ```
5. **A4-only summary**：现有 `summarize_four_route_batch` 要求 a3+a4 齐全，**需写一个只含 a4 的汇总器**（schema 保持 `g0_four_route_paired_summary_v1`，`routes.flexicubes_smooth.{cache,a4.audit}` 必填）。这是目前最后一个未写的组件。
6. **8 卡 E4 冒烟** → **正式训练 E4**（`runs/train-v1`，`--encoder-scale E4 --asset-device cpu --steps 12000`，8 卡 DDP）→ 评测（与 legacy R256/E0 配对，按面数分位分层报告）。

## 6. 目录与产物

```
/data/hi3d/g3/
├── converted/glb-v1/                    池级转换产物（20,693 GLB + conversion-manifest.json）
└── exp-35k-r256-e4-v1/
    ├── README.md                        实验卡（冻结配置/结构/状态/provenance）
    ├── manifest/                        format-audit-v1.json、manifest-g3-full.json、load-audit-g3-full.json、split-g3-full-v1.json
    ├── targets/                         A4-only 缓存（规划中，压缩格式）
    ├── runs/train-v1/                   训练 run（未开始）
    ├── logs/                            manifest-build.log、A4 records
    └── reports/                         probe/（E3/E4 显存时间探针）、e3/e4-cost-probe-v1
```

只读引用：源网格池 `/data/TRELLIS/datasets/ObjaverseXL_github/raw/github/objects/`；历史对照 `g2/targets/four-route-r256-v1`、`g2/runs/r256-r128cells-78a-12000s-v5`、`g2/eval/*`。

## 7. 坑与约束（务必遵守）

1. **0 号卡禁用**；GPU 白名单 1/2/3/7（本轮训练阶段放宽到 8 卡，但 0 号仍不用）。单卡脚本默认 `cuda:0`，**必须显式 `CUDA_VISIBLE_DEVICES`**。
2. R256 训练必须 `--asset-device cpu` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，否则 80 件全常驻显存或碎片导致 OOM。
3. `check_split` 要求 `--indices` 与 split 的 train_indices **完全相等**；换资产集必须新建 split 文件。
4. `pkill -f build_g3_inputs` 会连自己的 ssh 命令行一起匹配杀掉（已踩过），改用更精确的模式或 `pgrep -f` 后按 PID 杀。
5. R256 评测须 `--max-predicted-sites 8000000`；`evaluate_g1` 的 `--role` 只接受 train/val，test 直接给 `--indices` 且不传 `--split`。
6. A3/A4 拟合**不可逐位重现**（A4 场 RMS 0.22、网格 Chamfer 1.07e-3），因此目标**必须缓存复用**，不能每次访问现场重算（会把比模型误差还大的噪声注进监督）。
7. 训练不缩放 lr（既定决定），batch 变化属独立 run，不能与历史 run 比 loss 轨迹，只能比固定 val/test 指标。

## 8. 关键数字速查

| 项目 | 数值 |
|---|---|
| 池规模 | 34,847 文件 / 121 GB；候选 33,621；转换成功 20,693 |
| R256 目标成本 | A1/A2 128 s/件（CPU）、A3/A4 11.8 s/件（GPU）；全链 ≈250 MB/件，A4-only ≈18 MB/件 |
| 训练单步 | E0 1.92 s / E4 2.07 s（中位，R256、单卡）；峰值 E0 87.7 GB / E4 84.6 GB（按资产 8.7–85 GB） |
| 显存单价 | E0 ≈23 kB/cell；E4 ≈27 kB/cell |
| 12000 步预算 | 8 卡 batch 8：E4 ≈16–18 h，E0 ≈14–16 h |
| 历史 R256 基线（E0，78 件） | val Chamfer 3.943e-4、test 4.736e-4、F@0.5% 0.9947/0.9957 |
| 磁盘 | `/data` 2 TB，剩 ≈1.1 TB |