# G3 执行交接（2026-11-02 12:40）

本文件是「重启 session 前后的续接卡」。实验全部落在 `/data/hi3d/g3/exp-35k-r256-e4-v1/`；方案演进与每次取舍的完整证据在 `docs/reports/g3-pipeline-tuning-2026-11-02.md`，阶段与资源图在 `docs/g3-stage-and-resource-map-2026-11-02.md`。

## 0. 一句话现状

**目标生成正在跑**（场 7,784 / 31,551、目标 6,469 / 31,551，两侧速率约 0.18 件/秒），预计 **35–40 小时**跑完；训练侧代码（流式加载、DDP）已就绪但**尚未启动**，8 卡冒烟还没做。

## 1. 正在运行的作业

### 远端（`setsid` 独立会话，与 ssh 解耦）

| 作业 | 进程数 | 日志 |
|---|---|---|
| 编排器 `run_g3_stream.py` | 1 | `logs/stream.log`（每 10 s 一行 status JSON） |
| 生产者 `run_g3_producer_worker.py`（A1/A2） | 18 | `logs/stream/produce-*.log` |
| 消费者 `run_g3_a4_batch.py`（A3+A4） | 12 | `logs/stream/consume-*.log` |
| 压缩守护 `compact_g3_targets.py` | 1 | `logs/compact.log` |

启动命令（可直接照抄）：

```bash
ssh tc-hk-h20-02-ide
cd /data/hi3d/hi3d-vae-field
export PYTHONPATH=/data/hi3d/hi3d-vae-field:/data/hi3d/g0-prep/vendor-full:/data/hi3d/g0-prep/vendor-a3:/data/hi3d/g0-prep/vendor-a4:/data/hi3d/g0-prep/vae3d
export LD_LIBRARY_PATH=/data/hi3d/hi3d-validation/.venv/lib/python3.10/site-packages/torch/lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VAE3D_TORCH_THREADS=1
E=/data/hi3d/g3/exp-35k-r256-e4-v1
setsid nohup /data/hi3d/hi3d-validation/.venv/bin/python -B -m experiments.vae_sdf.run_g3_stream \
  --manifest $E/manifest/manifest-g3-full.json \
  --load-audit $E/manifest/load-audit-g3-full.json \
  --a0-audit $E/manifest/a0-audit-g3-id-only.json \
  --vae3d-root /data/hi3d/g0-prep/vae3d \
  --a1a2-root $E/targets/a1a2 --targets-root $E/targets/a4 \
  --records $E/logs/a4-records --logs $E/logs/stream --resolution 256 \
  --producers 18 --consumers 12 --gpus 0 1 2 3 4 5 6 7 \
  --slice-size 40 --pending-gb 400 --min-free-gb 150 \
  > $E/logs/stream.log 2>&1 < /dev/null &
```

### 本地查看页

| 端口 | 内容 | 启动方式 |
|---|---|---|
| 8771 | **G3 流水线进度**（清单/策略/A1A2/A4/磁盘/ETA/利用率） | `python3 .local-cache/detach.py 8771 python3 -m experiments.vae_sdf.view_g3_pipeline --port 8771 --no-browser --experiment /data/hi3d/g3/exp-35k-r256-e4-v1` |
| 8765 / 8767 / 8768 / 8769 / 8770 | 旧页面（网格浏览器、G2 预实验、场切片、R256 v5、本实验训练监控） | 见 `docs/g3-execution-handoff-2026-11-01.md` |

利用率当前是**每 5 分钟采样**（3 次读数平均，缓存文件 `<实验>/reports/utilization-cache.json`）。

## 2. 已完成

| 项目 | 结果 |
|---|---|
| 全量清单 | 33,621 候选 → **31,551 可用**（failed 1,561 / degenerate 509）；进程池 11 分钟建完 |
| 容量策略 | 预估 cell > 4.0M 整件排除 → **512 件（1.62%）**；split **24,818 / 3,106 / 3,115** |
| A4 缓存压缩 | 33.3 MB → **20.8 MB/件**（去 `cell_indices` + 训练集删 ply）；全量约 658 GB |
| 流式训练 | `--targets-root` 惰性加载 + `cell_indices` 重建；实测单步 0.3 / 2.5 / 5.3 s（64k / 957k / 2.76M cell） |
| 解耦流水线 | 按资产领取的生产者 + 按就绪扫描的消费者 + 字节背压（详见 §4） |
| A4-only 汇总器 | `summarize_g3_a4_targets.py`（schema `g0_four_route_paired_summary_v1`，输出必须写在 targets-root 内） |
| 8771 页面 | 含实时速率、ETA、利用率（利用率采样间隔待改为 30 s 常驻采样，见 §6） |
| 文档 | `docs/reports/g3-pipeline-tuning-2026-11-02.md`（方案演进全集）、`docs/g3-stage-and-resource-map-2026-11-02.md`（阶段与资源图） |

## 3. 当前数字（2026-11-02 12:36）

| 项目 | 数值 |
|---|---|
| 场（A1/A2） | **7,784 / 31,551**（24.7%） |
| 目标（A4） | **6,469 / 31,551**（20.5%） |
| 速率 | 场 **0.18 件/秒**、目标 **0.18 件/秒** |
| 积压 | 1,315 件 / 204 GB（预算 400 GB，未触发暂停） |
| 负载 | 136 / 128（5 分钟均值 145） |
| 磁盘 | `/data` 2 TB，可用 **732 GB** |
| 单件成本 | A1/A2 118 s 墙钟 ≈ **500 核·秒**；A4 41 s（空载）～120 s（争用）≈ **200–240 核·秒** |
| ETA | 场剩余 23,767 件 ≈ 37 h；目标剩余 25,082 件 ≈ 39 h → **11 月 4 日凌晨** |

## 4. 流水线结构（当前生效）

```
源网格池 ──► A0 审计（已完成）
                 │
                 ▼
        A1 98 向投票定号 + A2 背面壳修正      ← 18 个 CPU worker，按资产 claim 领取
                 │   写 targets/a1a2/{id}/   161 MB/件
                 ▼
        缓冲区（字节背压：>400 GB 或可用 <150 GB 就写 pause 让生产者退出）
                 │
                 ▼
        A3 首次可微拟合(30 步) + A4 delta 精修(200 步)  ← 12 个 GPU 消费者
                 │   写 targets/a4/{id}/   20.8 MB/件（压缩守护随后处理）
                 ▼
        训练（未启动）：G1 VAE，E4 编码器 98.89 M + 解码器 94.29 M，12000 步 × batch 8
```

- 生产者：`run_g3_producer_worker.py`，用 `os.O_CREAT|O_EXCL` 抢 `claims/{index}.claim`，一件一领，进程内循环（省掉每件约 10 s 的进程启动）；claim 30 分钟可被接管。
- 消费者：编排器每轮直接扫 `targets/a1a2/assets/`，挑「有结果、无 A4」的资产按 40 件一批派发；`inflight_indices` 防重复。**不要退回"按 index 队头取料"的旧逻辑**——那会因队头全是未就绪资产而把消费者饿死（2026-11-02 11:30 事故）。

## 5. 待办（按顺序）

1. **8 卡 DDP 冒烟**（200 步，约 30–40 min）：验证 NCCL、每 rank 峰值显存、流式加载、续训；需短暂停 A4（会损失约 500 件 A4 进度）。
2. **A4-only 汇总器**：目标全量完成后跑
   `python -m experiments.vae_sdf.summarize_g3_a4_targets --manifest $E/manifest/manifest-g3-full.json --targets-root $E/targets/a4 --output $E/targets/a4/paired-summary-g3-a4-v1.json`
   （`evaluate_g1` 的 `--paired-summary` 依赖它）。
3. **正式训练**：E4 臂 8 卡 batch 8 × 12000 步（`train_g1_ddp --targets-root ... --split split-g3-full-v2-capacity.json`），随后 E0 对照臂。
4. **评测**：val 0/30/50–57、test 1/58–66，与 g2 的 R256 基线（val 3.943e-4 / test 4.736e-4）配对比较。
5. **利用率采样改 30 s**：按"远端常驻采样进程写缓存、页面只读"的方式实现（方案 B，尚未做）。

## 6. 坑与约束（务必遵守）

1. **0 号卡已解禁**，白名单 `(0,1,2,3,4,5,6,7)`；4/5/6 上是 `bosheng` 用户的常驻进程（GPU 利用率长期 0%、各 650–750 MiB），用户已同意挤占。
2. **白名单外的卡会让 A4 分片 argparse 秒退**，编排器只看到"分片退出"，极易误判成并行正常。
3. **`run_initial_batch` 的 plan.json 固定了自身 sha256**：改这个文件会让进行中的 output-root 报 "Existing batch plan differs"。要改行为只能用外部包装（现有 `run_initial_batch_single_thread.py`）。
4. **`pkill -f "run_g3_..."` 会连自己的 ssh 命令行一起杀掉**（命令行里含模块名），必须用 `ps -eo pid,comm,args | awk '$2 ~ /^python/ && /run_xxx/ {print $1}' | xargs -r kill`。
5. **`back_shell.cpu_scene` 的 `nthreads=8` 是空转**：猴补丁改 1 线程后单件耗时与 CPU 都无变化（实测 121 s / 435–518% CPU 不变）；把 `--samples` 从 4096 降到 64 也只快 4 s。**A1/A2 的 500 核·秒是算法成本，不要在这上面继续调参。**
6. **A4 子进程每件约 31 s 是非 GPU 开销**（导入 9.9 s + 读 161 MB 缓存 + 指标），GPU 拟合只有 12.8 s；这是 A4 单件 41 s 的由来。
7. **训练首步有 455 s 预热**（cuDNN/cuBLAS/spconv 算法初始化），前 5–8 分钟慢属正常；`--stream-resident 1` 每步重载一件，重载成本可忽略。
8. **Vulkan/OptiX 路径在本机不可用**（`Failed to create Vulkan instance`），CUDA 设备正常但不能跑 slang RT；A1/A2 只能 CPU。
9. **容量判定是估计值**（P90 误差 33%），训练侧仍以 A4 实测 cell 为准做最终核对。
10. **组尾/队头这类"饿死"问题会伪装成"变慢"**：先看进程数与负载，再看单件耗时，不要只看完成数。

## 7. 目录与产物

```
/data/hi3d/g3/
├── converted/glb-v1/                     池级转换 GLB（20,693 件，26.4 GB）
└── exp-35k-r256-e4-v1/
    ├── README.md                         实验卡
    ├── manifest/                         manifest-g3-full.json（含 bytes/area/extent）
    │                                     load-audit、a0-audit-g3-id-only、split v1/v2
    │                                     capacity-policy-v1.json、capacity-calibration.json
    ├── targets/
    │   ├── a1a2/                         场缓存（161 MB/件）+ assets/*.json 结果 + claims/ + plan.json + summary.json
    │   └── a4/                           A4 目标（压缩后 20.8 MB/件）+ 压缩守护处理的产物
    ├── runs/train-v1/                    训练 run（未创建）
    ├── logs/                             stream.log、stream/、a4-records/、compact.log、pipeline/
    └── reports/                          探针、utilization-cache.json
```

只读引用：源池 `/data/TRELLIS/datasets/ObjaverseXL_github/raw/github/objects/`；历史对照 `g2/targets/four-route-r256-v1`、`g2/runs/r256-r128cells-78a-12000s-v5`、`g2/eval/*`。

## 8. 关键数字速查

| 项目 | 数值 |
|---|---|
| 可用资产 / 排除 | 31,551 / 512 |
| split | 24,818 / 3,106 / 3,115 |
| A1/A2 单件 | 118 s，约 500 核·秒，产物 159–161 MB |
| A4 单件 | 41 s（空载）～120 s（争用），产物 20.8 MB（压缩后） |
| 当前速率 | 场 0.18 件/秒、目标 0.18 件/秒 |
| 甜点并发 | 生产者 18、消费者 12（负载 136–183） |
| 磁盘 | 可用 732 GB，全量 A4 预计约 658 GB（含已生成部分） |
| 网络 | E4 编码器 98.89 M + 解码器 94.29 M = 193.18 M，latent 32 |