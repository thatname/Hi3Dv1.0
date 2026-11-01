# R256 分辨率尝试、多卡训练与 R2048 可行性

日期：2026-10-31
撰写：Codely CLI（基于本机与 `/data/hi3d/` 实测产物）

证据标注约定：**[源码]** 源码核查、**[静态]** 静态/配置检查、**[GPU]** GPU 实测、**[CPU]** CPU 实测、**[外推]** 由实测单位成本外推，非实测。

## 0. 结论摘要

1. **多卡训练可用但不是加速主力**：四卡（1/2/3/7）DDP 每步处理 4 件资产，按资产更新计约 **2.6×** 单卡吞吐（不是 4×），trace 合并与断点续训已验证；加卡**不改变单件资产的显存上限**。[GPU]
2. **R256 目标显著更好**：同 100 件、同协议、同 seed，A4 Chamfer 中位数 2.147e-3 → **1.492e-3**，P90 6.968e-2 → **4.585e-2**，F@0.5% 0.9874 → 0.9960；代价是 A4 单件 8.0 → 11.8 s、倒置 cell 资产 42 → 51。[GPU]
3. **R256 训练撞的是单卡激活显存墙**：单步峰值 ≈22.4–31 kB/最细 cell，单卡上限约 **4.2M cell**；80 训练件中 14 号（4.42M）、95 号（4.38M）超卡，68 号（4.17M，93.4 GB）与 70 号（4.01M，89.9 GB）可跑。[GPU]
4. **R256 训练出现整块 NaN**：第 126 步（68 号）前向输出全 NaN，参数审计显示第 125 步更新后权重仍有限、同一资产从初始权重单独跑 20 步正常；已改为"跳过并逐层记录"。[GPU]
5. **R2048 不是"把分辨率调高"**：三个卡点按严重度是 ①目标生成（CPU 射线 ≈R³，18 h/件）②存储（20–75 GB/件，4 万件 0.8–3 PB）③显存（1.9–6.9 TB/件）。必须换训练形态（分块）与目标生成方式，见 §8。[外推]
6. **R256 训练与评测已完成（2026-11-01 追加）**：78 件 12000 步 6.94 h 跑完；val/test 各 10 件与 R128 同集对照，Chamfer 中位 1.271e-3→3.943e-4（val）、1.230e-3→4.736e-4（test），test **10/10 件同向变好**；但绝对改善 <1e-3 与预先登记的闸门口径冲突，且 watertight/边界边未改善。详见 §6。

## 1. 本次接受以来的代码改动

| 文件 | 改动 | 理由 | 验证 |
|---|---|---|---|
| `experiments/vae_sdf/train_g1_ddp.py` | 重写加固：配置记录与单卡对齐（`code_commit`/`code_sha256`/`torch_version`/`gpu`/`assets`/`split`/`split_sha256`/`resume_from`/`rng_restored_ranks`/`save_every`/`elapsed_seconds`/逐 rank 峰值）；失败写 `status=failed` + `error` 并保证 `finally` 落盘；trace 每行含 `rank` 与 `gradients`；周期 checkpoint 记 `status=completed` 以支持续训；resume 校验 schema/stage/indices/两个 sha256/lr/loss 权重/resolution/world_size/split/asset_device 且要求 `world_size` 相同；RNG 逐 rank 恢复；失败用 collective 失败位同步中止；`--indices` 长度必须 ≥ rank 数 | 原实现只记录 world 与 lr，失败不落盘、rank 非 0 的错误会挂住其它 rank，且无法定位"哪一步哪个 rank" | [GPU] 四卡 smoke + 两次续训 |
| 同上 | `find_unused_parameters` 保持 `True` | 设 `False` 时第 31 步报"参数 100/101 未收到梯度"，说明某些步存在未参与 loss 的分支 | [GPU] |
| `train_g1.py` | 新增 `--asset-device {cpu,cuda}` 与 `upload()`：资产张量可留在主机，每步只上传当前件 | 80 件 R256 全常驻 GPU 需 ~120 GB，与激活叠加必然 OOM | [GPU] |
| 同上 | trace 每行加 `cuda_allocated_before` / `cuda_step_peak_bytes` / `cuda_allocated_after`；`peak_cuda_allocated_bytes` 取全部 step 峰值的最大 | 定位"件大"与"跨步累积"必须分开看 | [GPU] |
| 同上 | `loss_diagnosis()`：逐项（field/delta/support/validity）、逐解码层（`levels_nonfinite`/`levels_sites`）、latent 非有限计数 | NaN 出现位置决定是"空组/退化归一化"还是"权重被写坏" | [GPU] |
| 同上 | `optimizer.step()` 后审计全部参数是否有限，违者报出模块名 | 区分"前向产生 NaN"与"更新后权重毒化" | [GPU] |
| 同上 | `--max-skipped`（默认 20）：非有限批次跳过且不更新，记录 `skipped_nonfinite_steps` 与 `skipped_nonfinite_detail`；resume 的 trace 长度校验计入被跳过步 | 单个坏批次不应报废多小时 run | [GPU] v5 第 126 步跳过、跑完 12000 步 |
| `evaluate_g1.py` | 归一化半径由硬编码 `(128-2)/128` 改为 `(args.resolution-2)/args.resolution` | R256 评测否则会把网格按 R128 尺度归一化，指标全错 | [GPU] val/test 四项评测 |
| `run_four_route_batch.py` | 新增 `--resolution {128,256}`，转发给 `run_four_route_asset` 并写入 batch identity，旧根目录缺该字段按 128 兼容 | R128 目标只能靠单件入口生成，无法用带锁的分片驱动 | [GPU] 四卡 R256 分片 |
| `summarize_four_route_batch.py` | 缓存目录不再硬编码 `R128`，改为 `R{identity.resolution}`；输出增加 `resolution`；仍需显式 `--output` 才落盘 | 否则 R256 批次无法汇总 | [GPU] |
| `probe_r256_memory.py`（新增） | 单步显存探针：`--cycles` 重复同一批资产、`--keep-locals` 模拟训练循环（保留上一批张量、不 flush 缓存）、`--asset-device`，逐件输出 cells / 峰值 / bytes_per_cell | 区分"这件太大"与"跨步累积" | [GPU] |

另有过程性问题记录：`train_g1.py` / `evaluate_g1.py` 不设 `CUDA_VISIBLE_DEVICES` 时用 `cuda:0`，本次有一次单卡基线误落到物理 0 号卡（已停掉重跑），**所有单卡命令必须显式指定白名单卡**。

## 2. 实验一：四卡 DDP smoke [GPU]

配置：`--gpus 1 2 3 7`，R128，`split-g2-100-v1.json` 的 80 训练索引，20 步，`--save-every 10`；输出 `/data/hi3d/g2/runs/ddp-smoke-r128-80a-{20s,30s-resume}-v1`。

| 项目 | 结果 |
|---|---|
| trace | 20 步 × 4 rank = 80 行；每步四 rank 资产互不相同（step1 = 2/3/4/5，step20 = 96/97/98/99） |
| 梯度 | 每行四组梯度齐全且有限（encoder / decoder_backbone / support_heads / continuous_head） |
| wall | 69.3 s（首步 54 s 为 80 件加载 + 算子预热），稳态 0.4–1.2 s/步，均值 0.83 s/步 |
| 单卡基线（同索引同 20 步，GPU7） | 26.0 s，稳态约 0.55 s/步 |
| 吞吐结论 | 每步只算一件资产，四卡只把"每步"换成 4 件并行 → 按资产更新计约 **2.6×**；NCCL 全参 allreduce 占比高，规模小时不可能线性 |
| 续训 | 从 completed checkpoint 续 (20→30)、再续 (30→32)：`start_step` 正确、资产序接续（step21 = 2/3/4/5）、四 rank RNG 全部恢复（`rng_restored_ranks=[0,1,2,3]`） |
| 峰值 | 逐 rank 25.3 / 22.5 / 17.4 / 15.6 GB（rank0 另担 trace 与 checkpoint 写入） |

## 3. 实验二：R256 目标全量生成 [GPU]/[CPU]

产物：`/data/hi3d/g2/targets/r256-a1a2-v1`（`summary.json` 100/100）、`/data/hi3d/g2/targets/four-route-r256-v1`（`batch-status.json`、`paired-summary.json`）。协议与 R128 相同：单路线 `flexicubes_smooth`、A3 30 步/8192 采样、A4 200 步/4096 渲染、metric 4096、seed 42。

| 阶段 | 覆盖 | 单件成本 | 备注 |
|---|---|---|---|
| A1/A2 | 100/100 | 缓存复用后评价 1.6–10.6 s；全量生成 128 s/件 | 12 路并行补齐 90 件约 25 分钟；必须 `--max-sites 6000000`（7、44 号超默认 2M） |
| A3/A4 | 100/100 | 19–127 s（中位约 53 s） | 四卡分片，10 件子集约 4 分钟、其余 90 件约 18 分钟 |

目标质量（100 件配对，`paired-summary.json` 的 `routes.flexicubes_smooth`）：

| 指标 | R128 | R256 | 变化 |
|---|---|---|---|
| A4 Chamfer 中位数（归一化） | 2.147e-3 | **1.492e-3** | −30% |
| A4 Chamfer P90 | 6.968e-2 | **4.585e-2** | −34% |
| F@0.5% 中位数 | 0.9874 | **0.9960** | +0.9 pt |
| 法向一致性中位数 | 0.9716 | 0.9745 | +0.3 pt |
| 边界边中位数 | 0 | 8 | 变差 |
| 带倒置穿越 cell 的资产数 | 42 | 51 | 变差 |
| A4 单件耗时中位数 | 8.0 s | 11.8 s | +48% |
| 存储/件（A1/A2 + 四路线） | ≈60 MB | **≈250 MB** | 4.1× |

注意：单件 A4 差值 6.5e-4 低于 A4 复跑噪声 1.07e-3，但这是 100 件中位数的系统性位移，不是单件比较。

## 4. 实验三：R256 训练显存与单卡上限 [GPU]

方法：`probe_r256_memory.py`，单件一次前向+反向，`--asset-device cpu`，`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。产物 `/data/hi3d/g2/analysis/r256-mem-probe-v{1..4}/probe.json`。

| 资产 | 最细 cell | 单步峰值 | kB/cell | 结论 |
|---|---|---|---|---|
| 5 | 0.32M | 7.9 GB | 25.2 | 可跑 |
| 7 | 2.74M | 60.0 GB | 23.2 | 可跑 |
| 11 | 3.40M | 73.7 GB | 21.7 | 可跑 |
| 70 | 4.01M | 89.9 GB | 22.4 | 可跑 |
| 68 | 4.17M | 93.4 GB | 22.4 | 可跑 |
| 95 | 4.38M | **>139 GB** | — | OOM |
| 14 | 4.42M | **>139 GB** | ≈31 | OOM |

- H20-3e 每卡 143 GB、可用 139.8 GiB → 单卡上限约 **4.2M 最细 cell**；`kB/cell` 与资产的空间展开度有关（紧凑件约 22.4，跨范围大的件约 31）。
- 累积与件大要分开看：`--cycles 4` 重复同一批资产时 `allocated_before` 稳定在 1.0–1.6 GB，峰值按 cell 数可重复；`--keep-locals` 模拟训练循环同样不累积。**真正的累积来自"80 件资产常驻 GPU"**（≈120 GB），用 `--asset-device cpu` + `expandable_segments:True` 后每步 `allocated_before` 只有 2–3 GB。
- 因此 R256 的训练约束是"单件峰值 ≤ 卡容量"，**加卡无法缓解**：DDP 每 rank 仍处理整件资产、每卡一份完整模型。
- 为能开训，新建训练 split `/data/hi3d/g2/inputs/split-g2-100-v1-r256-minus2.json`（78 训练 / 10 验证 / 10 测试，只剔除 14、95，两者均属训练集，val/test 不受影响，剔除原因与实测 cell 数写入文件）。

## 5. 实验四：R256 单步训练成本 [GPU]

单卡（GPU7）R256、78 件轮转：

- 稳态 **≈2.0–2.2 s/步**（中位 2.02，均值 2.22；R128 为 0.55 s/步）→ 约 3.3–3.7×；
- 单步峰值 8.7–73.7 GB，随当前资产的 cell 数变化，`allocated_before` 稳定 2–3 GB；
- 12000 步预计 **7–10 h**（含大件的轮转步会更慢）。

对比口径：既有 R128 基线是 80 件 × 12000 步（8161 s，峰值 24.46 GB）。为了与它同数同步数比较"提分辨率是否是有效杠杆"，R256 走**单卡 78 件 12000 步**；四卡 DDP 要等资产更新次数则需 3000 步（约 3 h），但等效 batch 4 且按既定决定不缩放 lr，只能比评测指标、不能比 loss 轨迹。

## 6. R256 训练结果与 R128/R256 评测对照（2026-11-01 追加）[GPU]

### 6.1 训练结果

`/data/hi3d/g2/runs/r256-r128cells-78a-12000s-v5`：78 件 × 12000 步（每件约 154 次更新）。

| 项目 | 数值 |
|---|---|
| 墙钟 | **6.94 h**（逐件排程预估值 6.42 h，偏低 8%） |
| 单步 | 中位 1.92 s / 均值 2.24 s；拟合 `sec ≈ 0.17 + 1.35 µs × 最细 cell` |
| loss | 2.157（前 10 步均值）→ **0.046**（末 10 步均值） |
| 分项 | field 1.559→0.033、delta 0.0054、support 0.004 |
| 显存 | 末 20 步峰值 87.7 GB；单件最大约 93 GB（68 号） |
| NaN | 第 126 步 1 次，跳过未更新；其后 11873 步无复现 |
| checkpoint | 每 2000 步一份 + 最终 `checkpoint.pt`，各约 1.17 GB |

### 6.2 评测协议

`evaluate_g1`：val = 0/30/50–57，test = 1/58–66，两分辨率**同集**（被剔除的 14/95 只在训练集）；`samples=2048`、`seed=20261030`、`support_threshold=0`、R256 加 `--max-predicted-sites 8000000`；raw mesh、原坐标、不清理；Chamfer 按源网格包围盒对角线归一化。R128 基线用既有 12000 步 checkpoint 在**本次评测器**下重算（历史报告 1.16e-3/1.18e-3 与本次 1.27e-3/1.23e-3 有 0.1e-3 级差异，对照一律用本次值）。

### 6.3 结果（10 件中位）

| 指标 | R128 val | **R256 val** | R128 test | **R256 test** |
|---|---|---|---|---|
| 网格 Chamfer（归一化） | 1.271e-3 | **3.943e-4** | 1.230e-3 | **4.736e-4** |
| Chamfer P90 | 2.340e-3 | **7.863e-4** | 2.775e-3 | **1.882e-3** |
| F@0.5% | 0.9734 | **0.9947** | 0.9864 | **0.9957** |
| 支撑精度 | 0.9880 | **0.9941** | 0.9888 | **0.9948** |
| 支撑召回 | 0.9936 | **0.9967** | 0.9945 | **0.9964** |
| 场 MAE（近表面加权） | 0.387 | **0.356** | 0.388 | **0.353** |
| watertight | 1/10 | 1/10 | 4/10 | **3/10（变差）** |
| 边界边中位数 | 118 | 106 | 9 | **74（变差）** |

逐件配对（test 10 件）：**10/10 资产 R256 更好**，Chamfer 比值 0.10–0.68（中位 0.34，≈2.9×），中位绝对改善 **−8.47e-4**、均值 −1.32e-3；F@0.5%、支撑精度、召回每件提升或持平；边界边 7/10 变差（58 号 2140→13830、59 号 1824→5566）。

### 6.4 结论与闸门口径冲突

- **分辨率是有效杠杆**：相对改善 2.6×（test）/3.2×（val）、逐件 10/10 同向、P90 与 F@0.5% 同向。A4 拟合噪声（1.07e-3）是逐件随机，不产生 10/10 同向的系统性改善。
- **"表示下限"是分辨率绑定的**：原先记为 R128 + factor-8/latent-R16 共同下限的 1.16e-3，在 R256 降到 0.39–0.47e-3，说明该下限主要由细网格分辨率决定，而非 latent 容量。
- **口径冲突（需登记）**：预先登记"改善 <1e-3 即不划算"，按绝对中位数 val 8.77e-4、test 7.56e-4 判负；按逐件配对符号 + 相对比值判正。基线本身已接近噪声量级时绝对中位数会失效，建议改用逐件配对口径，需用户确认后写入方案文档。
- **边界/水密与分辨率正交**：watertight、边界边未改善（test 反而变差），与"支撑比真实窄一圈、零面在支撑边缘被切断"的定位一致。提分辨率只是让更多边界暴露，不修根因。

## 7. 未解决的问题：R256 训练第 126 步整块 NaN [GPU]

现象（`r256-r128cells-78a-12000s-v1/v2/v3`，三次同位置复现）：

- 第 125 步正常（loss 0.397，梯度有限），第 126 步（68 号，4.17M cell）前向即产 NaN：`signed_nonfinite = 33,365,152 = 4,170,644×8`、`delta_nonfinite = 100,095,456 = ×24`，即所有角点全 NaN，而 `target_nonfinite = 0`。
- 同一资产用初始权重单独跑 20 步完全正常（loss 2.16→0.90，峰值 88 GB，5.5 s/步）。
- 第 125 步 `optimizer.step()` 后的全参数审计**未报非有限**，所以不是"更新把权重写坏"；NaN 出现在第 126 步前向内部。
- 最可能机制：稀疏解码器在某个层级出现空组/退化统计量（`support` 项即 cls_head logits 为 NaN），而 PointGroupNorm 类的按组归一化会把单点 NaN 扩散到整批输出。

当前处置与实测结果（`r256-r128cells-78a-12000s-v5`）：

- 坏批次不再中止 run：记录逐项与逐层诊断，跳过该步且不更新，`skipped_nonfinite_steps` 与明细写入 config；超过 `--max-skipped`（默认 20）才中止。
- **v5 结果：12000 步中仅第 126 步 1 次触发**，跳过后续跑到 12000 步，其后 11873 步无复现 —— 属低频事件，不是持续退化。
- 逐层诊断这次没有给出有效信息：`occupancy` 是三个 0-d 标量（每层 logits 的均值），因此 `levels_nonfinite=[0,0,0]`、`levels_sites=[1,1,1]` 无意义；而 `latent_nonfinite=0`（`mean`/`logvar` 有限）说明 NaN 出现在 **latent 之后、解码器内部**。要定位到具体层需在解码器模块上加前向 hook，捕获第一个输出非有限的模块（失败时单件重跑一次即可，代价数秒）。

监视：本地 `http://127.0.0.1:8769/`（`view_g1` 指向 v5，已完成；8767 仍为旧 G2 预实验 run，`view_g1` 一次只接一个 run 目录）。

## 8. R2048 的问题与可选方案

### 8.1 三条硬约束（按严重度）

| 维度 | 实测/外推 | R2048 的量级 |
|---|---|---|
| 目标生成 | A1/A2 是 98 方向 UE50 射线投票，查询数 ≈98×R³；R128 20 s、R256 128 s/件 | **≈18 h/件**（CPU 绑定） |
| 存储 | R256 ≈250 MB/件 | **20–75 GB/件**，4 万件 0.8–3 PB，200 万件 40–150 PB |
| 显存 | 单步峰值 ≈22 kB/最细 cell；R256 中位 1.03M cell → 24 GB | 中位 **1.9–6.9 TB/件**，单卡 143 GB 差 1–2 个数量级 |

支撑 cell 数随分辨率近似按面积增长：实测 R128→R256 为 4.3×（中位）到 6.7×（大件），即 ≈R^2.1–2.6；每翻一倍显存/存储 ×4.3–6.7。按此外推：

| 分辨率 | 单件 cell | 单件单步显存 | 单件存储 | A1/A2 | A3/A4 |
|---|---|---|---|---|---|
| R256（实测） | 1.03M | 24 GB | 250 MB | 128 s | 12 s |
| R512 | 4.4–6.9M | 100–160 GB | 1.1–1.7 GB | 17 min | 47 s |
| R1024 | 19–46M | 0.43–1.0 TB | 4.7–11 GB | 2.3 h | 3 min |
| R2048 | 82–306M | 1.9–6.9 TB | 20–75 GB | 18 h | 12 min |

### 8.2 可选方案

| 方案 | 做法 | 解决什么 | 代价 |
|---|---|---|---|
| **A 分块训练** | 资产切成固定体素块（如 128³ 最细体素、带重叠），每步只喂一块；跨块用重叠感受野 + 一致性损失，推理滑窗拼接 | 显存与资产规模解耦（128³ 表面带约 2–4M cell ≈ 50–90 GB，单卡可跑，R2048 同样适用） | latent 语义从"整件"变"块局部"；边界一致性是新风险；属训练形态改动 |
| **B 换目标生成** | 只对最细层 cell 中心查距离（实测距离查询约 5e6 次/s，单件 3–8e7 查询 ≈6–16 s），或局部平面/近邻面近似，替代 98 方向投票 | 把 18 h/件 降到十几秒 | 目标语义从"UE50 投票一致"变"几何 SDF"，A3/A4 拟合协议需重标定，与历史 R128/R256 目标不可直接比较 |
| **C 降 decoder 倍率** | factor 8 → 16/32：R2048 栅格上只预测 R128/R64 的 latent | 显存与算力回到今天的量级，输出仍是 2048 栅格 | 最细细节由 base grid 决定，"2048 的收益来自栅格化而非表征" |
| **D 两段式/超分** | R256 全局 VAE + 块级细节网络学高频残差 | 只需存块样本，不必为每件存全量 R2048 场 | 多一个网络与一套训练/拼接流程 |
| **E 精度与分布式** | bf16 激活、梯度重算、激活 offload、FSDP 分片激活 | 合计约 4–8× 降低 | 对 R512（100–160 GB）够用；对 R2048 的 TB 级仍差 10–40×，只能辅助 A |
| **F 数据管线** | 4 万件 R256 需 10 TB（本机 2 TB，现剩 142 GB）：扩盘到 ≥16 TB，或只缓存 A1/A2 且 A3/A4 在线生成（4 卡 4 万件约 33 h），或压缩到 fp16，或把训练集缩到 8–10k | 解决 4 万件的落盘 | 在线生成会增加训练侧耦合；缩集改变实验口径 |

### 8.3 建议

- 若目标是"4 万件规模 + 最终能出 R2048 结果"：训练集走 **R256 全局**（先解决 10 TB 存储或 A3/A4 在线生成），R2048 **不作为端到端训练分辨率**，而用 **C（factor 16/32）+ D（块级细节网络）** 达成；必要时用 **A** 训练细节块。
- 若一定要端到端训 R2048：必须先落 **A（分块）+ B（廉价目标）+ E（精度/重算）**，且训练集要显著小于 4 万（存储与生成时间决定，可能 2–5k）。这是一个独立项目的工作量，不是参数调整。

### 8.4 建议先做的低成本验证

1. 在现有 100 件上取样几件做 **R512** 目标生成（预计 17 min/件）+ 单步显存/时间实测，把 §8.1 的 R^2.1–2.6 指数与 kB/cell 常数校准成实测值；
2. 用 **B** 的单件原型（只查 cell 中心距离）与现有 A1/A2 场做对比，量化"目标语义改变"带来的差异，判断 B 是否可接受。

## 9. 复现命令

```bash
# 环境（远端）
export PYTHONPATH=/data/hi3d/hi3d-vae-field:/data/hi3d/g0-prep/vendor-full:/data/hi3d/g0-prep/vendor-a3:/data/hi3d/g0-prep/vendor-a4:/data/hi3d/g0-prep/vae3d
export LD_LIBRARY_PATH=/data/hi3d/hi3d-validation/.venv/lib/python3.10/site-packages/torch/lib
PY=/data/hi3d/hi3d-validation/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # R256 训练必需

# 四卡 DDP（R128）
$PY -B -m experiments.vae_sdf.train_g1_ddp --manifest ... --paired-summary ... --weights ... \
    --split /data/hi3d/g1/g2-prep/split-g2-100-v1.json --indices <80 训练索引> \
    --gpus 1 2 3 7 --steps 20 --save-every 10 --stage G2 --output <新目录>

# R256 目标（A1/A2，缓存不存在时才会重算）
$PY -B -m experiments.vae_sdf.run_initial_batch --manifest ... --load-audit /data/hi3d/g2/inputs/load-audit-r256.json \
    --vae3d-root /data/hi3d/g0-prep/vae3d --output-root /data/hi3d/g2/targets/r256-a1a2-v1 \
    --resolution 256 --max-sites 6000000 --indices <索引>

# R256 目标（A3/A4，四卡分片）
$PY -B -m experiments.vae_sdf.run_four_route_batch --manifest ... --a1a2-summary /data/hi3d/g2/targets/r256-a1a2-v1/summary.json \
    --a0-audit /data/hi3d/g1/g2-prep/a0-audit-g2-100.json --vae3d-root /data/hi3d/g0-prep/vae3d \
    --output-root /data/hi3d/g2/targets/four-route-r256-v1 --routes flexicubes_smooth --resolution 256 \
    --a3-steps 30 --a3-samples 8192 --a4-steps 200 --image-resolution 4096 --metric-samples 4096 --seed 42 \
    --indices <分片索引> --physical-gpu <1|2|3|7>

# R256 训练（单卡，资产留主机）
CUDA_VISIBLE_DEVICES=7 $PY -B -m experiments.vae_sdf.train_g1 --manifest ... \
    --paired-summary /data/hi3d/g2/targets/four-route-r256-v1/paired-summary.json --weights ... \
    --split /data/hi3d/g2/inputs/split-g2-100-v1-r256-minus2.json --stage G2 \
    --output /data/hi3d/g2/runs/r256-r128cells-78a-12000s-v4 --indices <78 训练索引> \
    --steps 12000 --resolution 256 --asset-device cpu --save-every 2000

# 单步显存探针
CUDA_VISIBLE_DEVICES=7 $PY -B -m experiments.vae_sdf.probe_r256_memory --manifest ... --paired-summary ... \
    --weights ... --output /data/hi3d/g2/analysis/r256-mem-probe-v4/probe.json --indices 68 70 95 14
```

## 10. 未验证与风险

- R2048 的全部数字是 [外推]，指数（R^2.1–2.6）与 kB/cell 常数需 R512 实测校准。
- NaN 的根因未确认，当前是"跳过 + 留证"（v5 中 12000 步仅 1 次）；逐层诊断字段无效，需改解码器模块 hook 才能定位，是否做空组保护/归一化 eps 取决于定位结果。
- **闸门口径待用户确认**：按绝对中位数（val 8.77e-4、test 7.56e-4 < 1e-3）判负，按逐件配对符号（test 10/10 同向、比值中位 0.34）判正；未确认前，R256 是否作为主分辨率不能写入方案文档。
- DDP 只验证到 20/30 步规模；长训练下的 NCCL 稳定性、checkpoint 体积（四份 RNG + 完整模型）与断点续训的等价性未验证。
- `--asset-device cpu` 每步上传约 1.2–1.9 GB，实测已包含在 2.0–2.2 s/步内，但未单独测量其占比。
- 78 件 R256 与 80 件 R128 的比较存在 2 件资产差异，评测时须在固定 checkpoint 上按同一批资产重算 R128 指标，不能直接比历史数字。