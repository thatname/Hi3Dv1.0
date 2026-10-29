# Hi3D VAE 场迁移实验进展（2026-10-30）

## 一句话状态

固定 50 个 mesh 的 G0 目标生成已跑到 A4，G0b 的 **A4 缓存场→Hi3D 角点接口**通过数值验收；完整 G0 质量结论、A0 人工符号批准、FlexiCubes 几何质量判断和编码器训练均未完成。按当前决定，先保留 G0 的质量放行判断，继续使用已固定的缓存与疑点标志做后续分析。

## 固定样本和已完成工作

| 环节 | 当前结果 | 证据 |
|---|---|---|
| G0 选样 | 从 ObjaverseXL GitHub 原始 mesh 固定 50 件：20 个拓扑闭合候选、30 个非闭合；50/50 可加载 | [准备记录](g0-preparation-2026-10-29.md)，远端 `manifest-balanced50.json` |
| A0 自动审计 | 12 个单体 `candidate`、8 个多连通 `needs_review`、30 个 `excluded_nonclosed`。最后一类只从 A0 闭合符号真值基准排除，并未从 A1–A4/G0b 排除。正式 A0 真值尚未生成 | [A0 记录](g0-a0-audit-smoke-2026-10-29.md)，远端 `a0-sign-audit-balanced50-v3.json` |
| A1/A2 初始场 | R128、CPU，同源查询，50/50 双缓存及回读成功。A2 相对 A1 的逐资产 Chamfer：14 件改善、20 件变差、16 件持平，不能宣称整体质量提升 | [A1/A2 报告](g0-a1a2-batch-r128-2026-10-29.md) |
| A3 几何拟合 | R128、GPU、30 步，50/50 成功；相对 A2 的同一 classic MC 提取器，39 件 Chamfer 改善、11 件回退；无倒置穿越 cell | [A3 报告](g0-a3-batch-r128-2026-10-30.md) |
| A4 一致深度渲染拟合 | R128、4096 图像、200 步，50/50 成功；相对 A3 的同一 classic MC 提取器，39 件 Chamfer 改善、11 件回退。16 件资产共 42 个倒置穿越 cell，仍带疑点标志 | [A4 报告](g0-a4-batch-r128-render4096-2026-10-30.md) |
| G0b 迁移接口 | 50/50 A4 场打包、回读和两种提取路径成功；场与位移回读精确，共享点最大误差 `4.77e-7`，坐标最大误差 `2.98e-8`；在同一 classic MC 中顶点与面索引完全一致 | [G0b 报告](g0b-a4-r128-interface-2026-10-30.md)，远端 `audit-summary.json` |

A1/A2 的 Lewiner 提取器指标不能直接与 A3/A4 的 classic MC 指标当作同一提取器消融。A3、A4 使用配对的同提取器比较。完整优化目标是成对的有符号场 `s` 与位移 `delta`，**不是严格欧氏 SDF**。上述运行成功表示生成和接口可执行，不等于每件 mesh 的目标质量合格。

## G0b 到底验证了什么

G0b 绕过神经网络，把 A4 唯一中心点的 `s`/`delta` 按固定 `cell_indices` 打包为 Hi3D 每 cell 的 8 个角点；显式重排两套角点顺序，并保持 `(ijk+0.5+delta)/R-0.5` 坐标约定。回读后的字段、共享点和坐标误差均低于 `1e-6`。同一场再走同一 classic MC 提取器，50 件的输出逐面、逐顶点一致。因此**缓存场到 Hi3D 提取接口的迁移成功**。

把提取器改为 Hi3D FlexiCubes 后，对原 mesh 的 Chamfer 在 41/50 件变大、9/50 件变小；逐资产差值中位数为 `+0.000235`（原 mesh 坐标单位）。回退较大的清单 index 为 **7、46、11、40、0、27、39、24**；index 45 的边界边数从 104 增至 114。此差异属于提取路径的对照结果，不能归因于尚未运行的 VAE。是否保留 FlexiCubes 及其几何质量是否可接受，仍需结合局部结构可视复核；目前以 classic MC 作为 A4 表示基线。

## 尚未完成与当前边界

1. **G0 质量判断暂缓。** 20 个闭合候选的实体符号、自相交及组件语义尚未人工复核；8 个多连通体继续保留 `needs_review`。正式 A0 真值和 A0–A4 统一对照未完成。
2. **A4 疑点待检查。** 16 件资产存在倒置穿越 cell，11 件相对 A3 的 Chamfer 回退；这些结果仍保留于缓存，不应覆盖或默认为合格目标。
3. **G0b 仅是无网络接口验收。** 尚未实现新的 encoder、输出头、倍率/latent 适配和训练路径；没有网络重建、训练梯度、checkpoint 或质量收益数据。流式 decoder 仍是推理路径，未验证新模型的流式一致性 G0c。
4. **数据和运行条件未定。** 正式训练划分、权重路径及哈希、GPU 小时预算和旧 DiT 保留策略尚未确定。现有 50 件是表示验证集，不是正式训练集。

当前最直接的后续分析是检查 G0b 回退较大编号的 MC/FlexiCubes 网格，确认是否存在薄结构或局部缺失，并与 A4 倒置 cell 标志交叉看；G0 是否通过、是否修订 A4/提取器和何时进入训练，再依据这些证据决定。

## 位置与版本

| 项目 | 路径或版本 |
|---|---|
| 本地工程 | `/Users/changle/code/3dgen/hi3d-vae-field`；当前 HEAD `28dc8dcf844c2155be0dde4473eb59f2341e050a`，G0/G0b 改动尚在工作区 |
| 远端工程 | `/data/hi3d/hi3d-vae-field`；当前 HEAD `9c72ea6bba8f345e6f7770325298da57ec9bfe25`，Git 工作区干净 |
| 固定 vae3d 源码 | `/data/hi3d/g0-prep/vae3d`，提交 `422e79988ccda06ef0995bf7768cd05a8894a124` |
| G0 样本清单 | `/data/hi3d/g0-prep/manifest-balanced50.json` |
| A1/A2、A3、A4 缓存 | `/data/hi3d/g0-prep/g0-a1a2-r128-cpu-v1/`、`g0-a3-r128-gpu-v1/`、`g0-a4-r128-render4096-v1/` |
| G0b 结果 | `/data/hi3d/g0-prep/g0b-a4-r128-v1/`；汇总 `summary.json`，回读审计 `audit-summary.json`，逐资产含 `mc_raw_mesh.ply` 和 `flexicubes_raw_mesh.ply` |
| 实验方案与交接 | [v0.3 方案](../experiment-plan-v0.3.md)、[HANDOFF](../HANDOFF.md) |

本报告依据阶段报告、2026-10-30 本地 Git 状态及远端 Git/审计文件只读核对整理。未创建提交或推送；后续提交须先向用户确认，push 由用户操作。
