# G1 八训练样本、两验证样本与优化场诊断

日期：2026-10-30。固定 50 清单内训练编号 `48,5,35,2,20,37,44,7`，验证编号 `0,30`；资产及 manifest 哈希固定在 [split 配置](../../experiments/vae_sdf/configs/g1-eight-train-two-val-v1.json)。两验证资产未参与此轮 400 步优化。模型为随机初始化的 vae3d 结构编码器、Hi3D 预训练解码骨干和新连续头；R128、A4 成对场与位移、GT 子支撑、`β=0`。验证时分别用 GT 和预测子支撑提取原坐标 raw mesh。预测子支撑仍使用 GT 粗 latent 坐标，不能视为完全自主生成。

## 运行与证据

- 远端训练：`/data/hi3d/g0-prep/g1-a4-eight-train-step400-v1/`，`resolved_config.json` 状态 `completed`，`trace.json` 400 条；第 200/400 步与最终 checkpoint 均在该目录。400 步耗时 176.8 秒，PyTorch CUDA peak allocated 16.17 GB，不含外部进程。
- 验证：`/data/hi3d/g0-prep/g1-eval-eight-train-step200-val-v1/` 与 `g1-eval-eight-train-step400-val-v1/`。两目录 `report.json` 均为 `completed`，0/30 在两种支撑下均产出非空 raw PLY。
- 训练命令：`python -m experiments.vae_sdf.train_g1 --manifest /data/hi3d/g0-prep/manifest-balanced50.json --a4-summary /data/hi3d/g0-prep/g0-a4-r128-render4096-v1/summary.json --weights /data/hi3d/hi3d-validation/models/Hi3Dv1.0/vae/model.safetensors --split /data/hi3d/hi3d-vae-field/experiments/vae_sdf/configs/g1-eight-train-two-val-v1.json --indices 48 5 35 2 20 37 44 7 --steps 400 --save-every 200 --output /data/hi3d/g0-prep/g1-a4-eight-train-step400-v1`。环境变量 `PYTHONPATH=/data/hi3d/g0-prep/vendor-full:/data/hi3d/g0-prep/vae3d:/data/hi3d/hi3d-vae-field`。
- 验证使用同一 manifest、A4 summary、权重与 split，`python -m experiments.vae_sdf.evaluate_g1 --role val --indices 0 30 --samples 2048`，分别传入第 200、400 步 checkpoint 和上述输出目录。完整哈希、配置、每资产质量标志与指标见各 `report.json`。本地与远端仓库 HEAD 为 `99538267910d354e1c2441d0ac489fd9a096933f`；此轮 G1 脚本尚未提交。
- 独立本地训练监测页：`http://127.0.0.1:8767/`，已指向上述 400 步目录，接口返回 `completed`、400 条 trace 与可用 checkpoint。它显示训练日志，不显示评测网格。

训练前 40 步与末 40 步平均损失分别为：总损失 `1.5244→0.1284`，场 `1.1055→0.0872`，位移 `0.0096→0.0063`，支撑 `0.4093→0.0349`。这是混合训练样本的优化轨迹，不能充当验证质量。

## 验证集配对数值

Chamfer-L1 为原 mesh 坐标单位。每行都使用同一 asset、相同 2048 点评测设置；A4 目标对原 mesh 的误差与网络对 A4 目标的误差分别列出。

| 编号 | 步数 | A4 目标对原 mesh | GT 支撑：网络对 A4 | 预测支撑：网络对 A4 | GT 场 MAE | 预测 cell P/R | 预测 raw mesh 边界边 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 200 | 0.02844 | 0.55356 | 0.51117 | 0.7730 | 0.860 / 0.995 | 0 |
| 0 | 400 | 0.02844 | 0.47902 | 0.50270 | 0.8550 | 0.915 / 0.972 | 6 |
| 30 | 200 | 0.000521 | 0.007784 | 0.009706 | 0.5658 | 0.926 / 0.950 | 395 |
| 30 | 400 | 0.000521 | 0.005514 | 0.007118 | 0.5526 | 0.951 / 0.957 | 340 |

第 400 步网络对原 mesh 的 Chamfer：0 号 GT `0.47834`、预测 `0.49102`；30 号 GT `0.005980`、预测 `0.007564`。原坐标尺度相差很大：0 号参考包围盒对角线 `141.43`，30 号 `2.13`，不能直接按两行绝对值判断哪个形状更差。除以对角线后，第 400 步 A4 目标自身约为 `0.00020/0.00024`，GT 网络约为 `0.00339/0.00259`（0/30 号，对 A4 目标）。网络与目标仍有数量级上的差距。

## 问题定位与限制

1. **A4 目标生成与网络压缩需要分开看。** 0/30 号 A4 raw mesh 对原 mesh 的 Chamfer 小，而网络即使使用完全正确的 GT 细胞支撑也明显偏离 A4。因此这两个验证资产当前的主要几何误差在 G1 编码、解码及连续场拟合链路，不能仅归咎于预测 occupancy。此前 G0b 的中心点/角点回读误差最大 `4.77e-7`、同一 MC 网格精确一致，也不支持“布局迁移损坏 A4 目标”的解释。
2. **预测支撑增加额外拓扑风险。** 30 号 GT 支撑网格在第 400 步已有 24 条边界边，预测支撑为 340 条；精度/召回约 `0.951/0.957` 不等于网格拓扑正确。0 号预测支撑第 400 步有 6 条边界边。应继续单独看 occupancy，而不是只看总损失。
3. **`s` 是优化有符号场，不是严格欧氏 SDF。** 30 号原 mesh 的 A0 状态为 `excluded_nonclosed`；其 A4 提取网格可变成 watertight，但不代表原资产有可靠的实体内外真值。0 号 A4 审计有 4 个倒置穿越 cell，应保留疑点标志。
4. **场 MAE 与表面质量不是一一对应。** 0 号第 200→400 步 GT 场 MAE 上升，但网格 Chamfer 下降；场 MAE 覆盖所有缓存点，而训练场损失含近表面权重及截断单侧项。下一步诊断应分别统计近零面符号一致率、穿越 cell 一致率、近表面 `s` 误差和位移误差，并对原/A4/GT/预测四份 raw mesh 同视角查看。此轮不能据现有 MAE 判定符号方向实现错误。

结论：8 样本链路可运行，两个未训练资产的 GT/预测支撑都能生成网格，并从第 200→400 步改善；**G1 质量尚未放行**。应先完成场符号与零面定位诊断、核查 30 号边界缺口，再决定是否扩大训练或调整损失/支撑策略。追加的 8 个训练样本评测命令被自动审批拒绝，未生成该项结果；本报告不使用它。
