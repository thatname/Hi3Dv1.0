# G1 双样本恢复训练与重建评测

日期：2026-10-30。输入为固定 50 清单中的 48、5 号 A4 成对优化场与位移，R128，采样相位 0.5。训练仍为 GT 子支撑、`β=0`；评测时额外让 occupancy 头预测三级子支撑。粗层 latent 坐标仍由 GT cell 确定，因此这不是从任意 latent 自动生成几何的完整测试。

## 代码与运行

- 独立 [G1 模型](../../experiments/vae_sdf/g1_model.py) 增加预测子支撑解码，按原 Hi3D 阈值 `logit>0` 保留子点；超过 2,000,000 子点或全空时明确记录失败，不静默裁剪。
- [训练入口](../../experiments/vae_sdf/train_g1.py) 增加严格配置一致性检查、模型与优化器恢复、CPU/CUDA/NumPy/Python 随机状态恢复及原子化 checkpoint/trace 写入。第 50 步生成 checkpoint，随后恢复训练到第 500 步，`rng_restored=true`；第 250、500 步另存固定快照。首次恢复因 CUDA RNG 字节状态被 checkpoint `map_location` 搬至 GPU 而失败，修正为恢复前放回 CPU；失败前没有创建输出目录，也未修改原 checkpoint。
- [评测入口](../../experiments/vae_sdf/evaluate_g1.py) 严格读取训练权重与 A4 目标身份；对同一 checkpoint 分别计算 GT/预测子支撑下的网格，使用 vae3d classic MC，原坐标系、原采样相位，**不做清理、简化或重新归一化**。2048 个双向确定性采样点计算 Chamfer-L1、F-score 和法向等；记录场/位移 MAE、预测 cell precision/recall、空网格及 raw PLY。目标 A4 网格误差与网络重建误差分开报告。
- 源码提交：本地与远端 HEAD `99538267910d354e1c2441d0ac489fd9a096933f`，此轮脚本仍未提交。脚本 SHA256：`g1_model.py` `ee9f8c97c0b90c07426fc93ea5c52cfa3fb7767c4ad87133eef2712a46bf48b9`，`train_g1.py` `e17b3e440c9f8e2eb20db6472571320090cf00b4e34218ea0f0a24cac82f1bf5`，`evaluate_g1.py` `4f7a59145442c2d66af214bbb8c48cd6c87d49829a91af7a2ee4e4e59f08eaec`，本地与远端一致。固定 vae3d 源码 `422e79988ccda06ef0995bf7768cd05a8894a124`。

主要远端输出：

| 内容 | 路径 |
|---|---|
| 双样本 1–50 步训练、checkpoint | `/data/hi3d/g0-prep/g1-a4-overfit-48-5-step50-v1/` |
| 恢复后 51–500 步、250/500 快照、最终 checkpoint | `/data/hi3d/g0-prep/g1-a4-overfit-48-5-step500-v1/` |
| 第 50/250/500 步配对评测、raw PLY | `/data/hi3d/g0-prep/g1-eval-overfit-48-5-step{50,250,500}-v1/` |
| 先前 50 资产 100 步 checkpoint 的 48 号基线 | `/data/hi3d/g0-prep/g1-eval-50run-step100-48-v1/` |

训练命令中第一次使用 `--indices 48 5 --steps 50 --save-every 50`，第二次使用 `--resume /data/hi3d/g0-prep/g1-a4-overfit-48-5-step50-v1/checkpoint.pt --indices 48 5 --steps 450 --save-every 250`；其余固定 `--manifest /data/hi3d/g0-prep/manifest-balanced50.json`、`--a4-summary /data/hi3d/g0-prep/g0-a4-r128-render4096-v1/summary.json`、`--weights /data/hi3d/hi3d-validation/models/Hi3Dv1.0/vae/model.safetensors`、`PYTHONPATH=/data/hi3d/g0-prep/vendor-full:/data/hi3d/g0-prep/vae3d:/data/hi3d/hi3d-vae-field`。完整参数与哈希见各 `resolved_config.json`，逐步值见 `trace.json`。

## 配对实测

以下 Chamfer-L1 均为原 mesh 坐标单位。表中“对目标”表示网络重建网格对 A4 目标提取网格；“对原 mesh”表示最终误差。目标 A4 自身对原 mesh 的 Chamfer 为 48 号 `0.000334`、5 号 `0.001293`，不应和网络压缩误差合并成同一个指标。

| 资产 | 步数 | GT 支撑：对目标 | GT 支撑：对原 mesh | 预测支撑：对目标 | 预测支撑：对原 mesh | 预测 cell P/R | GT 支撑场 MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 48 | 50 | 0.001629 | 0.001536 | 0.001991 | 0.001809 | 0.943 / 0.959 | 0.515 |
| 48 | 250 | 0.000608 | 0.000775 | 0.000611 | 0.000776 | 0.998 / 1.000 | 0.417 |
| 48 | 500 | 0.000511 | 0.000720 | 0.000510 | 0.000721 | 1.000 / 1.000 | 0.377 |
| 5 | 50 | 0.019338 | 0.019748 | 0.023726 | 0.024322 | 0.947 / 0.967 | 0.436 |
| 5 | 250 | 0.004101 | 0.004597 | 0.004097 | 0.004601 | 0.999 / 1.000 | 0.342 |
| 5 | 500 | 0.002464 | 0.002926 | 0.002483 | 0.002934 | 1.000 / 1.000 | 0.309 |

两资产的 GT/预测支撑均在三个 checkpoint 产生非空 raw mesh；从 50→500 步，训练损失、场误差和网格 Chamfer 都下降。第 500 步预测子支撑的 cell 集与 GT 一致，是**这两个训练样本上的过拟合结果**，不代表其余 48 个资产或独立数据。第 500 步对原 mesh 的误差仍高于 A4 目标本身：48 号约 2.2 倍、5 号约 2.3 倍，压缩损失还未消除。第 50→500 步是同一训练链：模型、优化器和 RNG 已恢复；没有改变目标缓存或源码结构。后 450 步用时约 104.6 秒，训练进程峰值 PyTorch CUDA allocated 约 3.84 GB；不含评测和外部进程显存。

## G1 状态与下一步

**通过：**GT 支撑下的可微训练、严格恢复、两资产长期损失下降、同一 checkpoint 的 GT/预测子支撑解码及原坐标 raw mesh 配对评测。**尚未通过：**8–16 或 50 资产的预测支撑重建、固定验证集、KL、采样 posterior 与泛化。下一步应固定 50 资产内的训练/验证划分，在 8–16 个资产上继续训练并同时评测未参与训练的资产；若预测支撑在新资产上回退，应单独修 occupancy 路径。现有 50 样本是刻意挑选的 G1 小数据集，仍不能代替 G2 的 100–500 资产试验。
