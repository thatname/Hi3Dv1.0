# G1/G2 执行交接（2026-10-31）

此文档是新会话的执行入口；当前事实见 [HANDOFF](HANDOFF.md)，目标与评测定义见 [实验方案](experiment-plan-v0.3.md)，四路线协议见 [实施规格](g0-four-route-implementation-spec.md)。**尚未开始新路线的 G1/G2 训练；旧 R128/A4 G1 权重已清理，不能续训。**

## 新会话第一轮只读检查

在本地仓库执行 `git status --short --branch`、`git log -3 --oneline`、`git remote -v`。远端执行 `cd /data/hi3d/hi3d-vae-field && git status --short --branch && git log -3 --oneline && git remote -v`。本地是代码与文档的提交源；用户负责 push；**不要从本地复制源码到远端**。用户 push 后，在远端先核对 remote、当前分支和未提交改动，再通过 Git 拉取/合并对应提交；若远端有改动，先比较并保护，不能直接覆盖或重置。训练前记录两端 HEAD 与工作区状态，确保远端运行源码对应已记录的提交。

固定输入：`/data/hi3d/g0-prep/manifest-balanced50.json`；正式四路线缓存 `/data/hi3d/g0-prep/g0-four-route-r128-v5/`，其中 50 资产 × 4 路 A3/A4 `target.npz` 共 400 个。**用户选择 `flexicubes_smooth`** 用于后续目标生成与最终提取；人工 mesh 已复核，但逐资产记录未归档。此前四路线 200/200 运行成功只说明可执行，质量审计中的回退与倒置 cell 标志仍须随资产保留。仅使用物理 GPU 7（进程配置 `CUDA_VISIBLE_DEVICES=7`）；需其他 GPU 时先向用户说明。

## G1：同一批 50 件上打通并验证选定路线

1. **锁协议与适配入口。** 核对 v5 的 `flexicubes_smooth` A4 `target.npz` 的 s/delta、原坐标恢复、R、相位、源哈希、法线模式和提取器。当前 `experiments/vae_sdf/train_g1.py` 只接受旧 A4 `summary.json`，并硬编码 R128/phase0.5；不能直接将 v5 路线目录塞给它。先实现独立的目标读取和训练配置，或受控扩展入口，明确缓存 schema、资产索引和输出提取器。Hi3D decoder 骨干热启动，encoder 与新输出头从头初始化；不复用旧 G1 checkpoint。新 run_id 与旧目录隔离。
2. **相位与分辨率门槛。** v5 是 R128/phase0.5 历史协议。先用少量资产做 phase=0 与 `Point.matrix` 审计，确认相同物理坐标、s/delta 单位、共享点与 GT/预测支撑坐标一致；另建缓存，不修改 v5。再用 R256/512 各取简单体和复杂体量测目标生成时间、细层点/cell 数、CPU RAM、磁盘和 GPU allocated/reserved/driver 显存；完成一次网络 forward/backward。R1024 先做单资产目标与训练 smoke，能稳定反传、保存、恢复后再扩大；R2048 留待可微分块和边界一致性实测。
3. **训练前准备监控。** 为新 run_id 预留 `/data/hi3d/g0-prep/` 下独立输出目录；本地先运行 `python3 -m experiments.vae_sdf.view_g1 --port 8767 --no-browser --run /data/hi3d/g0-prep/<新run_id>`，打开 `http://127.0.0.1:8767/` 并确认页面上的路径。新训练日志若改变 schema，须先更新只读监控 API。G0 预览占 8765，监控固定 8767；不要启动重复服务。
4. **阶梯训练。** 先 1 简单 + 1 复杂资产、`beta=0`、GT 支撑，检查 encoder/decoder/支撑头的有限非零梯度和损失趋势；再固定 8–16 训练/2 验证，最后覆盖固定 50 件。每级保留严格恢复证据、GT 与预测支撑 raw mesh、场/位移和支撑指标。旧 8+2 的 index 30 预测支撑边界缺口是重点回归项。实际训练若只使用场/位移损失，记录未在训练反传中运行可微 FlexiCubes；目标生成和最终 mesh 提取仍使用选定路线。
5. **G1 放行。** 新路线 50 件标签来源和配置可追溯；训练没有 NaN/静默跳层；至少两资产过拟合改善，未训练验证资产的 GT/预测支撑网格均非空且原坐标正确；相对目标的压缩误差、相对原 mesh 的最终误差分开报告。R1024 的显存、吞吐和恢复只依据实际测量决定是否扩大。质量未达标时先定位场、零面、支撑或目标问题，不以运行成功放行 G2。

## G2：扩大资产并做同预算对照

1. 先确定 100–500 件去重资产、固定 train/val/test 划分与来源哈希；同源、旋转及派生分辨率不可跨集合。保留闭合、非闭合、薄结构和疑点分层；不得将人工疑点静默删除。
2. 按 G1 锁定的分辨率、相位、归一化、架构和评测协议生成 B2/B4 配对目标。B4 使用 `flexicubes_smooth` 的 A4 成对场；B2 为 A2 目标，需明确其提取协议与 B4 比较方式。按相同资产、初始化策略、训练步数和资源预算训练；各自独立 run_id、缓存与 checkpoint。若预算不足，先做 100 件预实验，再根据覆盖率和资源决定扩至 500 件。
3. 统一报告目标对原 mesh 的表示误差、VAE 对目标的压缩误差、VAE 对原 mesh 的最终误差；都用原坐标、未清理 raw mesh，固定采样数/种子和失败分母。分别给出预测支撑与 GT 支撑、闭合/非闭合/薄结构子集、倒置 cell/自交/边界等异常，以及每资产标签时间、训练吞吐、峰值显存和缓存体积。不可将不同提取器或不同相位的 Chamfer 直接解释为纯场损失收益。
4. G2 放行需要预测支撑下的稳定重建、固定验证集的配对改善及可接受资源成本；不达标则根据误差分层回到目标、支撑或网络适配，暂不进入 G3。

## 每轮必须留下的证据

写入 `/data/hi3d/g0-prep/<新run_id>/`：完整 resolved config、源码 HEAD/dirty 状态、manifest 与源/目标哈希、GPU/依赖版本、实际启动与恢复命令、逐步 trace、失败清单、checkpoint 和 raw mesh/指标。更新 [HANDOFF](HANDOFF.md) 与对应阶段报告时注明“源码核查、静态检查、GPU 实测、浏览器实测”各自范围；没有实测不得宣称质量或 R1024/2048 可训练。远端写入范围仅 `/data/hi3d`。
