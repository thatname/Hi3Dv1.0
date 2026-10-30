# G0 四路线执行记录（2026-10-30）

状态：**固定 50 件 × 四路线的 A3/A4 200/200 成功；A1/A2 双提取器共用场缓存已生成。质量决策和高分辨率训练均未完成。** 本报告仅记录本轮实际证据，协议见 [四路线实施规格](../g0-four-route-implementation-spec.md)。所有新 GPU 作业只用物理 GPU 7（`CUDA_VISIBLE_DEVICES=7`，进程内 `cuda:0`），没有申请或占用其他 GPU。代码和文档尚未提交、推送。

## 版本与缓存

- 本地与远端同步前 HEAD：`1c08ea9a642d7712728c984cf2c042ac47e44f20`；新实验代码在工作区，远端同路径复制，未提交。固定 manifest `/data/hi3d/g0-prep/manifest-balanced50.json`，沿用 A1/A2 R128 phase=0.5、原坐标恢复。远端仅写入 `/data/hi3d` 下。
- v1 缓存：`/data/hi3d/g0-prep/g0-four-route-r128-v1/`。四件 × 四路中 13/16 成功。index 7 的 `mc_smooth` A4、`flexicubes_smooth` A3，index 30 的 `mc_smooth` A4 因非有限梯度失败。日志保存为 `index<编号>-<路线>.log`。这是失败证据，不可作质量比较集合。
- 原因定位：smooth 法线角度计算在退化边上出现 `atan2(0,0)` 未定义梯度。`four_route_normals.py` 改为在计算前掩蔽退化角；远端 CPU 的一退化面例对顶点反传有限。修复版单独写入 v2 缓存 `/data/hi3d/g0-prep/g0-four-route-r128-v2/`，不混用 v1。
- v2 固定小样本命令：`CUDA_VISIBLE_DEVICES=7 ... python -m experiments.vae_sdf.run_four_route_batch --manifest /data/hi3d/g0-prep/manifest-balanced50.json --a1a2-summary /data/hi3d/g0-prep/g0-a1a2-r128-cpu-v1/summary.json --a0-audit /data/hi3d/g0-prep/a0-sign-audit-balanced50-v3.json --vae3d-root /data/hi3d/g0-prep/vae3d --output-root /data/hi3d/g0-prep/g0-four-route-r128-v2 --indices 48 5 7 30`；实际启动时 `PYTHONPATH` 包括 G0 vendor 与固定 vae3d，`LD_LIBRARY_PATH` 指向验证 venv 的 CUDA runtime。状态文件 `batch-status.json`，完整命令与日志由其逐项记录。
- v2 index 7 `mc_smooth` 单项耗时约 1200 秒，CPU 单核接近满载；v3 减少同一步重复法线计算后按用户暂停指令中止，完成项保留，运行项标为 `interrupted`。此后实现 `corner_fans_gpu`，在 GPU 上对有向流形边配对并做角点连通分组，保留非流形和绕序不一致处的扇区隔离。`check_four_route_fans.py` 对四个合成拓扑例及 index 7 的 207444 面网格与 CPU 参考逐角验证：分组完全相同，法线最大差 `1.19e-6`，梯度最大相对差 `1.54e-7`；大网格单次分组 CPU `1.636s`、GPU `0.00431s`，约 380 倍。小网格 GPU 启动开销更高。
- v4 `/data/hi3d/g0-prep/g0-four-route-r128-v4/` 用 GPU 扇区实现跑完 index `48,5,7,30` 的 16/16 路线，A3/A4 均成功，总耗时 `307.9s`；index 7 `mc_smooth` 单项 `28.8s`，相对 v2 的约 `1199.9s` 大幅缩短。v4 占用约 `224MB` 磁盘，未使用额外 GPU。具体逐项日志和缓存见 v4 的 `batch-status.json`。
- v5 `/data/hi3d/g0-prep/g0-four-route-r128-v5/` 在 GPU 7 上顺序完成固定 index `0..49` ×四路线，`batch-status.json` 为 200 success、0 failed。`paired-summary.json` 校验固定协议，在四路共同成功的 50 件上配对汇总。A1/A2 同一输入场分别用 Classic MC 与 FlexiCubes 提取，50 件的共享缓存写于 `shared/`；两种法线模式复用这两阶段的网格。`post-status.json` 中初次总损失有限差分探针失败；修订后位置项探针通过，结果见 `post-finite_difference-retry.log`。
- 2026-10-31 清理：v1–v4 的 `target.npz` 与 `raw_mesh.ply` 共 194 个、597347288 字节已删除；各版本 `batch-status.json`、逐项日志、`resolved_config.json`、`trace.json`、`audit.json`、`metrics.json` 均保留，供失败和性能对照。v5 正式缓存未删，仍有 400 个 A3/A4 `target.npz`。本地 8765 临时预览服务已停止；其 14 MB 可再生成的预览缓存已删除，若重新打开页面需重新生成预览，因此首次加载会变慢。

## 已做验证与限制

- GPU 梯度 smoke：四组合对 s/delta 有有限非零梯度，四组合一轮 A3 和 A4 能生成网格。`mc_flat` 一轮 A4 与固定 vae3d 原实现字段和损失精确一致。index 48 的四组合正式 30/200 步和 4096 px 均成功；`flexicubes_smooth` 的 A4 Chamfer L1 从 A3 的约 `0.0004668` 降至约 `0.0003751`，但这只是单资产结果。
- v1 曾有 smooth 非有限梯度，不能据成功资产推断全 50 通过。v3/v4 已完成同一步共享角点法线及 GPU 扇区计算；v3 的 128px `mc_flat` 一步回归对固定 vae3d 字段、loss 和初始 held-out 误差均为零差。v4 四件 16/16 仅证明该小样本可运行，不代表全 50 或质量放行。
- 固定拓扑下四路线对 s/delta 的位置项有限差分相对误差均低于 0.9%；总损失对两者的梯度已检查为有限且非零。先前对包含法线归一化的总损失作 float32 密集差分不稳定，该失败不能断定提取器梯度错误。四路 A4 Chamfer L1 中位数依次为 MC flat `0.001699`、MC smooth `0.001909`、FlexiCubes flat `0.001756`、FlexiCubes smooth `0.002062`；A4 相比各自 A3 改善数依次为 38、44、37、45/50。倒置穿越 cell 资产数依次为 13、34、8、15。当前统计不支持直接放行 smooth+FlexiCubes。尚欠两种法线评测器统一重算、自交检测、driver 显存、人工质量复核及 R1024/2048 资源实测；reserved 显存各路峰值约 5.41、7.57、5.51、7.57 GB。

## 预览

- 新本地 `/compare` 为 2×5 单 WebGL2 画布；旧同场 G0b 页面保留 `/compare/legacy`。目录 API 按固定 index、R128/phase0.5、route、stage 读取，不接受浏览器自定义路径。A1/A2 两行从相同场分别用各自提取器生成；A3/A4 按对应路线配置取独立缓存。index 7 的目录 API 返回两行 A1–A4 共 16 个 `ready` 状态。
- 浏览器实测 index 7：10 格布局、两行 A1–A4 快速预览全部加载，界面显示各网格原面数、6000 面预览数及原精度 Chamfer；A3 隐藏后两行同时消失；切换拟合法线后 A3/A4 改为对应 flat 缓存，A1/A2 保持。当前本地 `127.0.0.1:8765/compare?index=7` 已指向 v5。窄窗口下每格过小，视觉质量判断应在常规桌面宽度或更少可见列进行。全 50 的视觉复核及代表样本原精度加载仍待完成。

## 下一步

1. 完成双行 UI 的代表样本原精度加载和 50 件目录抽检；快速预览首次加载仍需远端简化，可能较慢。
2. 按四路线规格补统一法线评测、自相交检测和人工重点 mesh 复核。人工 A0 符号批准与最终训练路线选择仍需单独进行。
3. GPU7 上实测 R1024 的目标生成与训练资源，再决定 R2048；当前 R128 phase=0.5 的结论不得外推。
