# Hi3Dv1.0-modified 基线分析

日期：2026-10-29。工程现名：hi3d-vae-field。

## 结论

以修改版 commit `5e5622d0baea953e48bde31607936f1ac1036be6` 为实验基线，保留其完整 Git 历史。相对原基线 `13657ad09afaa92e350b461a1cd66399e44a59c4`，共 3 个提交、13 个文件变化、1438 行新增、162 行删除。修改主要面向高分辨率推理效率，并未提供 VAE 训练系统。

## 新增能力

| 提交 | 变化 | 对实验的用途 |
|---|---|---|
| b13257a | Windows spconv CUDA12 DLL 搜索兼容 | Windows 环境辅助，Linux 训练不依赖此分支 |
| 8ce71be | 主机内存驻留、多 GPU 流式解码、分块提取、补洞和连通分量处理加速 | 高分辨率验证候选；需实测主机 RAM、吞吐和几何等价性 |
| 5e5622d | 2048 档、显式递增 pass ladder | 后续生成验证可选 512→1024→2048 或单独 2048 |

2048 档使用 256³ 稀疏支撑，复用 transformer_1024 并调整旋转位置尺度，不是新增训练得到的 2048 模型。README 中的 RTX4090/12GiB 为该修改版文档声明，本次没有复现实测，不能用于估算 VAE 训练显存。

## 对训练方案的重要约束

1. `sparc3d/models/autoencoders/streaming.py::StreamedDecoder.decode` 使用 `@torch.inference_mode()`；`sparc3d/ops/tiling.py::run_chunks` 的工作线程也进入 inference_mode。仅移除一个装饰器不能获得可训练系统，还涉及主机缓存、全局归一化统计、模型副本及梯度同步。首轮训练采用普通可微 decoder 路径。
2. 流式执行不是 DDP 训练；模型阶段副本不会自动同步训练更新。用新 checkpoint 或改输出头后必须重新建立 streamer，不能复用旧副本。
3. 原 `SparconvVAE`、PTv3 架构、角点属性定义和 Gaussian 缺口未变化。仍需迁移 vae3d encoder/posterior、倍率8/latent32、GT 支撑监督、连续场头、训练器与评测。
4. 流式 `_conv_out` 及 tiled extractor 仍假设原 SDF sign + distance + deformation 布局。新 signed-tanh/半体素坐标适配必须同时覆盖普通和流式路径；完成前仅将流式用于原模型回归。
5. `set_decoder_skip_points` 提供显式阈值接口，但流式残差代码本身仍保留超限跳过逻辑。demo 启用流式时将阈值抬高并关闭 feature ceiling；单独调用 enable_streamed_decoding 不等于自动取消这些保护。训练不得静默跳层。
6. 文档所述流式与普通路径等价需要相同 skip 策略、精度、模型状态和坐标；全局 group norm 聚合顺序及符号阈值附近误差应单独检查。
7. tiled extraction 采用 halo 和跨块顶点标识，仍需测试边界接缝、重复/缺失面、模糊 case 及共享角点平均。新连续头不能沿用旧硬符号覆盖。
8. postprocess 的 target_face_count=None 仅取消简化，仍会过滤组件并重新居中缩放。因此原始几何评测应直接捕获提取结果，不能把该选项当作完全关闭后处理。

## 实验增补

- G0/G0b/G1/G2 的目标流程与小规模训练保持 v0.2 主设计，开发基线换为当前修改版。
- 新增 G0c：在能容纳普通路径的小输入上，对同一 latent 比较普通解码、单卡流式、多卡流式；同样的精度/skip阈值，先比较共享角点字段，再比较未后处理 mesh。测试至少两种 chunk 预算和跨 tile 边界形状；数值容差按浮点误差预先固定，阈值导致拓扑变化单列失败。
- 输出头迁移完成后重复 G0c；在此之前不能用旧 streamer 验证新模型。
- 2048 放在后续高分辨率推理验收，首轮 VAE 仍从 R=128/256 做显存与梯度 smoke，不能按推理成功推断训练可行。
- 远程预算补充 CPU 峰值内存、主机锁页内存、PCIe 传输时间及并行 GPU 利用率；流式降低显存需求但转移了内存与传输压力。

## 迁移与验证范围

Hi3Dv1.0-modified 的全部文件（包括 .git）迁入现有 hi3d-vae-field；原目录只在内容全部转移后移除空目录。目标目录原有 v0.2 文档保留。包名 sparc3d、模型下载标识和上游链接保留，工程目录名更改不要求改 Python import 或下载远程模型的标识。

本次仅工程整理与静态分析：检查文件内容校验和、Git HEAD、Python AST 语法、相对文档链接。没有训练实现、CUDA 推理、反向传播或多 GPU 等价性实测。以下为后续必测，不是已通过的测试：encoder/decoder 梯度、流式一致性、分块接缝、2048 质量和性能。
