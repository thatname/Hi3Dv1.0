

## Codely Structured Memories

### User

### Feedback
- [2026-10-31 10:54:06] 远端实验约束（用户明确要求）：除 /data/hi3d/hi3d-vae-field 仓库外，任何脚本/产物都必须放进单独一个独立目录（如 /data/hi3d/g1/analysis-0-7-v1/），防止改坏；v5 缓存与既有训练目录只读；GPU 只用物理 7（CUDA_VISIBLE_DEVICES=7，进程内 cuda:0），用其他卡前先问用户。原因是曾改出问题的风险。
- [2026-10-31 10:54:08] 用户不需要助手主动编写/更新交接文档或 HANDOFF（2026-10-31 明确纠正"不是让你写交接文档"）：文档由用户与另一个模型维护，助手只做阅读、分析、实验与代码改动。仅在用户明确要求时才写文档。

### Project
- [2026-10-31 12:57:17] G1 诊断结论（已用实验修正）：0/7 号差不是几何/目标（目标 Chamfer/包围盒对角为 1.7e-4 与 7.9e-4）也不是表示能力或损失盲区，而是**欠收敛 + 泛化**。"近表面正偏 + 内部塌缩 + 薄壁消失"是欠收敛症状：单资产各 400 步后 0 号偏置 +0.73→−0.001、穿越召回 0.32→0.93、Chamfer 1.649→0.078；7 号 Chamfer 3.52→1.55（目标自身 1.57）。8 资产 4000 步（1911 秒）后训练侧 8/8 收敛（穿越召回 0.91–0.97、内部比例匹配、两次前向漂移 1e-5），但验证侧 0/30 只有 0.66/0.72 且 30 号 3000 步后轻微回退 → 瓶颈是泛化，不是步数。总场 MAE 仍不是有效闸门（73–79% 误差在 |s|>2 远场带，与网格无关），应看穿越 cell 召回/精度、内部体积比、近表面有符号偏差。
- [2026-10-31 12:57:19] G0/G2 缓存协议约束（2026-10-31 核实）：A1/A2 与 A3/A4 的缓存键都含 manifest_sha256，因此**换任何 manifest 都会让既有缓存全部失效**（例如从 balanced50 扩到 100 件需重算 100 件 A1/A2 约 37 分钟 CPU，加选定路线 A3/A4 约 35 分钟 GPU7；v5 实测 200 次路线运行 1.18 小时≈21 秒/次）。扫描池已无新闭合候选（89 个新件全为非闭合），扩到 100 件后为 20 闭合/80 非闭合，必须在报告中按 topology_candidate 分组，且扩展件的 A0 行标 provisional（未跑符号探针）。

### Reference
- [2026-10-31 12:57:23] G1 场诊断与闸门工具索引：/data/hi3d/g1/analysis-0-7-v1/（analyze_fields.py 分段场统计含近表面有符号偏差、support_analysis.py 多余支撑、support_consistency.py 两次前向漂移、reference_export.py 任意 index/checkpoint 导出、gate_report.py 一次性出闸门表并写 gate.json）。G2 输入构建脚本 /data/hi3d/g1/make_g2_inputs.py，产物在 /data/hi3d/g1/g2-prep/（manifest-g2-100 / load-audit-g2-100 / a0-audit-g2-100 / split-g2-100-v1，80 训练/10 验证/10 测试，val 锚点 0 与 30）。本地监控：`python3 -m experiments.vae_sdf.view_g1 --port 8767 --no-browser --run <远端run目录>`，页面 http://127.0.0.1:8767/ ，未启动的训练目录显示 waiting。远端可用 python 为 /data/hi3d/hi3d-validation/.venv/bin/python（numpy 1.26.4 + torch 2.5.0+cu124）；跑仓库模块需 PYTHONPATH=/data/hi3d/hi3d-vae-field:/data/hi3d/g0-prep/vendor:/data/hi3d/g0-prep/vae3d，评估类脚本（涉及 open3d）要用 vendor-full，并设 LD_LIBRARY_PATH=<venv>/lib/python3.10/site-packages/torch/lib，加 -B 避免写入仓库 __pycache__。

