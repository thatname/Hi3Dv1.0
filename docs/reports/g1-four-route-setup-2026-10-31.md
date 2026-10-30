# G1 v5 路线接入记录（2026-10-31）

## 状态

本地已接入固定 50 件 v5 的 `flexicubes_smooth/a4/target.npz`，保留旧 `--a4-summary` 入口用于历史复核。新入口 `--paired-summary` 校验完整固定清单、资产与源哈希、路线/提取器/法线模式、R128/phase0.5、细体素单位、cell/唯一点布局与目标哈希。评测使用同路线 FlexiCubes，输出原坐标未经清理 raw mesh；训练仍是场、位移、支撑和有效性损失反传，没有将可微 FlexiCubes 放进训练反传。新 G1/G2 运行结果须写到 `/data/hi3d/g1/`、`/data/hi3d/g2/`，G0 缓存只读。

这一步是**源码核查与静态检查**，还没有使用新入口进行 GPU 训练或评测。旧 R128/A4 checkpoint 已删除，不可续训。R256 以上、phase=0、`Point.matrix`、G2 数据集和质量放行均未完成。

## 已核对证据

- 本地开始时 `git status --short --branch` 为 `main...origin/main`，仅 `.codely-cli/` 未跟踪；`git log -3 --oneline` 的 HEAD 为 `bf770e8`；remote 为 GitLab `origin` 和 GitHub `github`。
- 远端只读 `git status --short --branch && git log -3 --oneline && git remote -v`：`main...origin/main`，HEAD `bf770e8`，工作区干净；v5 `paired-summary.json` 与固定 manifest 存在。未通过复制源码同步远端。
- 远端只读检查 index 48：`paired-summary` schema 为 `g0_four_route_paired_summary_v1`，50 个配对 index；路线成功，缓存配置为 `flexicubes/smooth/R128/phase0.5`。A4 NPZ 形状为 `coords(39733,3)`、`features(39733,4)`、`cells(33747,3)`、`cell_indices(33747,8)`，标记 `A4/flexicubes_smooth/fine_voxel`。
- 本地 `python3 -m compileall -q` 检查新增/修改的 G1 Python 文件通过；`git diff --check` 通过；纯 Python 目标选择与目录边界小样本检查通过。本机 `python3` 无 NumPy，因此未在本机运行网络或完整目标读取。

## 远端启动顺序（待用户 push 后）

先复核远端工作区与 remote，通过 Git 同步用户推送的提交并记录两端 HEAD。核对 GPU 7、依赖及 v5 缓存身份；先开指向 `/data/hi3d/g1/flexicubes-smooth-r128-two-v1` 的本地 8767 监控，再在物理 GPU 7 启动 48/5 两资产新 run。示例命令（实际环境变量和路径在启动前复核）：

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=/data/hi3d/g0-prep/vendor-full:/data/hi3d/g0-prep/vae3d:/data/hi3d/hi3d-vae-field \
  /data/hi3d/hi3d-validation/.venv/bin/python -m experiments.vae_sdf.train_g1 \
  --manifest /data/hi3d/g0-prep/manifest-balanced50.json \
  --paired-summary /data/hi3d/g0-prep/g0-four-route-r128-v5/paired-summary.json \
  --weights /data/hi3d/hi3d-validation/models/Hi3Dv1.0/vae/model.safetensors \
  --indices 48 5 --steps 50 --save-every 25 \
  --output /data/hi3d/g1/flexicubes-smooth-r128-two-v1
```

随后检查梯度、损失和 checkpoint，严格恢复并对 GT/预测支撑分别评测；合格后扩至固定 8+2、50 件。phase=0 与 R256/512/1024 门槛按执行交接另做，不修改 v5。G2 待 G1 质量放行和 100–500 件去重、固定划分后做 B2/B4 同预算对照。
