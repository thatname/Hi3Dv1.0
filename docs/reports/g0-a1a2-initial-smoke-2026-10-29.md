# G0 A1/A2 初始目标小样本记录（2026-10-29）

验证类型：本地静态检查、远端 CPU 小样本实测与参考实现数值对照。**这不是完整 G0 评测，也没有训练 VAE。**

## 实现与产物

- 本工程新增 `experiments/vae_sdf/initial_fields.py`，复用固定版 vae3d 的 `Queries` 与 98 方向采样，在同一次查询中分别保留背壳前 A1（`B>49` 取负）与背壳后 A2（25% 投票门控）稠密场。没有用阈值不同的 legacy `ue_grid` 冒充 A1。
- `prepare_initial_targets.py` 提供 `--stage A1|A2|both`、分辨率、射线/距离后端及 `--verify-reference`。按源文件哈希、manifest 哈希、vae3d commit、代码哈希、参数和阶段生成独立缓存身份；已有同名缓存会拒绝覆盖。
- 每阶段保存 `field.npy`、`target.npz`（唯一中心点 `[s,delta]`、cells、cell_indices，s/delta 均为细体素单位，delta 初始为零）、原坐标系的 `raw_mesh.ply`、`audit.json` 和 `resolved_config.json`。`both` 模式使用 A1/A2 近表面及符号交叉支撑的并集，膨胀两层后构造完整 cells；原始场和提取 mesh 不做补洞、简化或组件过滤。当前 raw mesh 提取器是 `skimage.measure.marching_cubes`（Lewiner），后续 G0 跨阶段评测须固定提取器并与 vae3d classic MC 的差异分开记录。
- 代码与小样本产物位于远端独立的 `/data/hi3d/g0-prep/`，远端 `/data/hi3d/hi3d-vae-field` Git 仓库未修改。源 mesh 仍在原目录。

## 实测

固定输入：`/data/hi3d/g0-prep/manifest-balanced50.json` 的 index 20，非闭合样本 `0000eec145...`；vae3d commit `422e79988ccda06ef0995bf7768cd05a8894a124`；R=128、sampling_offset=0.5、CPU 射线/CPU 距离。产物根目录：`/data/hi3d/g0-prep/targets-a1a2-smoke-v1/0000eec1454799ccfe5054ba1ff710ef3829e4a61bb1e186310045b69359fd82/R128/`，A1 与 A2 各有独立的哈希命名目录。

| 检查 | 结果 |
|---|---|
| A2 对固定版 `back_shell_grid` | 最大绝对误差 0，逐元素相等 |
| A1 → A2 | 372 个场采样值改变，其中 18 个由非负变负；与背壳审计一致 |
| 共同支撑 | 141747 个完整 cells、158912 个唯一中心点；与 vae3d `complete_fitting_support` 输出一致 |
| 阶段 raw mesh | A1 19748 顶点 / 39480 面；A2 19776 顶点 / 39536 面 |
| 回读 | 坐标、cell 索引、4 通道特征、细体素单位、零 delta、共享支撑和 raw mesh 面数通过 |
| 单独 A1 | 同一资产 R32 `--stage A1` 成功，仅导出 A1；结果位于 `/data/hi3d/g0-prep/targets-a1-only-smoke-v1/` |

R128 同源 A1/A2 查询约 17.37 秒，额外参考实现对照约 15.20 秒；这是单资产 CPU 计时，不是 50 资产或 A3/A4 的成本估计。R32 的一个闭合和一个非闭合样本均与参考 A2 逐元素相等，但该分辨率没有触发背壳新增负值点。

实际命令：

```bash
PYTHONPATH=/data/hi3d/g0-prep/vendor-full:/data/hi3d/g0-prep/vae3d:/data/hi3d/g0-prep \
  /data/hi3d/hi3d-validation/.venv/bin/python -m experiments.vae_sdf.prepare_initial_targets \
  --manifest /data/hi3d/g0-prep/manifest-balanced50.json --index 20 \
  --vae3d-root /data/hi3d/g0-prep/vae3d \
  --output-root /data/hi3d/g0-prep/targets-a1a2-smoke-v1 \
  --stage both --resolution 128 --raycast-backend cpu --distance-backend cpu \
  --verify-reference

PYTHONPATH=/data/hi3d/g0-prep/vendor-full:/data/hi3d/g0-prep/vae3d:/data/hi3d/g0-prep \
  /data/hi3d/hi3d-validation/.venv/bin/python -m experiments.vae_sdf.check_initial_targets \
  --a1 /data/hi3d/g0-prep/targets-a1a2-smoke-v1/0000eec1454799ccfe5054ba1ff710ef3829e4a61bb1e186310045b69359fd82/R128/A1-4a2340e97c4a \
  --a2 /data/hi3d/g0-prep/targets-a1a2-smoke-v1/0000eec1454799ccfe5054ba1ff710ef3829e4a61bb1e186310045b69359fd82/R128/A2-b3268e0c07cb
```

## 尚未完成

1. 当前只有 A1/A2 的小样本目标缓存；尚未对固定 50 个资产运行批量缓存、失败分母和指标。运行前需确认分辨率、后端、提取器与输出预算。
2. A0 只可用于人工确认符号可靠的闭合子集；目前 20 个仅为 `is_watertight` 候选。
3. A3/A4 的 Kaolin、nvdiffrast、Warp 与 CUDA 前后向尚未验证，几何/渲染拟合入口、阶段缓存和共同支撑跨阶段协议尚未实现。
4. 未运行 G0 的 Chamfer、F-score、薄结构、轮廓、深度或资源汇总；单样本 raw mesh 顶点/面数差异不能解释为质量收益。
