# G0 A0 审计与候选缓存小样本（2026-10-29）

验证类型：远端真实 mesh 拓扑/符号探针审计、Open3D CPU 有符号距离小样本与缓存回读。**没有任何资产被自动批准为正式 A0 真值，也没有完成 G0 质量评测。**

## 实现

- 独立入口为 `experiments/vae_sdf/a0_signed_distance.py` 的 `audit` 和 `prepare` 子命令。输入固定 `manifest-balanced50.json` 与 `mesh-load-balanced50.json`，核对索引、ID、路径、源主文件哈希和 vae3d 固定 commit。
- 审计沿用 vae3d `load_mesh` 的三角面验证与归一化；30 个非闭合样本明确记为 `excluded_nonclosed`。对 20 个闭合候选记录 watertight、绕序一致性、有向体积、`is_volume`、连通体数和 Euler 数；用 Open3D `compute_signed_distance` 的 1/5 射线探针核对远离表面的符号、域外点和无符号欧氏距离幅值。`is_watertight` 不等于严格可靠实体：自相交以及多组件的交叉/嵌套没有被自动验证，所有结果仍要求人工复核。
- `prepare` 在 `(ijk+0.5)/R-0.5` 中心阵列用 Open3D 5 射线奇偶判定生成**完整欧氏有符号距离场**（内部负、外部正）。缓存 `field.npy` 保留世界归一化距离；`target.npz` 的 `s=clip(R*field,-4,4)` 与零 `delta` 均以细体素为单位，并保存唯一中心点、cells、cell_indices。使用 A0 本身近表面/交叉支撑、两层膨胀和完整 cell；提取的 `raw_mesh.ply` 回到源坐标系，未清理、简化或过滤组件。A0 与 A1/A2 跨阶段共同支撑仍待最终 G0 评测时统一。
- 默认不能把自动审计候选导出成正式 A0 标签。`--allow-candidate` 仅生成 `candidate_unverified` smoke 缓存。正式资格需提供人工复核 JSON，绑定 manifest/audit 哈希、资产索引/ID/源哈希、复核人和备注；入口才标记 `review_approved`。本轮**未创建任何批准记录**。

## 实测证据

最终自动审计为 `/data/hi3d/g0-prep/a0-sign-audit-balanced50-v3.json`。对 50 个输入：**12 个单连通体候选、8 个多连通体需复核、30 个非闭合排除**。20 个闭合输入均有正有向体积、绕序一致；各资产 512 个确定性随机探针都至少采到 1 个内部点。远离表面的 1/5 射线符号不一致数、域外误判数及距离幅值最大差均为 0。多连通体的 8 个需复核项不能凭这些探针排除相交或嵌套。v1/v2 是开发过程中的旧审计，后续使用 v3。

Open3D 0.18.0 的立方体控制测试：中心有符号距离 -0.25，已知外点 +0.15，1/5 射线结果相同。R32 候选 smoke 使用最终清单 index 19、单连通体 OBJ `008b9bd073d1...`，缓存位于 `/data/hi3d/g0-prep/targets-a0-candidate-smoke-v3/008b9bd073d1ccceccf3f53cac18fe0cf0b10eb4aab36b7f1b1888729e0fe9eb/R32/A0-candidate-ac71e1203611/`。结果：22930 cells、26508 唯一点、raw mesh 7836 面；缓存回读及与 vae3d `complete_fitting_support` 对照通过。对 512 个场采样点独立抽查 Open3D UDF 与 occupancy：距离幅值最大误差 0，明确远离表面的符号分歧 0。此为单资产 CPU 检查，不证明其他资产质量或 GPU 性能。

实际命令：

```bash
PYTHONPATH=/data/hi3d/g0-prep/vendor-full:/data/hi3d/g0-prep/vae3d:/data/hi3d/g0-prep \
  /data/hi3d/hi3d-validation/.venv/bin/python -m experiments.vae_sdf.a0_signed_distance audit \
  --manifest /data/hi3d/g0-prep/manifest-balanced50.json \
  --load-audit /data/hi3d/g0-prep/mesh-load-balanced50.json \
  --vae3d-root /data/hi3d/g0-prep/vae3d --resolution 128 \
  --report /data/hi3d/g0-prep/a0-sign-audit-balanced50-v3.json

PYTHONPATH=/data/hi3d/g0-prep/vendor-full:/data/hi3d/g0-prep/vae3d:/data/hi3d/g0-prep \
  /data/hi3d/hi3d-validation/.venv/bin/python -m experiments.vae_sdf.a0_signed_distance prepare \
  --manifest /data/hi3d/g0-prep/manifest-balanced50.json \
  --audit-report /data/hi3d/g0-prep/a0-sign-audit-balanced50-v3.json \
  --index 19 --vae3d-root /data/hi3d/g0-prep/vae3d \
  --output-root /data/hi3d/g0-prep/targets-a0-candidate-smoke-v3 \
  --resolution 32 --nsamples 5 --allow-candidate

PYTHONPATH=/data/hi3d/g0-prep/vendor-full:/data/hi3d/g0-prep/vae3d:/data/hi3d/g0-prep \
  /data/hi3d/hi3d-validation/.venv/bin/python -m experiments.vae_sdf.check_a0_cache \
  --cache /data/hi3d/g0-prep/targets-a0-candidate-smoke-v3/008b9bd073d1ccceccf3f53cac18fe0cf0b10eb4aab36b7f1b1888729e0fe9eb/R32/A0-candidate-ac71e1203611
```

后续人工复核文件格式（示意值，**不是批准记录**）：

```json
{
  "schema": "g0_a0_manual_review_v1",
  "manifest_sha256": "填写固定 manifest 的 SHA-256",
  "a0_audit_sha256": "填写 v3 审计报告的 SHA-256",
  "approved": [
    {"index": 19, "id": "填写资产 ID", "source_sha256": "填写主文件 SHA-256",
     "reviewer": "填写复核人", "note": "填写内外符号与自相交复核依据"}
  ]
}
```

## 剩余工作

1. 对 12 个单连通体候选和 8 个多连通体需复核项进行自相交、嵌套、法向及实体语义复核，再建立明确的 A0 允许子集；不能因自动审计通过就宣称可靠真值。
2. A0 当前只完成一个 R32 候选缓存；批量运行分辨率、共同支撑、统一 MC 提取器与 G0 几何指标还未固定。A1/A2 小样本见 [独立记录](g0-a1a2-initial-smoke-2026-10-29.md)。
3. A3/A4 的几何和渲染拟合入口及依赖仍待实现与实测。
