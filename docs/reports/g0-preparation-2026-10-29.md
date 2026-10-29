# G0 准备记录（2026-10-29）

验证类型：源码核查、远端环境清点、真实 mesh 加载和单资产 CPU 计时。**A1/A2 后续单资产目标 smoke 见 [独立记录](g0-a1a2-initial-smoke-2026-10-29.md)；尚未运行完整 G0、训练或质量评测。**

## 固定版本与目录

- Hi3D 本地/远端基线：`9c72ea6bba8f345e6f7770325298da57ec9bfe25`；修改版代码基线为其祖先 `5e5622d0baea953e48bde31607936f1ac1036be6`。
- vae3d：远端独立目录 `/data/hi3d/g0-prep/vae3d`，浅克隆实际 GitLab remote 并固定到 `422e79988ccda06ef0995bf7768cd05a8894a124`。未初始化其 gitlink 子目录；当前 A1/A2 代码未发现需要它们。
- 数据源：`/data/TRELLIS/datasets/ObjaverseXL_github/raw/github/objects`。以文件名的升序固定前 50 个支持格式文件，**不跳过失败文件，也不自动补样**。清单为 `/data/hi3d/g0-prep/manifest-first50.json`，每个主文件的 SHA-256 与字节数见 `/data/hi3d/g0-prep/preflight-first50.json`。清单来源共约 215 MB，含 OBJ 21、GLB 13、glTF 13、STL 3。
- 预检和计时脚本在本工程的 `experiments/vae_sdf/`；远端临时副本在 `/data/hi3d/g0-prep/experiments/vae_sdf/`。未修改远端 hi3d-vae-field 仓库。

## 加载审计

使用 vae3d `load_mesh` 和 `R=128` 对 50 个文件逐个只读加载，报告为 `/data/hi3d/g0-prep/mesh-load-first50.json`：42 个加载成功，8 个失败。失败中 7 个 glTF 缺少外部 `.bin`，1 个加载后没有有效三角形。42 个成功样本中，`trimesh.is_watertight` 为真者 6 个；这只是闭合候选检查，不是人工类别或 A0 可靠符号的最终认定。原始 50 个需保留失败在分母。

前 50 是按哈希文件名排序的用户指定 pilot，不能视为随机或分层代表性样本；类别目前均为 `unclassified`。若后续需要固定数量的有效样本，必须另建带版本的补样清单，不覆盖本次清单。

### 补足 50 个可加载样本

按用户后续要求，保留原始首 50 清单和 8 个失败记录，从其最后一个文件名之后继续按文件名字节序检查。9 个候选中 8 个通过相同的 vae3d `load_mesh`、R=128 加载，1 个失败。新清单为 `/data/hi3d/g0-prep/manifest-usable50.json`，由原始成功 42 个和新增成功 8 个组成；对应审计为 `mesh-load-usable50.json`，补样逐项记录为 `supplement-after-first50.json`。对新清单独立重跑完整加载审计，`mesh-recheck-usable50.json` 显示 **50/50 成功**。新旧清单都指向原始数据路径，没有复制 mesh。补样脚本为 `experiments/vae_sdf/supplement_usable.py`，运行前核对原 50 个主文件的 SHA-256 与原预检记录一致；新成功文件也记录了 SHA-256。此清单只保证在当前环境可加载，类别与可靠闭合性仍待审查。

补样与复检命令（使用同一 `PYTHONPATH` 和 Python 环境）：

```bash
PYTHONPATH=/data/hi3d/g0-prep/vendor-full:/data/hi3d/g0-prep/vae3d:/data/hi3d/g0-prep \
  /data/hi3d/hi3d-validation/.venv/bin/python -m experiments.vae_sdf.supplement_usable \
  --manifest /data/hi3d/g0-prep/manifest-first50.json \
  --audit /data/hi3d/g0-prep/mesh-load-first50.json \
  --preflight /data/hi3d/g0-prep/preflight-first50.json --resolution 128 \
  --output-manifest /data/hi3d/g0-prep/manifest-usable50.json \
  --output-audit /data/hi3d/g0-prep/mesh-load-usable50.json \
  --output-supplement /data/hi3d/g0-prep/supplement-after-first50.json

PYTHONPATH=/data/hi3d/g0-prep/vendor-full:/data/hi3d/g0-prep/vae3d:/data/hi3d/g0-prep \
  /data/hi3d/hi3d-validation/.venv/bin/python -m experiments.vae_sdf.inspect_meshes \
  --manifest /data/hi3d/g0-prep/manifest-usable50.json --resolution 128 \
  --report /data/hi3d/g0-prep/mesh-recheck-usable50.json
```

### 限额寻找额外闭合候选

按用户要求，从补样扫描最后一个文件之后，按文件名字节序最多检查后续 50 个支持格式的主文件，使用同一 vae3d `load_mesh`、R=128 和 `mesh.is_watertight` 判定；新闭合候选达到 19 个即停。实际检查满 50 个后因达到上限停止：**新闭合候选 5、可加载但非闭合 38、加载失败 7**（6 个 `FileNotFoundError`、1 个 `ValueError`）。原 50 可用清单中有 6 个闭合候选，因此当前累计 **11 个候选，距额外 19 个的目标仍差 14 个**。没有继续扫描，也没有替换原 50 可用清单。

完整逐项记录（含源文件 SHA-256、面数、法向绕序一致性和失败原因）为 `/data/hi3d/g0-prep/closed-scan-next50.json`；11 个候选的独立清单为 `manifest-closed-candidates.json`；38 个可加载非闭合样本的待定记录为 `nonclosed-pending-next50.json`。这些文件均指向原始 mesh，没有复制 mesh。闭合标志只表明加载后的三角网格拓扑满足 `is_watertight`，尚不能证明实体符号可靠或适合 A0。扫描脚本为 `experiments/vae_sdf/scan_closed_candidates.py`，实际命令：

```bash
PYTHONPATH=/data/hi3d/g0-prep/vendor-full:/data/hi3d/g0-prep/vae3d:/data/hi3d/g0-prep \
  /data/hi3d/hi3d-validation/.venv/bin/python -m experiments.vae_sdf.scan_closed_candidates \
  --manifest /data/hi3d/g0-prep/manifest-usable50.json \
  --audit /data/hi3d/g0-prep/mesh-recheck-usable50.json \
  --previous-scan /data/hi3d/g0-prep/supplement-after-first50.json \
  --limit 50 --needed 19 --resolution 128 \
  --output-report /data/hi3d/g0-prep/closed-scan-next50.json \
  --output-manifest /data/hi3d/g0-prep/manifest-closed-candidates.json \
  --output-pending /data/hi3d/g0-prep/nonclosed-pending-next50.json
```

### 固定 20 闭合候选 / 30 非闭合清单

后续用户将扫描范围再扩大最多 100 个文件，要求新增 9 个闭合候选。沿上一轮终点继续检查满 100 个：新增闭合候选 8、可加载非闭合 72、加载失败 20，因达到上限停止，累计闭合候选 19。用户随后允许再检查最多 50 个、找到 1 个即停；实际检查 21 个后找到第 1 个闭合候选并立即停止，这 21 个中另有非闭合 17、加载失败 3。对应逐项证据为 `/data/hi3d/g0-prep/closed-scan-next100.json` 与 `closed-scan-final-next50.json`；非闭合待定记录分别为 `nonclosed-pending-next100.json` 和 `nonclosed-pending-final-next50.json`。三轮额外闭合扫描一共检查 171 个主文件，新增 14 个闭合候选。

独立候选清单 `/data/hi3d/g0-prep/manifest-closed-candidates-final.json` 含原 6 个及新增 14 个 `is_watertight` 候选。最终 G0 输入清单 `/data/hi3d/g0-prep/manifest-balanced50.json` 取全部 20 个闭合候选，再从原 `manifest-usable50.json` 的 44 个非闭合样本中按原顺序取前 30 个；每个主文件记录字节数与 SHA-256。其完整独立复检 `/data/hi3d/g0-prep/mesh-load-balanced50.json` 为 **50/50 加载成功，20/20 闭合候选、30/30 非闭合**；清单与审计的 ID、路径和顺序逐项一致。原 `manifest-usable50.json`、三轮扫描记录及所有待定非闭合样本均保留。源 mesh 仍在 `/data/TRELLIS/datasets/ObjaverseXL_github/raw/github/objects`，没有复制。此分组是拓扑预筛，不是经过人工验证的可靠闭合或随机代表性样本。

第二轮、第三轮及最终清单的实际命令：

```bash
PYTHONPATH=/data/hi3d/g0-prep/vendor-full:/data/hi3d/g0-prep/vae3d:/data/hi3d/g0-prep \
  /data/hi3d/hi3d-validation/.venv/bin/python -m experiments.vae_sdf.scan_closed_candidates \
  --manifest /data/hi3d/g0-prep/manifest-usable50.json \
  --audit /data/hi3d/g0-prep/mesh-recheck-usable50.json \
  --previous-scan /data/hi3d/g0-prep/closed-scan-next50.json \
  --existing-candidates /data/hi3d/g0-prep/manifest-closed-candidates.json \
  --limit 100 --needed 9 --resolution 128 \
  --output-report /data/hi3d/g0-prep/closed-scan-next100.json \
  --output-manifest /data/hi3d/g0-prep/manifest-closed-candidates-after-next100.json \
  --output-pending /data/hi3d/g0-prep/nonclosed-pending-next100.json

PYTHONPATH=/data/hi3d/g0-prep/vendor-full:/data/hi3d/g0-prep/vae3d:/data/hi3d/g0-prep \
  /data/hi3d/hi3d-validation/.venv/bin/python -m experiments.vae_sdf.scan_closed_candidates \
  --manifest /data/hi3d/g0-prep/manifest-usable50.json \
  --audit /data/hi3d/g0-prep/mesh-recheck-usable50.json \
  --previous-scan /data/hi3d/g0-prep/closed-scan-next100.json \
  --existing-candidates /data/hi3d/g0-prep/manifest-closed-candidates-after-next100.json \
  --limit 50 --needed 1 --resolution 128 \
  --output-report /data/hi3d/g0-prep/closed-scan-final-next50.json \
  --output-manifest /data/hi3d/g0-prep/manifest-closed-candidates-final.json \
  --output-pending /data/hi3d/g0-prep/nonclosed-pending-final-next50.json

PYTHONPATH=/data/hi3d/g0-prep \
  /data/hi3d/hi3d-validation/.venv/bin/python -m experiments.vae_sdf.build_balanced_manifest \
  --usable-manifest /data/hi3d/g0-prep/manifest-usable50.json \
  --usable-audit /data/hi3d/g0-prep/mesh-recheck-usable50.json \
  --scan /data/hi3d/g0-prep/closed-scan-next50.json \
  --scan /data/hi3d/g0-prep/closed-scan-next100.json \
  --scan /data/hi3d/g0-prep/closed-scan-final-next50.json \
  --closed-manifest /data/hi3d/g0-prep/manifest-closed-candidates-final.json \
  --output /data/hi3d/g0-prep/manifest-balanced50.json

PYTHONPATH=/data/hi3d/g0-prep/vendor-full:/data/hi3d/g0-prep/vae3d:/data/hi3d/g0-prep \
  /data/hi3d/hi3d-validation/.venv/bin/python -m experiments.vae_sdf.inspect_meshes \
  --manifest /data/hi3d/g0-prep/manifest-balanced50.json --resolution 128 \
  --report /data/hi3d/g0-prep/mesh-load-balanced50.json
```

## 远端环境与初始化计时

- 远端 8 张 NVIDIA H20-3e；系统 `python3` 为 3.12.3。现有验证环境为 Python 3.10.22、torch 2.5.0+cu124。G0 工具使用该解释器，并用独立的 `/data/hi3d/g0-prep/vendor-full` 提供 Open3D 0.18.0 等 CPU 后端依赖，未改动已有虚拟环境。
- `kaolin`、`nvdiffrast` 未装；几何和渲染拟合环境未放行。当前没有验证 Vulkan/RTX 路径，已验证的射线与距离路径为 `cpu/cpu`。本地电脑为 Darwin arm64，未发现 NVIDIA CUDA 工具；完整 A3/A4 优先留在远端。
- 单资产 `0000eec...`、5178 faces 的 UE50/背壳初始化：R32 查询 2.82 s，R64 4.36 s，R128 17.49 s，R256 118.92 s。R256 其中射线投票 112.91 s、距离 2.00 s、场景建立 3.26 s；报告分别为 `/data/hi3d/g0-prep/bench-r{32,64,128,256}-cpu-asset0.json`。
- 较大资产 `0015e789...`、566606 faces 的 R64：加载 30.61 s，初始化查询 6.10 s；报告 `/data/hi3d/g0-prep/bench-r64-cpu-asset28.json`。这些是少量样本的单次计时，不能直接外推 50 个资产或 A3/A4 总成本。

## 实际执行入口

```bash
PYTHONPATH=/data/hi3d/g0-prep python3 -m experiments.vae_sdf.make_manifest \
  --root /data/TRELLIS/datasets/ObjaverseXL_github/raw/github \
  --limit 50 --output /data/hi3d/g0-prep/manifest-first50.json

PYTHONPATH=/data/hi3d/g0-prep/vendor-full:/data/hi3d/g0-prep/vae3d:/data/hi3d/g0-prep \
  /data/hi3d/hi3d-validation/.venv/bin/python -m experiments.vae_sdf.inspect_meshes \
  --manifest /data/hi3d/g0-prep/manifest-first50.json --resolution 128 \
  --report /data/hi3d/g0-prep/mesh-load-first50.json

PYTHONPATH=/data/hi3d/g0-prep/vendor-full:/data/hi3d/g0-prep/vae3d:/data/hi3d/g0-prep \
  /data/hi3d/hi3d-validation/.venv/bin/python -m experiments.vae_sdf.bench_back_shell \
  --manifest /data/hi3d/g0-prep/manifest-first50.json --index 0 --resolution 256 \
  --raycast-backend cpu --distance-backend cpu \
  --report /data/hi3d/g0-prep/bench-r256-cpu-asset0.json
```

## 剩余工作

1. A1/A2 分阶段中间场导出与缓存身份已在单资产验证；下一步是固定整批运行配置、处理失败分母与共同评测支撑。实现 A3/A4 阶段开关时不得用原 `convert(render_steps=0)` 代替。
2. 安装并验证 A3/A4 的 Kaolin、nvdiffrast、Warp 等依赖与 CUDA 前后向。先对 1–2 个资产完成全流程计时，再决定 50 个资产的分辨率与运行预算。
3. 最终 20 个闭合候选已有 [A0 自动审计](g0-a0-audit-smoke-2026-10-29.md)，但实体符号可靠性仍需人工复核；继续审查 30 个非闭合样本的类别与法向。A0 正式评测仅使用复核通过的闭合子集。保留各轮失败记录，并调查 glTF 缺失附件来源。
4. 在固定输入和共同支撑上导出 raw mesh，执行 G0 指标与失败统计；没有实测前不宣称重建质量提升。

## 本地浏览工具补充

本机入口为工程根目录的 `view_g0_meshes.command`，说明见 `docs/g0-mesh-browser.md`。它沿用 vae3d `view_train_t1.bat` 的“启动本机服务并打开 WebGL 页面”方式，但只浏览固定 G0 输入清单，不加载 checkpoint。远端只读辅助脚本位于 `/data/hi3d/g0-prep/experiments/vae_sdf/remote_mesh.py`。

原始首 50 清单的浏览验证：HTTP 清单返回 50 项、42 项可加载。`/mesh/0` 返回 2623 顶点、5178 三角面，二进制长度 93620 字节与格式一致；原失败项 `/mesh/8` 及越界索引均返回 404。本机 WebGL 页面已实际显示第一个 mesh。最终本地查看器默认读取 `balanced50`；`--dataset usable50` 可查看旧 50 可用清单，`--dataset first50` 可查看原始失败项。此验证属于浏览功能检查，不代表 G0 目标场或重建质量验证。
