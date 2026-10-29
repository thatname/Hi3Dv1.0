# 会话交接：hi3d-vae-field

更新日期：2026-10-29。此文件记录进度与入口，技术方案以 [v0.3](experiment-plan-v0.3.md) 为准，修改版事实以 [基线分析](modified-repo-analysis.md) 为准。

## 1. 当前状态

| 项目 | 状态与证据 |
|---|---|
| 工程来源 | Hi3Dv1.0-modified，已迁移为 hi3d-vae-field，保留 Git 历史 |
| 官方比较基线 | `13657ad09afaa92e350b461a1cd66399e44a59c4` |
| 修改版代码基线 | `5e5622d0baea953e48bde31607936f1ac1036be6` |
| 本次交接补充前的文档提交 | `9e76d8e`；当前 HEAD 以 `git log -1` 为准，不将固定文档哈希当作运行版本 |
| vae3d 参考版本 | `422e79988ccda06ef0995bf7768cd05a8894a124` |
| 方案与分析 | 已完成静态源码核查、v0.3 方案、修改版分析；v0.2 仅作历史归档 |
| 实验代码 | 已有 G0 清单/浏览、A0 闭合与符号审计、候选欧氏有符号距离入口，以及 A1/A2 同源查询与独立阶段缓存。A3/A4、批量评测和训练适配仍未实现 |
| 目标缓存/实验结果 | 远端 G0-prep 已有 A0 未验证候选 R32 与 A1/A2 R32/R128 单资产 smoke 缓存；A0 自动审计为 12 单体候选、8 多体需复核、30 非闭合仅从 A0 符号基准排除。尚无正式 A0 批准、完整 G0 指标或训练 checkpoint，见 [A0 审计记录](reports/g0-a0-audit-smoke-2026-10-29.md)与 [A1/A2 记录](reports/g0-a1a2-initial-smoke-2026-10-29.md) |
| G0 准备 | 固定独立的 20 闭合候选 / 30 非闭合清单 `manifest-balanced50.json`，完整复检 50/50 可加载；原始与补样清单及失败记录均保留。选样证据见 [G0 准备记录](reports/g0-preparation-2026-10-29.md)，本地浏览入口见 [说明](g0-mesh-browser.md)。A0 自动审计与 A0/A1/A2 单资产 smoke 已做，完整 G0 尚未完成 |
| GPU 验证 | 尚未执行本项目目标迁移、梯度、多 GPU 等价性或 2048 性能验证 |
| Git 操作约定 | 未经用户确认不创建新提交；push 由用户操作。本轮 G0 代码及文档仍在本地工作区，远端 Git 仓库未修改 |

新会话先运行以下只读检查（从工程根目录执行）：

```bash
git status --short --branch
git log -3 --oneline
git remote -v
git branch -vv
```

核查时 remote 名为 origin 和 github，本地 main 跟踪 origin/main；这仅是当前机器的配置，clone 后可能不同。不要把此前对话中示例 remote 名 gitlab 写死到脚本，也不要把含凭据的地址复制进文档。

## 2. 已确认要求与尚未确认的设计

**用户已明确：**基于修改版 Hi3D 做实验；目标生成参考 vae3d 的 prepare_and_train_t1.bat；工程名称为 hi3d-vae-field；G0 先用 20 个闭合候选与 30 个非闭合 mesh；后续提交须先向用户确认，push 由用户操作。

**技术事实：**目标生成包含 UE50 投票、门控背壳、几何拟合和渲染拟合；最终是配对的场与形变；两仓库编码布局/倍率不同；当前流式解码只能作为推理路径。

**方案建议，尚非用户逐项确认：**使用4通道场输入 encoder、调整为倍率8/latent32、采用连续 signed-tanh 输出头、先独立 VAE 后接 DiT、使用骨干热启动、首轮 B2/B4 对照。实施相关改动前结合当前用户指令确定范围，不能以“文档写了”当作已批准的最终架构。

**尚缺运行信息：**训练用数据划分、权重位置、GPU 小时预算、是否必须保留旧 DiT，以及 A3/A4 的依赖/后端与资源实测。G0 的远端主机、工程目录、数据源、GPU 与磁盘已核查；不要从瞬时设备清点推断训练配额。

## 3. 下一项可交付任务

当前应继续完成 G0 表示验证，暂不改网络骨干：

1. 复核 20 个闭合候选的自相交、组件相交/嵌套、法向及实体语义，形成明确的 A0 人工允许子集；8 个多连通体可在复核后纳入，不预先排除。
2. 固定 G0 的分辨率、CPU/GPU 后端、统一 raw mesh 提取器、共同支撑和评测预算；保留目前 A0/A1/A2 单资产缓存作为 smoke 证据。
3. 在 50 个固定资产上批量运行 A1/A2，包括 30 个非闭合样本；正式 A0 仅对人工批准的子集运行。记录失败分母、时间、缓存体积及原坐标系 raw mesh 几何指标。
4. 实现并验证 A3 几何拟合、A4 渲染拟合的独立入口与依赖，再用相同资产/协议完成 A1–A4 对照。
5. 输出完整 G0 报告；之后进行 G0b 中心点/角点适配，再进入梯度和训练任务。原模型流式一致性 G0c 可独立验证，新输出头实现后需重测。

现有 A0/A1/A2 入口和小样本验证见本文件首表及三份阶段报告；A3/A4 和批量评测尚无可运行入口。现有 vae3d `convert` 不支持直接用 render_steps=0 跳过渲染，不能写出未实现的命令。

已完成的入口验收与 CPU 数值对照见阶段报告；正式 G0 的几何指标阈值、分辨率和预算须在批量运行前锁定。当前没有完整 G0 质量结论。

## 4. vae3d 获取与依赖

vae3d 不是当前工程已纳入的代码或 submodule。只 clone 本工程无法执行其预处理。当前机器位置为 `/Users/changle/code/3dgen/vae3d`，入口为仓库相对路径 `prepare_and_train_t1.bat`；该绝对路径仅用于定位，远程不要写死。

远端已在 `/data/hi3d/g0-prep/vae3d` 固定上述提交，来源和清单记录见 [G0 准备记录](reports/g0-preparation-2026-10-29.md)。新环境获取时须核对来源、提交及依赖；不要复制 .vscode 或整个开发目录。

以下为可执行的源码导出示例，输出路径可替换（不含未提交文件）：

```bash
git -C /Users/changle/code/3dgen/vae3d archive --format=tar   --output=/tmp/vae3d-422e799-source.tar   422e79988ccda06ef0995bf7768cd05a8894a124
```

导出前检查该版本是否使用 submodule/gitlink：git archive 不包含其实际内容。迁移流水线对第三方目录的需要须按 imports 审查；如果需要则单独固定并导出，不声称单个 tar 已包含所有依赖。源码快照传输后核对校验和并记录来源。

优先审查模块：

| 类别 | vae3d 仓库相对路径 |
|---|---|
| 入口与编排 | prepare_and_train_t1.bat、vae3d/prepare_and_train.py、vae3d/prepare_pipeline.py |
| 场查询 | vae3d/ue_sign.py、back_shell.py、gpu_distance.py、rtx_raycast.py 及相关 shader |
| 几何/渲染拟合 | vae3d/refine.py、render_refine.py、geometry.py、normal_targets.py |
| 表示/支撑/缓存 | vae3d/center_points.py、occupancy.py、data.py、target_cache.py |
| 后续网络迁移 | vae3d/model.py、sparse.py、prediction.py、losses.py、train.py |

这是审查起点，不是完整依赖闭包。实现时保留必要许可证，并将新入口的实际依赖锁定。当前 docker/constraints.txt 面向继承的 Hi3D 推理，不能当作 vae3d+训练已验证环境。

## 5. 远程运行记录

已核实项与待补项如下；具体 G0 命令、结果及缓存路径见三份阶段报告。凭据、令牌和 SSH 私钥不写入仓库。

| 字段 | 当前状态 |
|---|---|
| 主机/调度环境、工作目录 | 2026-10-29 经 SSH 核对：`le.chang@124.156.97.173`，`/data/hi3d/hi3d-vae-field`；调度策略未确认 |
| GPU/显存/数量，CPU/RAM/空闲磁盘 | `nvidia-smi` 显示 8× NVIDIA H20-3e、每卡 143771 MiB；`df -h /data/hi3d` 曾显示剩余 415G；CPU/RAM 未系统测量。此为瞬时状态，不代表训练配额 |
| OS、driver、CUDA、PyTorch 与依赖版本 | G0 使用 Python 3.10.22、torch 2.5.0+cu124、独立 Open3D 0.18.0；完整训练/A3/A4 依赖组合未验证 |
| mesh manifest、训练/验证/测试划分 | G0 固定 `/data/hi3d/g0-prep/manifest-balanced50.json`，50/50 可加载；正式训练/验证/测试划分未建立 |
| Hi3D 权重路径及哈希 | 未记录 |
| vae3d 源码/预处理缓存路径 | 固定源码 `/data/hi3d/g0-prep/vae3d`；A0 候选及 A1/A2 smoke 缓存在 `/data/hi3d/g0-prep/targets-*`，精确路径见阶段报告 |
| 输出根目录、run_id、配额/最长时长 | G0 smoke 在 `/data/hi3d/g0-prep/`；正式批量 run_id、配额/最长时长未确定 |
| 射线/距离/渲染后端 | A1/A2 的 CPU 射线与 CPU 距离、A0 的 Open3D CPU 路径已小样本验证；GPU/渲染后端未验证，Linux 不直接照搬 DXR |
| 运行命令与断点恢复命令 | A0/A1/A2 实际小样本命令见阶段报告；批量与断点恢复命令待实现/固定 |

首轮离线生成目标后再训练。记录标签拟合和网络训练两类成本；4096 渲染标签拟合不是一次普通 SDF 查询。原预处理内部有 cuda:0 假设；多卡映射和进程配置须实测。

## 6. 验证证据与后续更新规则

| 类型 | 当前已有结论 | 不代表什么 |
|---|---|---|
| 静态源码分析 | 调用链、数据格式、流式 inference_mode 等已核对，见方案与分析 | 不证明生成质量提升 |
| 上次迁移检查 | 文件内容/Git历史保留、Python AST 检查曾通过 | 不证明依赖可安装或 GPU 可运行 |
| 本轮文档检查 | 检查相对链接与 `git diff --check` | 不等于实验测试 |
| G0 准备与小样本 | 固定 50 个可加载 mesh；A0 自动审计及未批准候选缓存；A1/A2 CPU 单资产数值对照，见三份阶段报告 | 不代表完整 G0 质量或 A0 人工批准 |
| 完整 G0/G0b/G0c/G1–G4 | 均未完成 | README 性能声明不能作为本项目实测 |

每完成一个阶段，记录：日期、实验代码 commit、两套源码版本、resolved_config、数据/权重哈希、环境、执行命令、产物位置、指标、失败清单和下一步。可在 docs/reports/ 放轻量报告并引用外部结果目录；模型权重和大型缓存不直接加入普通 Git。

维护分工：本文件管状态和任务；experiment-plan-v0.3.md 管实验设计；modified-repo-analysis.md 管固定基线的事实。若设计发生实质变化，增加决策记录/修订说明，保留 v0.2 历史归档。

## 7. 2026-10-29 初始工程回顾快照

以下为 G0 准备前的历史核查记录；当前工作区与远端 G0-prep 状态以前文及阶段报告为准。

- 本地和远端 `/data/hi3d/hi3d-vae-field` 均为 `main`、HEAD `9c72ea6bba8f345e6f7770325298da57ec9bfe25`，工作区干净，跟踪 `origin/main`。本地另有名为 `github` 的 remote；远端仅见 `origin`。未执行 fetch、push 或代码修改。
- 本地参考仓库 `/Users/changle/code/3dgen/vae3d` 为 `422e79988ccda06ef0995bf7768cd05a8894a124`，仅 `.vscode/` 未跟踪。当前 Hi3D 工程中未发现 `experiments/vae_sdf` 或训练入口；`SparconvVAE` 仍只有 decoder。
- 只读核对命令：本地 `git status --short --branch`、`git rev-parse HEAD`、`git remote -v`、`git log -5 --oneline`、`rg --files` 与相关源码 `sed`；远端通过 SSH 执行 `git status --short --branch`、`git rev-parse HEAD`、`git remote -v`、`nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader`、`df -h /data/hi3d`、目录存在性 `ls -ld`。上述均属源码核查/环境清点，没有 GPU 训练或质量实测。
- 当时的下一步是独立目标生成入口与 G0；现已完成 G0 选样、A0 审计和 A1/A2 单资产 smoke，剩余任务见第 3 节。
