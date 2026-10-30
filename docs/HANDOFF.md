# 会话交接：hi3d-vae-field

更新日期：2026-10-31。此文件记录进度与入口，技术方案以 [v0.3](experiment-plan-v0.3.md) 为准，修改版事实以 [基线分析](modified-repo-analysis.md) 为准。新会话的 G1/G2 操作顺序见 [执行交接](g1-g2-execution-handoff-2026-10-31.md)。

阶段性总览见 [G0/G0b 进展总结](reports/g0-progress-summary-2026-10-30.md)；逐项数值和命令仍以各阶段报告为准。

**最新设计决策（2026-10-31）：**用户要求 MC/FlexiCubes × flat/smooth 四条 A3/A4 路线分别重新拟合；最终训练固定选定路线，优先研究 smooth+FlexiCubes，目标分辨率先 R1024、再视可微分块实测推进 R2048。2×5 预览、缓存/API、梯度与质量验收写在 [四路线实施规格](g0-four-route-implementation-spec.md)。扇区分组已迁移 GPU，固定 50 件四路线 A3/A4 **200/200 成功**，A1/A2 共用场双提取器缓存生成成功；全部新 GPU 作业仅用物理 GPU 7。位置项有限差分通过，质量仍需复核，见 [执行记录](reports/g0-four-route-progress-2026-10-30.md)。现有 G0b 仍只是同一 A4 场换提取器的历史对照。

**2026-10-31 清理与路线锁定：**用户已选 `flexicubes_smooth` 作为后续目标生成与提取路线，并表示人工 mesh 复核已完成；逐资产复核记录尚未收进仓库，不能据此改写自动质量审计。远端 G0b 的 100 个旧对照 PLY、旧 R128/A4 G1 的 14 个 checkpoint 已删除，分别释放 164527506 与 16296072682 字节。其配置、trace、评测报告和日志保留；历史报告中的 checkpoint 路径现在只作运行记录，不能恢复旧训练。四路线 v5 正式目标完整保留。G1/G2 新实验必须另建 run_id，不复用旧权重。

## 1. 当前状态

| 项目 | 状态与证据 |
|---|---|
| 工程来源 | Hi3Dv1.0-modified，已迁移为 hi3d-vae-field，保留 Git 历史 |
| 官方比较基线 | `13657ad09afaa92e350b461a1cd66399e44a59c4` |
| 修改版代码基线 | `5e5622d0baea953e48bde31607936f1ac1036be6` |
| 本次交接补充前的文档提交 | `9e76d8e`；当前 HEAD 以 `git log -1` 为准，不将固定文档哈希当作运行版本 |
| vae3d 参考版本 | `422e79988ccda06ef0995bf7768cd05a8894a124` |
| 方案与分析 | 已完成静态源码核查、v0.3 方案、修改版分析；v0.2 仅作历史归档 |
| 实验代码 | G0/G0b 入口与结果见下；独立 G1 编码器、GT 支撑可微解码、连续场头、严格恢复及 GT/预测支撑评测已实现。完成 2/8/50 资产 smoke、双样本过拟合及固定 8 训练/2 验证诊断；G1 质量未放行，见 [初始报告](reports/g1-initial-a4-2026-10-30.md)、[双样本报告](reports/g1-overfit-reconstruction-2026-10-30.md)、[8+2 报告](reports/g1-eight-train-two-val-2026-10-30.md) |
| 目标缓存/实验结果 | 远端 A1/A2 R128 CPU、A3 R128 GPU 30 步与 A4 R128/4096 200 步均 50/50 运行成功，见 [A1/A2](reports/g0-a1a2-batch-r128-2026-10-29.md)、[A3](reports/g0-a3-batch-r128-2026-10-30.md)、[A4](reports/g0-a4-batch-r128-render4096-2026-10-30.md) 报告。A4 相对 A3 同提取器 Chamfer 改善 39/50，但 16 个资产存在倒置穿越 cell，需带质量疑点标志。A0 自动审计为 12 单体候选、8 多体需复核、30 非闭合仅从 A0 符号基准排除；A3/A4 均保留原标志。仅有 1 个 R32 未批准 A0 候选缓存，见 [A0 审计记录](reports/g0-a0-audit-smoke-2026-10-29.md)。尚无正式 A0 真值或完整 G0 对照；G1 诊断 checkpoint 见 [报告](reports/g1-initial-a4-2026-10-30.md) |
| G0 准备 | 固定独立的 20 闭合候选 / 30 非闭合清单 `manifest-balanced50.json`，完整复检 50/50 可加载；原始与补样清单及失败记录均保留。选样证据见 [G0 准备记录](reports/g0-preparation-2026-10-29.md)。本地[浏览器](g0-mesh-browser.md)可查看全部输入 mesh、A1/A2 零面和场切片，以及 G0b 原 mesh / classic MC / FlexiCubes 三视窗与半透明叠加。A1/A2/A3/A4 批量已完成；G0b 接口对照 50/50 成功，见 [G0b 报告](reports/g0b-a4-r128-interface-2026-10-30.md)。A0 正式真值及完整 G0 尚未完成 |
| G0 预览服务 | 当前 `/compare` 是四路线 2×5 页面，端口 `127.0.0.1:8765`；启动方式及独立快速预览缓存见 [浏览说明](g0-mesh-browser.md)。旧 G0b 网格已清理，历史页面不能再依赖其原精度文件；服务是否运行需现场检查 |
| GPU 验证 | Kaolin 距离算子、nvdiffrast 栅格化前向/梯度、A3/A4 50 资产 GPU 拟合及 G1 单卡网络训练 smoke 已实测；多 GPU 等价性和 2048 性能未验证 |
| G1 训练 | 固定 50 个 A4 目标已做各两次更新的 100 步链路诊断；48、5 号严格恢复训练至 500 步并评测第 50/250/500 步。另固定训练 48/5/35/2/20/37/44/7、验证 0/30，400 步完成；第 200/400 步两个未训练资产 GT/预测支撑均产出 raw mesh，但相对 A4 目标仍明显偏离，30 号预测支撑网格存在边界缺口。证据见 [8+2 报告](reports/g1-eight-train-two-val-2026-10-30.md)；G1 质量未放行 |
| G1 本地监控 | 独立只读服务 `http://127.0.0.1:8767/`，每 5 秒经 SSH 读取指定远端 G1 运行的配置和逐步日志；与 8765 独立。下次训练先用新 run_id 启动并核对监测页，再启动训练；历史 checkpoint 已清理，旧页面记录的权重可用性不可当作当前事实。入口与限制见 [G1 浏览说明](g1-train-browser.md) |
| Git 操作约定 | 本地作为源码/文档提交源，push 由用户操作；本轮已获本地提交授权。2026-10-31 对两端待提交的四路线代码与文档逐文件 SHA256 校验一致，提交前 HEAD 均为 `1c08ea9`。后续不再从本地复制源码到远端；用户 push 后在远端先审查工作区，再通过 Git 同步。新提交与同步状态以实际命令复核为准 |

A4 环境边界补充：独立目标、源码、工具链和构建产物均在 `/data/hi3d/g0-prep/`，但最初的 `uv` 依赖安装使用默认 `/home/le.chang/.cache/uv` 缓存；后续构建已改用 G0 内的 `UV_CACHE_DIR`。未清理共享缓存，详情见 [A4 报告](reports/g0-a4-batch-r128-render4096-2026-10-30.md)。

新会话先运行以下只读检查（从工程根目录执行）：

```bash
git status --short --branch
git log -3 --oneline
git remote -v
git branch -vv
```

核查时 remote 名为 origin 和 github，本地 main 跟踪 origin/main；这仅是当前机器的配置，clone 后可能不同。不要把此前对话中示例 remote 名 gitlab 写死到脚本，也不要把含凭据的地址复制进文档。

## 2. 已确认要求与尚未确认的设计

**用户已明确：**基于修改版 Hi3D 做实验；目标生成参考 vae3d 的 prepare_and_train_t1.bat；工程名称为 hi3d-vae-field；G0 先用 20 个闭合候选与 30 个非闭合 mesh；G1 先用同一批 50 个目标，编码器按 vae3d 结构适配但从头初始化，Hi3D 解码骨干热启动、新头从头初始化；后续希望 R1024 或 R2048 训练，并核查与 Hi3D 整数角点格的相位一致性；网页 mesh 预览改逐面 flat shading；后续提交须先向用户确认，push 由用户操作。

**本轮新增已确认设计：**A3/A4 的 MC/FlexiCubes 两行必须各自参与拟合；flat/smooth 是拟合损失的两档，必须重跑 A3/A4，不能仅切换网页着色。四路最终选择一条用于训练；优先验证 smooth+FlexiCubes，但质量结论待实测。预览需扩展为原 mesh、A1、A2、A3、A4 的两行五列，A1–A4 可按列隐藏；详见 [实施规格](g0-four-route-implementation-spec.md)。

**技术事实：**目标生成包含 UE50 投票、门控背壳、几何拟合和渲染拟合；最终是配对的场与形变；两仓库编码布局/倍率不同；当前流式解码只能作为推理路径。

**方案建议，尚非用户逐项确认：**使用4通道场输入 encoder、调整为倍率8/latent32、采用连续 signed-tanh 输出头、先独立 VAE 后接 DiT、使用骨干热启动、首轮 B2/B4 对照。实施相关改动前结合当前用户指令确定范围，不能以“文档写了”当作已批准的最终架构。

**尚缺运行信息：**正式训练/验证/测试划分、GPU 小时预算、是否必须保留旧 DiT，以及 A0 的人工符号批准。Hi3D 发布 VAE 权重已定位到远端 `/data/hi3d/hi3d-validation/models/Hi3Dv1.0/vae/model.safetensors` 并用于 G1 热启动；该路径是本机位置，不写死到模型实现。A3/A4 目标生成与 G1 单卡训练 smoke 已验证；R1024/2048 的目标缓存、训练反传和显存尚未验证，具体顺序见 [方案](experiment-plan-v0.3.md)。

## 3. 下一项可交付任务

**当前优先执行：**四路线 R128 固定 50 件及 `/compare` 已完成。用户选定 `flexicubes_smooth`，下一项按 [G1/G2 执行交接](g1-g2-execution-handoff-2026-10-31.md) 完成路线目标读取、phase=0/矩阵审计、R256/512 资源测量与 G1 新 run；R1024 先单资产 smoke。以下 1–7 项为历史遗留核查，不是四路线实施顺序。

用户决定先执行 G0b，G0 的质量是否通过暂缓判断；当前 G0b 已有数值与提取器对照，暂不改网络骨干：

1. 复核 20 个闭合候选的自相交、组件相交/嵌套、法向及实体语义，形成明确的 A0 人工允许子集；8 个多连通体可在复核后纳入，不预先排除。其疑点标志在 A3/A4 中保留，不影响两阶段纳入。
2. 保留已完成的 R128 CPU A1/A2、R128 GPU A3 与 R128/4096 GPU A4 批量基线；A2_classic→A3 与 A3_classic→A4 分别用同一提取器配对评测，不能与原 A1/A2 Lewiner 指标直接混用。
3. 只对人工批准的闭合子集运行正式 A0；30 个非闭合样本继续参与 A1–A4 表示验证。
4. 对 A4 的 16 个倒置穿越单元资产、11 个 Chamfer 回退资产和 A3 回退/离群资产做可视复核；决定 A4 是否需要硬约束修订，另建协议与缓存。
5. G0b 已验证 A4 场的中心点/角点无损适配和相同 MC 的精确一致性。FlexiCubes 相对 MC 的 Chamfer 在 41/50 样本增大，先复核 [G0b 报告](reports/g0b-a4-r128-interface-2026-10-30.md) 所列重点编号，再决定修接口或保持 classic MC。完整 G0 质量结论仍待定。原模型流式一致性 G0c 可独立验证，新输出头实现后需重测。
6. G1 已完成 2/8/50 样本梯度 smoke、48/5 号 50→500 步严格恢复，以及固定 8 训练/2 验证的 400 步诊断。下一步按 [8+2 报告](reports/g1-eight-train-two-val-2026-10-30.md) 检查场符号、零面及 30 号预测支撑边界缺口；再决定是否扩大训练或进入 G2。
7. 用户指定后续高分辨率目标与 Hi3D 角点相位一致：先做独立 phase=0/R128 对照和矩阵审计，再测 R256/512 目标与反传资源；以 R1024 为首个高分辨率训练目标，R2048 待可微分块路径验证。现有 phase=0.5/R128 实验结果不能直接改名或原地覆盖。网页预览已改逐面 flat shading，A4 渲染目标本身没有更改。

现有 A0/A1/A2/A3/A4 入口、小样本验证及批量评测见本文件首表与阶段报告。现有 vae3d `convert` 不支持直接用 render_steps=0 跳过渲染；A3/A4 已使用独立阶段入口实现。

已完成的入口验收、CPU 数值对照、R128 A1/A2/A3、R128/4096 A4 与 G0b 固定协议见阶段报告；A0 正式真值与跨阶段统一对照仍待完成。A4 运行 50/50 成功，但 16 个资产有倒置穿越 cell；G0b 接口数值验收通过，FlexiCubes 路径需进一步复核；当前没有完整 G0 质量结论。

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
| OS、driver、CUDA、PyTorch 与依赖版本 | G0 使用 Python 3.10.22、torch 2.5.0+cu124、独立 Open3D 0.18.0；A3 在独立目录使用 Kaolin 0.18.0、Warp 1.18.0；A4 在独立目录使用 nvdiffrast 0.4.0、临时 CUDA 12.4 nvcc。完整训练依赖组合未验证 |
| mesh manifest、训练/验证/测试划分 | G0 固定 `/data/hi3d/g0-prep/manifest-balanced50.json`，50/50 可加载；正式训练/验证/测试划分未建立 |
| Hi3D 权重路径及哈希 | 未记录 |
| vae3d 源码/预处理缓存路径 | 固定源码 `/data/hi3d/g0-prep/vae3d`；A1/A2 批量 `/data/hi3d/g0-prep/g0-a1a2-r128-cpu-v1/`，A3 批量 `/data/hi3d/g0-prep/g0-a3-r128-gpu-v1/`，A4 批量 `/data/hi3d/g0-prep/g0-a4-r128-render4096-v1/`，G0b 批量 `/data/hi3d/g0-prep/g0b-a4-r128-v1/`；A0 候选及 smoke 路径见阶段报告 |
| 输出根目录、run_id、配额/最长时长 | A1/A2、A3、A4 的 run_id 见上一行；A4 逐资产时间合计约 470 秒，非训练配额估计。A0 正式预算未确定 |
| 射线/距离/渲染后端 | A1/A2 使用 CPU 射线与距离，A0 候选仅 Open3D CPU；A3 使用 Kaolin GPU 距离与梯度；A4 使用 nvdiffrast CUDA 可微渲染；Linux 未照搬 DXR |
| 运行命令与断点恢复命令 | A1/A2、A3、A4 分别见各阶段报告；均逐资产记录和恢复。A0 正式批量与断点恢复待实现 |

首轮离线生成目标后再训练。记录标签拟合和网络训练两类成本；4096 渲染标签拟合不是一次普通 SDF 查询。原预处理内部有 cuda:0 假设；多卡映射和进程配置须实测。

## 6. 验证证据与后续更新规则

| 类型 | 当前已有结论 | 不代表什么 |
|---|---|---|
| 静态源码分析 | 调用链、数据格式、流式 inference_mode 等已核对，见方案与分析 | 不证明生成质量提升 |
| 上次迁移检查 | 文件内容/Git历史保留、Python AST 检查曾通过 | 不证明依赖可安装或 GPU 可运行 |
| 本轮文档检查 | 检查相对链接与 `git diff --check` | 不等于实验测试 |
| G0 准备与 A1/A2/A3/A4 批量 | 固定 50 个可加载 mesh；A0 自动审计及未批准候选缓存；A1/A2 CPU、A3/A4 GPU 均 50/50；A4 回读审计、held-out 渲染与同提取器指标见 [报告](reports/g0-a4-batch-r128-render4096-2026-10-30.md) | 不代表完整 G0 质量、A0 人工批准、网络训练收益或完整 GPU 峰值显存；A4 的 16 个倒置穿越资产需复核 |
| G0b 接口与提取器对照 | 固定 50 个 A4 场全数回读，最大共享点误差 `4.77e-7`，相同 MC 网格精确一致；FlexiCubes 相对 MC 对原 mesh 的 Chamfer 41/50 增大，见 [G0b 报告](reports/g0b-a4-r128-interface-2026-10-30.md) | 不代表 FlexiCubes 几何质量放行、A0 人工批准或网络训练收益 |
| G1 网络训练与验证 | 50 个 A4 目标各参与两次更新且梯度有效；48、5 号严格恢复至 500 步；8 训练/2 未训练验证 400 步，GT/预测支撑 raw mesh 均非空且第 200→400 步网格误差改善，见 [8+2 报告](reports/g1-eight-train-two-val-2026-10-30.md) | 不代表 G1 质量放行、代表性泛化或 G2 收益；两个验证资产相对 A4 目标仍有较大重建误差 |
| 完整 G0/G0c/G1、G2–G4 | 均未完成 | README 性能声明不能作为本项目实测 |

每完成一个阶段，记录：日期、实验代码 commit、两套源码版本、resolved_config、数据/权重哈希、环境、执行命令、产物位置、指标、失败清单和下一步。可在 docs/reports/ 放轻量报告并引用外部结果目录；模型权重和大型缓存不直接加入普通 Git。

维护分工：本文件管状态和任务；experiment-plan-v0.3.md 管实验设计；modified-repo-analysis.md 管固定基线的事实。若设计发生实质变化，增加决策记录/修订说明，保留 v0.2 历史归档。

## 7. 2026-10-29 初始工程回顾快照

以下为 G0 准备前的历史核查记录；当前工作区与远端 G0-prep 状态以前文及阶段报告为准。

- 本地和远端 `/data/hi3d/hi3d-vae-field` 均为 `main`、HEAD `9c72ea6bba8f345e6f7770325298da57ec9bfe25`，工作区干净，跟踪 `origin/main`。本地另有名为 `github` 的 remote；远端仅见 `origin`。未执行 fetch、push 或代码修改。
- 本地参考仓库 `/Users/changle/code/3dgen/vae3d` 为 `422e79988ccda06ef0995bf7768cd05a8894a124`，仅 `.vscode/` 未跟踪。当前 Hi3D 工程中未发现 `experiments/vae_sdf` 或训练入口；`SparconvVAE` 仍只有 decoder。
- 只读核对命令：本地 `git status --short --branch`、`git rev-parse HEAD`、`git remote -v`、`git log -5 --oneline`、`rg --files` 与相关源码 `sed`；远端通过 SSH 执行 `git status --short --branch`、`git rev-parse HEAD`、`git remote -v`、`nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader`、`df -h /data/hi3d`、目录存在性 `ls -ld`。上述均属源码核查/环境清点，没有 GPU 训练或质量实测。
- 当时的下一步是独立目标生成入口与 G0；现已完成 G0 选样、A0 审计和 A1/A2/A3/A4 50 资产批量基线，剩余任务见第 3 节。
