# G0 四路线拟合与预览实施规格

状态：2026-10-30 用户确认的设计；**尚未实现、尚未 GPU 实测**。本文件供下一位执行者直接实施。现有 R128 A1–A4、G0b 和 G1 结果均是历史基线，不得原地覆盖或改称四路线结果。进度见 [HANDOFF](HANDOFF.md)，A0–A4 定义见 [实验方案](experiment-plan-v0.3.md)。

## 1. 目标与边界

在固定 `manifest-balanced50.json` 的 50 个输入 mesh 上，从同一 A1/A2 场出发，分别生成四条完整的 A3/A4 目标。两个因素是 **优化过程中实际使用的提取器**（classic Marching Cubes，简称 MC；或 Hi3D 派生 FlexiCubes）和 **优化损失中的法线模型**（flat 或 smooth）。A3 几何拟合与 A4 渲染拟合都必须使用该路线的提取器及法线模型；A4 从同一路线的 A3 缓存续跑。最终训练选定一条固定路线，编码器/解码器目标及最终 mesh 提取器保持一致。用户优先关注 smooth + FlexiCubes，但尚未宣称它质量更好。

| route_id（固定写入缓存） | A3/A4 提取器 | A3/A4 法线 | 性质 |
|---|---|---|---|
| `mc_flat` | classic MC | flat | 当前 R128 历史协议的重现对照；新缓存仍独立生成 |
| `mc_smooth` | classic MC | smooth | 法线因素对照 |
| `flexi_flat` | FlexiCubes | flat | 提取器因素对照 |
| `flexi_smooth` | FlexiCubes | smooth | 优先研究候选 |

A1 = UE50 投票场，A2 = A1 + 25% 门控背壳，二者尚无 A3/A4 拟合，也没有法线模式。它们应从**同一份字段**各经 MC 和 FlexiCubes 提取，显示为两行的共用起点；不要复制成四套不同的 A1/A2 训练结果。A0 的人工符号批准继续独立进行，30 个非闭合 mesh 仍参加 A1–A4。四路线实验不是“对同一 A4 场换提取器”的旧 G0b；后者只用于隔离接口差异。

“flat / smooth”在本任务中是**拟合损失配置**，不是网页着色开关；每一档都要有独立 A3/A4 场、优化日志和原精度 mesh。网页可按所选路线以相同法线模式着色辅助查看，但着色不产生或修改实验结果。不要把 smooth 称作 SDF 线性插值：场在等值面边上仍按提取算法插值，这里的 smooth 指网格法线在三角形内的重心线性插值并归一化。

## 2. 固定协议与缓存身份

四路固定同一：输入资产顺序与 SHA256、预处理/原坐标恢复、A1/A2 字段、稀疏支撑和完整 cell 集、采样相位、R、A3 30 步与 8192 采样、A4 200 步与 4096 px、32 训练/12 留出视角、每步视角抽样与随机种子、损失项和权重、delta 单轴 ±0.5 体素、A4 跨零策略、目标厚度、设备精度、评测采样点与种子。比较四路线时只允许变动 `extractor` 与 `normal_mode`；为使两种提取器有效而新增的**共同**数值保护必须对四路一致并单独记录。若无法保持共同协议，标记为不同实验，不做纯二因素归因。

旧 R128 目标采用 `sampling_offset=0.5`。后续要与 Hi3D 整数格角点契约对齐，必须单独创建 `sampling_offset=0` 缓存并审计 `Point.matrix`；不得把相位差塞进有界 delta 或重命名旧缓存。每个缓存键至少包含 `manifest_hash/source_hash, R, sampling_offset, normalization_matrix, support_hash, A2_hash, route_id, stage, code_commit, config_hash`。A3 输出供同 `route_id` 的 A4 使用；回读核对 `s` 与 `delta` 成对、共享角点一致、坐标恢复一致、无 NaN/Inf。四路线之间不得复用 A3/A4 优化结果、Adam 状态或已提取 mesh。已有 `/data/hi3d/g0-prep/` 目录只新增独立 run_id，不覆盖历史基线；不写 `/data/hi3d` 以外远端路径。

建议新增独立实验模块，不改原 Hi3D 推理入口或已固定的 vae3d 快照。主要参考点：`vae3d/refine.py` 的 `surface_plan/refine`、`vae3d/geometry.py` 的 `surface_loss`、`vae3d/render_refine.py` 的 `DynamicSurface/Renderer/fit`、`sparc3d/sparcubes/extract.py` 和 `sparc3d/ops/flexicubes/flexicubes.py`。新入口提供 `--extractor {mc,flexicubes}`、`--normal-mode {flat,smooth}`、`--resolution`、`--sampling-offset`、`--manifest`、`--output-root`、`--resume`，并将解析后的完整配置写盘。示例接口是**待实现要求**，不是现有命令。

## 3. 可微实现要求

### 3.1 提取器

- 抽象成统一的 `extract(s, delta, cells, corner_indices, grid_transform) -> (vertices, faces)`；MC 和 FlexiCubes 均从同一唯一角点 `[s,delta]`、同一支撑和归一化矩阵得到网格。重建/提取时 `s` 与 `delta` 不能拆开。禁止对两份提取网格分别居中、缩放、清理、补洞或简化后评测。
- A3 每次迭代须用所选提取器计算当前网格和几何损失。MC 允许在符号固定时缓存连接关系；FlexiCubes 也可缓存离散连接关系，但必须证明对连续 `s/delta` 的顶点位置梯度仍有效。A4 允许符号跨零，必须每步重新判断当前连接关系；拓扑选择本身不可微可以接受，当前已存在的曲面顶点到损失必须可微。空曲面、非有限梯度、断图要明确失败，不能静默跳过。
- 当前 Hi3D FlexiCubes 为推理裁剪版，删去了原上游部分训练分支及可训练权重；不可仅把 `point_to_mesh` 包进优化器便宣称可训练。最小实现先复用现有推理几何语义，验证场值与 delta 的反传；若确需恢复上游训练分支，应另建版本和因子，不能混入二因素主比较。注意 `run_g0b_asset.py` 当前在 `torch.no_grad()` 下只验证前向。
- 有效性约束按同一规则检查每条路线：共享角点、delta 上限、单元 Jacobian/倒置穿越 cell、空面、非流形与边界。A4 现有软体积约束不能保证没有翻转；标志要保留，不能借组件过滤消除失败样本。

### 3.2 法线与损失

- `flat`：预测和目标都按每个三角形的几何面法线，面内常量。保持原 A4 的轮廓、深度、无朝向法线损失定义。`smooth`：在**同一几何网格**的相容连通扇区计算角度加权角点法线，三角形内按重心坐标线性插值后归一化；非流形边、断开的顶点扇区和不一致绕序不得跨越平均。这个扇区规则、零面积面处理和数值 epsilon 必须固定并记录。
- A3 `surface_loss` 的正向和反向法线项都使用该模式；距离项、采样与权重保持相同。现有 `normal_target='crease'` **只改变目标法线**，不能直接充当 `smooth`。实现需让预测网格也使用插值法线，并验证梯度到顶点。A3 不能只给 smooth 路线改变法线目标而继续以 flat 预测法线算另一半。
- A4 的源目标薄棱柱仍为所有路线相同的 1 像素一致深度代理；轮廓和深度目标及视角不因法线模式改变。目标原表面的 smooth 角点法线从原 mesh 相容扇区计算；棱柱侧壁及封边保持独立 flat 扇区，不能把正面与侧壁平均。预测网格按同模式生成法线，再交给 nvdiffrast 重心插值。所有路线使用同一法线损失权重及其余损失，不额外只给 smooth 路线加平滑正则。
- 报告至少同时给出共同的**几何指标**和按两种法线评测器统一重算的 held-out 图像指标。各路线自己的训练法线损失不可直接跨模式比较；网页说明这一点。网页着色同路线仅为可视检查，不是训练损失的证据。

## 4. 实施顺序与验证门槛

每一关保存命令、resolved config、源码 commit、环境、日志和产物路径；失败记录完整保留。没有 GPU 实测的门槛只能标为待验收。

| 顺序 | 交付物 | 必须通过的验收 |
|---|---|---|
| 0. 锁定协议 | 新配置 schema、manifest/输入哈希、四路线 run_id、旧结果只读 | 同一资产四路线 A2/支撑/归一化/随机设置逐字段相同；缓存键不同；断点恢复拒绝哈希或配置不一致 |
| 1. 提取器前向 | A1/A2 的 MC/FlexiCubes 原精度 mesh；MC 与现有 A2_classic 数值对照 | MC 同输入顶点/面连接在既有浮点容差内一致；FlexiCubes 非空、有限、原坐标对齐；所有结果注明 R/相位/提取器 |
| 2. 梯度 smoke | MC、FlexiCubes × flat、smooth 的 A3/A4 单步 GPU 日志 | 4 路对 `s`（可移动非零交叉点）和 `delta` 有有限、非零梯度；法线项对预测顶点有有限梯度；有限差分方向与自动微分在固定拓扑区符号一致；单步后字段有限、delta 不越界，空面/断图显式报错 |
| 3. 小样本拟合 | 固定 index `48, 5, 7, 30` 的四路 A3/A4；含简单、复杂、G0b 回退与非闭合样本 | 4×4 资产 A3/A4 均可回读；每条 A4 只读取同路线 A3；至少保存初始/最终网格和优化 trace；共同几何及 held-out 评测可重复，失败不能被隐藏为 success；记录 GPU allocated/reserved 峰值和墙钟时间 |
| 4. 固定 50 件 | 四路 A3/A4 独立批量缓存、逐资产指标及汇总 | 每条路线 50 个资产逐一给 `success/failed` 与原因，不要求强行 50/50 成功；统计必须以同一可比较资产集合配对，并另外报告全 50 的覆盖率；原精度、原坐标、无清理网格评测；A0 疑点与倒置 cell 标志保留 |
| 5. UI | 下节规定的 2×5 预览与服务 API | 50 件索引/切换/隐藏列/四路线选择/同步相机/缺失结果状态按 UI 验收清单通过；慢速网络用本地预览缓存，不使一次失败阻塞其他格 |
| 6. 路线决策 | 固定协议的四路配对报告与人工复核清单 | 报告 Chamfer L1 中位数及 P90、F-score@0.5%/1%、法向一致性、边界/非流形、自相交及倒置 cell、held-out 图像误差、时间/显存；逐项给胜负、离群值和置信区间/离散程度；没有预设胜者。用户决定最终路线后才将该 route_id 写入后续训练配置 |

测试必须包含一个零损失已知例、一个有限差分梯度例、一个非流形/薄片输入例、一个允许跨零导致拓扑改变的 A4 例、一个恢复配置不匹配例。A3 和 A4 要分别验证，不得用最终 mesh 看起来正常替代反传检查。`mc_flat` 新实现先在相同输入/相位/配置下复现现有 R128 路线；若不一致，记录首个分歧并修复或明确版本差异，再进行二因素结论。

路线质量判定是**研究决策**，不是 GPU 成功率：不得以 50/50 运行成功宣称最佳。至少在同一 50 件及重点编号 `7, 46, 11, 40, 0, 25, 30` 上复核薄结构、开口、背壳、断面与翻转；任一质量疑点可否用于训练需逐资产决定。`flexi_smooth` 仅为优先候选；若未通过梯度、拓扑或资源门槛，不能将它设成默认训练目标。

## 5. 对比 UI：从三视窗改为 2×5

现有 `experiments/vae_sdf/g0b_compare.html`、`view_meshes.py`、`remote_g0b_bundle.py` 与本地预览缓存只提供原 mesh / A4 同场 MC / A4 同场 FlexiCubes 三视窗。新页面在 `/compare` 显示如下；旧 G0b 同场对比保留独立入口（建议 `/compare/legacy`）及明确的“历史基线”标签，不能伪装成重新拟合结果。

| 提取器行 | 原始 mesh | A1 | A2 | A3 | A4 |
|---|---|---|---|---|---|
| classic MC | 同一源网格 | A1/MC | A2/MC | 所选 `mc_*` 路线拟合结果 | 同路线 A4 |
| FlexiCubes | 同一源网格 | A1/FlexiCubes | A2/FlexiCubes | 所选 `flexi_*` 路线拟合结果 | 同路线 A4 |

顶栏提供法线实验选择 `flat / smooth`，切换时 A3/A4 加载对应**独立 run_id**；A1/A2 几何保持同一结果。四个 A1–A4 列按钮各控制两行同时显示/隐藏，至少保留原 mesh 列；不让“关掉 A2”改变 A3/A4 的真实输入。默认选 `smooth` 便于关注候选，但若该路线未完成，显示“未生成/失败”而非退回 flat。每个格显示 `route_id, stage, R, phase, 状态, 原精度面数, Chamfer`；完整指标和异常标志在详情区。颜色按阶段一致，行标题明确提取器。原 mesh 两格共享同一数据。所有格共用源 mesh 的坐标框与相机，左键旋转、中键平移、滚轮缩放、重置和线框继续可用。网页画面如需 flat/smooth 着色，应只作为另一个明确标为“显示着色”的选项，不能改变顶栏的实验路线，也不改变指标。

服务提供按 `manifest index + R + phase + route_id + stage + quality` 读取固定缓存的目录/mesh API；不得接受浏览器任意服务器路径。目录返回各格 `ready/missing/failed`、哈希、审计与指标；请求缺失格必须有可读错误。原精度 mesh 与快速预览分离：预览只为加载速度，可简化、压缩、本地缓存；指标始终来自未经简化的原精度网格。缓存按 `source_hash, route_id, stage, R, phase, mesh_hash, preview_version` 失效，不能把旧 A4 G0b 预览当作新路线。按需加载当前可见格，隐藏格不请求；切换资产/路线应取消或忽略过期响应。尽量共用一个 WebGL2 上下文/画布分视口，避免十个同时存在的上下文与重复 GPU buffer；原 mesh只上传一次。原精度大网格失败不应抹掉已加载的其他格。

UI 验收：选择 index 7 后 10 格标题、route_id、源 mesh 哈希与缓存实际一致；关 A3 后两行 A3 一起消失、其他格及相机不变；flat↔smooth 后 A1/A2 的 mesh 哈希不变，A3/A4 哈希切换且来自对应配置；切换 index 或精度无旧响应串画；相机在任意格操作同步；缺失路线清楚显示状态；快速预览与原精度同源且指标不因质量档改变；旧 G0b 页面仍可进入。用浏览器实际交互验收并保存截图/日志，不能只做 HTML 静态检查。

## 6. R1024 / R2048 与训练路线锁定

四路线先在 R128 验证算法，不把 R128 质量排序直接当作 R1024/2048 结论。完成相位 0 与 `Point.matrix` 审计后，另建 R256/512 缓存，测每资产细层点/cell 数、A3/A4 生成时间、GPU allocated/reserved/驱动占用、峰值 CPU RAM 和磁盘；在同一目标上做一次可微网络前向/反向与预测支撑评测。R1024 是首个高分辨率训练目标；R2048 只在分块/多尺度**可微**反传、块边界共享点一致、全局支撑一致、网格无明显接缝并有可恢复训练证据后放行。现有 `inference_mode` 流式 decoder 和多 GPU 流式推理不算训练路径。

放行 R1024/2048 前，选定路线的目标生成器、训练时预测网格提取器、法线损失、采样相位/矩阵与缓存 schema 必须固定并在 checkpoint/resolved config 中可追溯。若实际训练只用场/位移损失、没有 mesh/render loss，明确记录“训练未运行该提取器的可微路径”，但目标生成与最终网格提取仍须采用选定路线；不能据此宣称整个训练链已使用同一可微几何路径。高分辨率若更改支撑、损失或分块策略，另立协议重新验收，不复用 R128 质量结论。

## 7. 执行记录模板

每个阶段报告填：`date, git_commit, vae3d_commit, manifest/source hashes, R, phase, normalization_matrix, route_id, stage, A2_hash, support_hash, full resolved_config, GPU/driver/CUDA/PyTorch versions, exact command, resume command, wall_time, peak allocated/reserved/driver memory, output path, asset status, gradient checks, mesh metrics, held-out metrics, inversion/topology audit, UI screenshot/API evidence, known failures, next action`。区分“源码核查”“静态检查”“GPU 实测”“浏览器实测”；缺字段写“未测”，不能推断通过。
