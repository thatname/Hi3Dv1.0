# 基于 Hi3D 与 vae3d 优化场目标的 3D VAE 实验方案

日期：2026-10-29  
版本：v0.3，供决策与远程实施使用。已完成静态源码核查；尚未修改训练代码、连接远程主机或启动实验。

阅读入口：[会话交接与决策状态](HANDOFF.md)。本文是设计方案，其中建议项不自动视为用户已确认；实际进度和可运行命令以交接及阶段报告为准。

源码基线：

- Hi3D 实验基线：本地 Hi3Dv1.0-modified，commit `5e5622d0baea953e48bde31607936f1ac1036be6`，迁移为 `/Users/changle/code/3dgen/hi3d-vae-field`。原官方基线 `13657ad09afaa92e350b461a1cd66399e44a59c4` 用于差异比较。
- 本地 `/Users/changle/code/3dgen/vae3d`：HEAD `422e79988ccda06ef0995bf7768cd05a8894a124`；核查时仅见未跟踪 `.vscode/`。
- 目标生成入口：`prepare_and_train_t1.bat`。本方案记录其默认参数；环境变量、附加命令行参数及恢复 checkpoint 均须在实际运行时核对。

## 0. 修改版 Hi3D 基线增补（v0.3）

修改版增加流式多 GPU 解码、分块提取与2048档位，但仍是推理系统。训练缺口及新增 G0c 一致性验证的完整依据统一维护在 [修改版基线分析](modified-repo-analysis.md)，避免两份文档重复漂移。

首轮训练采用普通可微 decoder；流式路径需适配新输出布局并通过 G0c 后才能用于新模型验证。2048不作为首轮训练分辨率。G0、G0b、G1/G2 设计见下文，G0c 要求见基线分析。原 v0.2 仅为历史归档。

## 1. 目标生成方案与决策

研究对象明确为：**将 vae3d 的“UE50 背面投票 + 25% 门控背壳 + 几何拟合 + 一致深度渲染拟合”目标生成流程迁入 Hi3D 派生 VAE，验证重建质量与训练成本。** 这不是只替换欧氏距离公式。最终监督是成对的有符号标量场与采样点位移，并非严格欧氏 SDF。

主要修订：

1. SDF 来源从“待提供”改为已核查的具体实现。
2. 标签生成、encoder/posterior、训练器、损失与缓存已有 vae3d 参考实现，工作重点改为迁移适配及验证，不再全部从零开发。
3. 完整目标的 deformation 必须与场值配对使用；固定零位移只适用于初始场对照，不能将拟合后位移直接丢弃。
4. 增加背壳、几何拟合、渲染拟合的逐阶段消融，以及中心点到 Hi3D 角点表示的适配验收。
5. 原 T1 为倍率 4 / latent 8 维；Hi3D decoder 为倍率 8 / latent 32 维。迁移 encoder 与层级支撑需要结构调整。

第一阶段只训练 **mesh → 目标场 → encoder → latent → decoder → mesh**。图片、图像编码器和 DiT 不在首轮训练范围内。这不是复现 Hi3D 未公开的原始训练方案。

| 决策项 | 建议默认方案 | 影响 |
|---|---|---|
| 目标生成范围 | 迁入完整流程，并保留分阶段中间结果 | 可归因背壳/几何/渲染分别的收益 |
| 编码器输入 | 稀疏采样点的 `[s, dx, dy, dz]` | 复用 vae3d 已有思路；表面点/法向 encoder 留作独立扩展 |
| 训练架构 | vae3d 派生 encoder 调至倍率 8、latent 32；Hi3D decoder 骨干 | 是新增适配架构，不保证原有 encoder 权重可加载 |
| 输出接口 | 先验证可表达完整目标的连续场头，保留独立符号头对照 | 与原 Hi3D 输出语义不同，不能承诺旧 DiT 兼容 |
| 参数初始化 | Hi3D 骨干热启动候选，修改输出头重新初始化 | 不宣称整模型严格加载；各实验使用相同初始化规则 |
| 远程环境 | Linux 时先验证 Vulkan 射线或 CPU 参考后端 | Windows BAT 的 DXR 参数不可原样用于 Linux |
| 数据、GPU、预算 | 用户确定 | 决定分辨率、拟合耗时、训练规模 |
| 是否保留旧 DiT | 第一阶段不要求 | 若必须兼容，增加潜空间对齐/生成器微调里程碑 |

目前需要用户决定的是路线范围、旧 DiT 兼容要求、数据和算力预算；新 SDF 的代码入口已经明确，无需再次提供。

## 2. 已核查的目标生成流程

调用链：`prepare_and_train_t1.bat → prepare_and_train_t1.py → prepare_and_train.main('t1') → prepare_pipeline.convert → ue_grid(field_mode='back_shell') → back_shell_grid → refine → render_refine.fit → deduplicate_fitted → select_sparsity`。缓存命中时可直接读取目标，必须核对缓存设置与版本。

### 2.1 归一化、采样与欧氏距离

mesh 按包围盒中心平移，最长边缩放为 `(R-2)/R`，位于 `[-0.5,0.5]³` 内并留边界余量。默认 R=256。

中心点阵：`p=(ijk+0.5)/R-0.5`，共 R³ 个候选采样点。其相邻八点构成曲面提取立方体，不能混同于原始未平移网格角点。

无符号距离仍为到最近三角形的欧氏距离 `d(p)=min_q ||p-q||₂`。默认 CUDA/Warp BVH 后端，初始上限 `4√3/R`；距离查询同时确定点是否处于最近三角形背侧。

### 2.2 UE50 投票与门控背壳

球面分层采样 98 个方向，射线长度覆盖归一化域，使用最近命中与三角形法向判断背面。每条方向最多一票，未命中也计入总方向分母。背面命中数 B>49 时取负，否则取正：`f_base = ±min(d,4√3/R)`。

然后用 `0.25-B/98` 的零等值面构造门控曲面，查询门控有符号距离 g：B≥25 的区域为负。对最近三角形背侧且距离满足代码局部条件的点，构造：

```text
r = √3/R
f_shell = max(-d, d-r, g)
f_init  = min(f_base, f_shell)
```

即在 25% 投票门控范围内补充约 1.73 体素厚的背壳。未选中的点保持 f_base。实现使用投票场插值得到的门控曲面，不是二值掩码的距离变换。

限制：依赖三角形朝向；所有三角形按单面处理；这不是严格实体内外判定，也不是 UE 源码的逐位复现。人工背面可改变零等值面，因此不得写成“保持原 mesh 零面不变的 SDF 替换”。

### 2.3 稀疏窄带与 30 步几何拟合

根据近表面条件 `|f|<3/R` 及符号交叉 cells 建立支撑，再膨胀两层并补齐由保留点构成的立方体。初始值转为体素单位：`s=clip(R*f_init,-4,4)`，位移初始为零。

默认 `refine(...,steps=30,mode='joint',max_deformation=0.5)`：共享采样点位移和场值幅度一起优化；场值乘正数因子，保持符号、精确零值及缓存的 MC 连接关系。目标包括表面距离、法向、位移一致性和缩放正则。使用局部体积检查与回退限制单元翻转。

### 2.4 200 步一致深度渲染拟合

默认 4096 像素、32 个训练视角，每步抽 2 个视角，另有 12 个留出视角；继续优化 s 与位移。调用明确设置 `allow_sign_changes=True`、`freeze_boundary=False`，不能按函数默认值误判为固定符号或固定边界。

参考几何由原 mesh 各三角形沿负法向挤出薄棱柱构成，厚度 `1.3/image_resolution`，约对应 1 个监督像素。轮廓、深度与法向来自同一代理，避免目标不一致。它不同于初始化阶段的背壳厚度。

```text
L_render = view_weight * (L_silhouette + 10*L_depth + 0.1*L_normal)
           + 10*L_cell_validity + 0.001*mean(delta²)
```

法向使用无朝向比较；掠射视角降权。场值直接优化，可跨零并重算 MC 拓扑，每步限制到 [-4,4]；位移各分量限制到 [-0.5,0.5] 体素。此阶段仅有软体积正则，不保证绝不翻转；固定支撑也不能凭图像梯度恢复完全缺失的孤立表面。

这些是每个资产的标签拟合步骤，不是 VAE 参数训练步数。其耗时必须单独计入预算。

### 2.5 最终监督与网络输出头

将拟合时的每 cell 32 通道表示去重，缓存唯一点：

```text
coords[N,3]             # 中心点阵索引
features[N,4]          # [s, dx, dy, dz]
cells[K,3]
cell_indices[K,8]
p_deformed = (coords + 0.5 + delta)/R - 0.5
```

`full-band` 保留完整拟合窄带，不等于保存全空间稠密场。代码检查共享点值一致；提取使用经典 Marching Cubes 连接及连续场插值，不是 Hi3D 当前 FlexiCubes 的等价实现。

入口指定 `signed-tanh` 网络头：`s_hat=-tanh(l)*(softplus(m)+1e-5)`，位移为 `0.5*tanh(raw_delta)`。signed-tanh 的整体场值并不被限制到 [-1,1]，也不是普通 tanh SDF 输出。它是连续有符号值参数化，不是标签生成算法。

`--sdf-truncation 4` 对饱和标签使用单侧损失：预测超过同号截断边界不再受罚；近表面权重由 `--sdf-surface-weight 4` 控制。不要将此参数解释为简单地把所有预测强制截断。入口还带有训练辅助项配置，复现实验必须保存完整 resolved_config，而非仅记录 SDF 参数。

## 3. 两个仓库的可复用部分与剩余缺口

| 模块 | 已有实现 | 实施任务 |
|---|---|---|
| 目标场与形变生成 | vae3d `back_shell.py/refine.py/render_refine.py/prepare_pipeline.py` | 提取独立接口；缓存各阶段；验证远程后端 |
| 数据与缓存 | vae3d `data.py/target_cache.py/prepare_and_train.py` | 复用逻辑，补实验分组、资产去重、版本和支撑规则 |
| encoder 与 Gaussian | vae3d `model.py`，输入 4 通道，有 mean/logvar/sample | T1 的倍率4/8维改为 Hi3D 所需倍率8/32维，审查 tensor/坐标接口 |
| 稀疏 decoder | Hi3D PTv3 decoder，可加载发布权重 | 增加训练层级监督、坐标适配、输出头；原推理路径保留 |
| occupancy 监督 | vae3d 训练返回 logits/labels；Hi3D 当前硬裁剪 | 移植训练思路；按倍率8重新构建各级 GT，不能复制 T1 的 latent_coords |
| 损失与训练循环 | vae3d 有场/形变/occupancy/validity/KL、AMP、恢复 | 适配 Hi3D Point 数据结构、batch、输出接口，实测梯度 |
| 网格提取 | vae3d MC；Hi3D 推理 FlexiCubes | 先做同目标双提取器测试；需要 mesh loss 时恢复/验证训练路径 |
| 评测 | vae3d 有 evaluate、quality、诊断等入口 | 审查指标，统一协议，固定数据集与失败统计 |
| DiT | Hi3D 发布模型 | 后续验证新 latent 对齐，不能按维度判断兼容 |

Hi3D 仍需要特别处理：`SerializedUnpooling` 中 `pred_octree>0` 的离散裁剪；超过点数上限就跳过残差层的推理保护；严格 runtime config；只支持单样本网格提取；FlexiCubes 删除了部分训练分支。不能仅增加 train() 和 backward() 就认为可训练。

## 4. 推荐架构与表示适配

### 4.1 主路线

```text
mesh → 归一化 → UE50/背壳 → 几何拟合 → 渲染拟合
     → 唯一中心点 [s,delta] + 支撑 + 目标 mesh
     → 稀疏 encoder（倍率8）→ μ/logvar[Nz,32] → 重参数采样
     → Hi3D 派生训练 decoder（三级×2）
     → 每 cell 8 个连续 s + 24 个 delta + 各级支撑 logits
     → 共享角点聚合 → 提取 mesh

唯一中心点目标 → cell_indices 打包 → 输出监督
N(0,I) → KL；目标 mesh/原 mesh → 分开评测
```

encoder 的输入本身就是待压缩的目标场，这属于场 autoencoding，不是从原始图片生成几何。不同目标组若同时改变输入与输出，结论应表述为“表示可压缩性及系统重建收益”，不能声称是纯监督标签变化的因果效果。若需要隔离监督效果，新增统一 encoder 输入的控制组；不要混入首轮主结论。

不提供高分辨率 encoder 特征到 decoder 的跳连，避免绕过瓶颈。稀疏 latent 是 `(C,z)`，坐标也需计入压缩开销；采样 z 不会自动产生 C。

### 4.2 中心点到 Hi3D 角点的关键契约

| 差异 | 必须落实的适配 |
|---|---|
| 中心点 vs cell 角点 | 用 cell_indices 无损打包 s 和 delta，显式转换角点顺序；不得直接 reshape |
| 半体素偏移 | 提取坐标使用 `(ijk+0.5+delta)/R-0.5`；将 +0.5 放入坐标映射，不能塞进有界 delta |
| 共享点副本 | 聚合前后数值一致；损失按唯一点或除以重复次数归一化 |
| 数值范围 | 目标 s 以细体素为单位，拟合终值 [-4,4]；不能直接监督 Hi3D 原始 [-1,1] tanh 通道 |
| 符号规则 | vae3d s<0 为内部；Hi3D 原 sign 分支约定须显式映射 |
| 压缩倍率 | T1 当前倍率4、latent8；Hi3D 倍率8、latent32；重建 occupancy hierarchy 和 encoder |
| 提取器 | 同一场通过 MC 与 FlexiCubes 结果未必相同，先验证提取误差再训练 |

推荐先保持 Hi3D 骨干，在训练分支改输出头为每 cell 8×5 个 raw 值：每点 l、m、三维 delta，转换为 8 s + 24 delta 后聚合。去除该分支原 `Point.activate` 的重复激活及提取器的硬符号覆盖；聚合真实有符号值和真实位移。这是建议设计，尚未实现。

保留原 8 sign + 8 distance + 24 delta 头作为对照可行，但需定义幅度激活、符号 BCE、归一化与共享角点规则；不得宣称两种头等价。若用 s/4 映射范围，全局缩放本身可保持线性插值比值，但输出饱和、独立符号覆盖与训练梯度仍需验证。

适配验收必须绕过神经网络：把缓存目标打包为 Hi3D 输出格式，验证唯一点回读误差、共享点一致性、归一化空间中的几何位置。先用同一 MC 实现比较应一致的结果，再独立测 FlexiCubes 替换误差。不要对两份网格分别重新归一化来掩盖半体素偏移。

## 5. 训练监督与验证策略

完整拟合目标 s 与 delta 是联合优化产物，作为一对监督。不要重新计算原 mesh 欧氏距离覆盖拟合 s，也不要保留拟合 s 却将 delta 设零。零位移消融应重新生成约束为零的拟合目标，或仅在未拟合的初始场进行。

首轮连续头训练建议：

```text
L = L_field + λdelta*L_delta + λsupport*L_support
    + λvalid*L_validity + β(t)*L_KL + λshared*L_shared
```

- field：复用 vae3d SmoothL1、近表面权重及饱和单侧损失，单位为细体素。
- delta：监督三维拟合位移，检查激活和单位，不能与世界坐标混用。
- support：三级 child logits 的 BCE，occupancy 指保留稀疏体素，不是 s 的正负。
- validity：变形 cell 的有效性正则，单独报告翻转率；软约束不提供无翻转保证。
- KL：`0.5*mean(mu²+exp(logvar)-1-logvar)`；每资产内归一化，再对 batch 求平均。
- shared：cell 副本间一致性或聚合后的唯一点监督；避免重复加权。

连续 signed-tanh 头默认不需要独立硬符号 BCE；原 Hi3D 符号头对照则需添加该项，记录配置差异。自定义场不默认加入欧氏 Eikonal 约束。

先 β=0 做梯度与小样本拟合诊断，此时是 AE 调试；之后 KL warm-up，候选 β=1e-6/1e-5/1e-4 仅作起点。训练 sample、主验证用 μ，另报告采样波动与 posterior collapse 指标。

初期 GT 支撑选取子体素，同时监督预测 logits。最终评价必须使用模型预测的后续支撑；GT 支撑结果仅为诊断。粗层 C 的来源、失配策略、预算与失败情况都需记录。

标签生成时已使用几何/渲染损失，不意味着 VAE 训练也必须执行同样的昂贵渲染。主实验先做场与位移监督；额外 mesh/render loss 独立消融。不同目标组的 VAE 辅助损失、训练预算和初始化保持一致，不能只给某组加入法向特征损失。

## 6. 分阶段对照实验

### 6.1 G0：先验证目标表示，不训练 VAE

| 编号 | 目标 | 位移 | 用途 |
|---|---|---|---|
| A0 | 可靠闭合 mesh 的欧氏有符号距离 | 0 | 传统表示基准；开口模型不伪造可靠 GT 符号 |
| A1 | UE50 投票场，不加背壳 | 0 | 投票符号对照 |
| A2 | UE50 + 25% 门控背壳 | 0 | 测背壳收益与人工厚度代价 |
| A3 | A2 + 30 步几何拟合 | 成对拟合 delta | 测几何拟合增量 |
| A4 | A3 + 200 步一致深度渲染拟合 | 成对拟合 delta | 完整参考目标 |

闭合子集做 A0–A4；开口/薄片子集重点做 A1–A4。若希望区分渲染代理厚度的贡献，再做相同拟合预算、目标厚度为零的控制组。逐阶段消融首先反映“增加模块和计算”的系统收益；若要声称优化方法在同预算下更优，再增加等计算预算对照。

对照先使用共同支撑，建议冻结各初始场窄带支撑的并集并完成完整 cell 构造；另报告各方法原生支撑的系统表现。并集无法保证包含所有可见细节，须记录覆盖缺失。固定 R、采样相位、MC 提取、输入 mesh 和随机种子，先报告未经补洞/简化结果。

A1/A2、中间缓存以及跳过渲染需要新增实验入口或阶段开关；现有 convert 强制执行 fit 调用，启动参数要求 render_steps≥1，不能杜撰现成 `--render-steps 0` 命令。

### 6.2 G0b：迁移接口验收

同一 A4 缓存先经过打包/回读，再通过两种提取路径。分别记录无损转换误差和 MC→FlexiCubes 的算法差异。若 Hi3D 提取明显损害薄结构，先修接口或明确选择保留 MC，不把损失归咎于 VAE。

### 6.3 G1–G4：逐步训练

| 阶段 | 内容 | 放行条件 |
|---|---|---|
| G1 | 1 个简单体 + 1 个复杂体，再 8–16 个；GT 支撑、β=0 | encoder/decoder/support 均有有限梯度，损失下降，重建改善 |
| G2 | 100–500 个资产试验，固定验证集；优先 B2(A2目标) 与 B4(A4目标) | 预测支撑条件下也能重建；完整目标的收益未被压缩损失抵消 |
| G3 | 增加 B1/B3 或零位移约束目标、3 个种子及更高分辨率 | 收益稳定，时间/显存/标签成本可接受 |
| G4 | 用户需要时接回 Hi3D 图片生成 | latent/坐标/尺度与旧生成器兼容，或明确微调生成器 |

B2/B4 使用同一新 VAE 架构、输出头、初始化、数据划分和步数；分别自编码其目标场。增加 vae3d 原 T1 作为可运行参考线，但 T1 与 Hi3D 派生模型的容量、倍率、提取器不同，结果只能作为系统比较，不能作为纯 SDF 消融。

评测分三层：①目标提取 mesh 对原 mesh（表示误差）；②VAE 重建对目标提取 mesh（压缩误差）；③VAE 重建对原 mesh（最终误差）。三者分别报告，不假设数值可直接相加。

## 7. 数据、缓存与评测协议

按资产去重和划分 train/val/test，派生分辨率、旋转和同源文件不可跨集合。数据分为闭合、开口、薄片/细杆、法向异常等子集；原算法不做法向修复，若引入修复必须单独标记，不能默默改变背面投票。

缓存记录：源文件哈希、归一化 center/scale、R、sampling_offset=0.5、目标阶段 A0–A4、唯一点 features、cell_indices、原 mesh、目标 mesh、各级 support、背壳参数、拟合步数、渲染分辨率/视角/种子、后端、prepare/schema 版本、代码 commit、实际命令配置和 audit。不同阶段与后端建立明确缓存身份，不共用错误标签。

不把投票负值当作开口 mesh 的真实实体内部标签；符号准确率在此表示对目标标签的拟合程度。目标场是优化场，不使用欧氏 SDF MAE 作为唯一质量结论。

统一指标：

- 双向表面 Chamfer-L1、F-score（阈值先选包围盒对角线 0.5%/1%，正式实验前锁定）。
- 法向一致性、薄结构召回、尖锐区域误差、多视角轮廓/深度；区分原始 mesh 与薄棱柱代理作为参考的指标。
- 人工厚度、背壳偏移、空 mesh、连通分量、边界/非流形边与变形单元翻转率。
- 场/位移损失、各层支撑 precision/recall、latent μ/σ/KL、均值与采样重建差异。
- 预处理各阶段时间与显存、训练稳态时间和峰值显存、token 分布、缓存体积、失败率。

每资产先计算再求均值/中位数/尾部，失败不得从分母删除。固定表面采样点数与种子，主结果使用 raw mesh，统一后处理作为次要报告。渲染标签拟合的留出视角不等于资产级 VAE 测试集。

## 8. 远程实施与资源验证

### 8.1 需要迁移或新增的模块

以下为拟实施路径，不是当前可运行命令：

```text
experiments/vae_sdf/
  configs/                         # 阶段、后端、输出头与训练配置
  data/target_pipeline.py           # 包装 vae3d 目标生成，新增阶段开关
  data/field_adapter.py             # 中心点/角点/坐标单位转换
  data/manifest.py                  # 资产划分、缓存身份与统计
  models/encoder.py                 # 倍率8 / latent32 适配
  models/trainable_decoder.py       # Hi3D骨干 + 训练支撑 + 新输出头
  losses.py                        # 复用并适配 vae3d
  train.py                         # 复用训练/恢复逻辑
  evaluate.py                      # 表示/压缩/最终误差
  preflight.py                     # 环境、后端、梯度、资源检查
  tests/                           # 相位、角点顺序、打包、梯度、恢复
```

保留原推理路径与 checkpoint；新训练模型使用独立配置及权重导出版本。输出头改变后，明确列出已加载/未加载权重，不能以 strict=False 掩盖缺失参数。

### 8.2 跨平台和设备问题

Windows BAT 默认 `--raycast-backend dxr --distance-backend cuda`。Linux 应评估 Vulkan/SlangPy 射线后端是否可用；不可用时用 CPU 射线作正确性参考并量测性能，CUDA 距离可单独保留。切后端属于实验配置改变，需比较投票数、符号、距离和最终 mesh，不预设逐位一致。

目标拟合还依赖 Kaolin、nvdiffrast、Warp、Open3D 等；训练依赖 spconv，Hi3D 骨干还需 flash-attn。核查两仓库依赖与 CUDA/PyTorch 组合，安装后分别测试 forward/backward。渲染 auto 在 Windows 偏向 OpenGL、其他平台为 CUDA，远程无显示环境需验证实际选定后端。

当前预处理路径要求 cuda:0，部分内部直接使用 device='cuda'；多卡通过单进程 CUDA_VISIBLE_DEVICES 映射物理卡后再验证，不能只传 device='cuda:1' 就假设支持。首轮建议先离线生成并缓存目标，再训练，避免 4096 渲染拟合与 VAE 反向争抢显存；在线 prepare+train 作为后续选项。

### 8.3 所需信息及运行门槛

用户仍需提供：远程主机/工作目录、GPU型号/显存/数量、CPU/RAM/磁盘、数据目录及规模、Hi3D 权重位置、GPU小时预算及旧 DiT 兼容要求。vae3d 路径和实现已明确，只需同步固定版本及所需依赖，不再要求重新描述算法。

顺序：

```text
固定双仓库版本与manifest → 后端查询对照 → A0–A4目标缓存
 → G0表示评测 → G0b适配与提取验证
 → forward/backward smoke → G1过拟合
 → B2/B4小规模训练 → 固定测试集评测 → 决定扩大实验
```

阶段开关、Linux 启动脚本与真正可执行命令在实现后交付。原 BAT 会对多数失败自动重启；远程实验应使用有限重试并记录原因，持续 OOM/NaN/配置错误不可进入无限循环。

先测每资产 UDF/投票/门控/几何/渲染/缓存耗时，以及 20–50 个稳定训练 step。标签准备与训练分别计费，不能把 200 步渲染拟合误算为一次简单 SDF 查询。R=128/256、低图像分辨率可用于 smoke，但图像分辨率会改变 1 像素代理的世界厚度，因此不能把这类配置混入 4096 默认基准；应显式固定世界厚度或标为另一实验。

先单 GPU microbatch=1，根据 token 数设预算，再用梯度累积或 checkpointing；不承诺固定卡型显存足够。禁止超限时静默跳过网络层或丢弃复杂资产。错误场值、全空曲面、翻转失控、支撑爆炸、缓存错版本、NaN/Inf 均保存诊断并停止对应 run。

断点保存模型、optimizer、scheduler、AMP scaler、随机状态、数据游标/采样状态、步数与 resolved_config。每 run 输出 manifest、环境版本、权重/代码哈希、metrics、checkpoint、固定样本 mesh/截图、阶段耗时、失败清单。验证恢复连续性，不要求稀疏 GPU 算子逐位确定。

## 9. 旧 DiT 兼容与验收

新目标、新 encoder、输出头以及采样相位都会改变表示。32 维 latent、标准正态 KL 或相似均值方差均不能证明旧 DiT 兼容。

独立 VAE 路线最适合先验证表示与压缩；如要接回图片生成，可选旧 latent 蒸馏约束或用新 encoder 编码图片–mesh 配对集再微调 Flow/DiT。必须另处理 GT 目标支撑与实际粗 mesh 支撑失配，并重新统计 latent 归一化参数。固定旧 decoder 可作为编码器诊断，但不足以验证完整新表示的优势。

首轮验收：目标生成流程可复现；表示适配无未解释的坐标/单位误差；B2/B4 在预测支撑条件下给出成对结果；收益能区分初始化、拟合、压缩与提取器影响；资源成本可量测。现阶段只完成源码核查，尚无实验结果证明质量改善。

用户决定实施后，交付应为最小实验分支、锁定配置、G0/G0b/G1 报告和按实测资源确定的远程运行清单。若 G0 或适配验收失败，先修目标/接口，不扩大 VAE 训练。

## 附录 A：vae3d 源码索引

路径基于本次本地 checkout；远程执行时以固定 commit 的仓库相对路径定位。

- [入口默认参数](/Users/changle/code/3dgen/vae3d/prepare_and_train_t1.bat)
- [目标生成总流程](/Users/changle/code/3dgen/vae3d/vae3d/prepare_pipeline.py)
- [投票方向与旧分支分流](/Users/changle/code/3dgen/vae3d/vae3d/ue_sign.py)
- [当前 UE50 背壳实现](/Users/changle/code/3dgen/vae3d/vae3d/back_shell.py)
- [CUDA 欧氏距离](/Users/changle/code/3dgen/vae3d/vae3d/gpu_distance.py)
- [几何拟合](/Users/changle/code/3dgen/vae3d/vae3d/refine.py)
- [一致深度渲染拟合](/Users/changle/code/3dgen/vae3d/vae3d/render_refine.py)
- [唯一中心点表示与 MC](/Users/changle/code/3dgen/vae3d/vae3d/center_points.py)
- [支撑与归一化](/Users/changle/code/3dgen/vae3d/vae3d/occupancy.py)
- [encoder/posterior 与架构配置](/Users/changle/code/3dgen/vae3d/vae3d/model.py)
- [连续预测头](/Users/changle/code/3dgen/vae3d/vae3d/prediction.py)
- [训练损失](/Users/changle/code/3dgen/vae3d/vae3d/losses.py)
- [训练入口及恢复](/Users/changle/code/3dgen/vae3d/vae3d/train.py)

## 附录 B：Hi3D 原始组件源码依据

以下固定原始版本链接用于未变化组件；修改版新增/改变的执行路径以本目录 `modified-repo-analysis.md` 和当前本地源码为准。

以下链接固定到本次核对 commit：

- [公开仓库范围与推理限制](https://github.com/hi3d-official/Hi3Dv1.0/blob/13657ad09afaa92e350b461a1cd66399e44a59c4/README.zh-CN.md)
- [SparconvVAE：仅 decoder](https://github.com/hi3d-official/Hi3Dv1.0/blob/13657ad09afaa92e350b461a1cd66399e44a59c4/sparc3d/models/autoencoders/sparconv_vae.py)
- [PTv3 decoder 架构](https://github.com/hi3d-official/Hi3Dv1.0/blob/13657ad09afaa92e350b461a1cd66399e44a59c4/sparc3d/models/autoencoders/ptv3.py)
- [上采样裁剪与残差块保护](https://github.com/hi3d-official/Hi3Dv1.0/blob/13657ad09afaa92e350b461a1cd66399e44a59c4/sparc3d/models/autoencoders/blocks.py)
- [属性布局、激活与单位](https://github.com/hi3d-official/Hi3Dv1.0/blob/13657ad09afaa92e350b461a1cd66399e44a59c4/sparc3d/structures/point.py)
- [网格提取与独立符号](https://github.com/hi3d-official/Hi3Dv1.0/blob/13657ad09afaa92e350b461a1cd66399e44a59c4/sparc3d/sparcubes/extract.py)
- [共享角点与索引](https://github.com/hi3d-official/Hi3Dv1.0/blob/13657ad09afaa92e350b461a1cd66399e44a59c4/sparc3d/sparcubes/sparse_cube.py)
- [FlexiCubes 删除的训练能力](https://github.com/hi3d-official/Hi3Dv1.0/blob/13657ad09afaa92e350b461a1cd66399e44a59c4/sparc3d/ops/flexicubes/flexicubes.py)
- [runtime config 与加载约束](https://github.com/hi3d-official/Hi3Dv1.0/blob/13657ad09afaa92e350b461a1cd66399e44a59c4/sparc3d/models/autoencoders/config.py)
