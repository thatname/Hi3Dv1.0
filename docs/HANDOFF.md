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
| 实验代码 | 尚未创建 experiments/vae_sdf；方案里的目录和执行顺序均为规划 |
| 目标缓存/实验结果 | 本工程尚未产生 G0–G4 目标缓存、指标或训练 checkpoint |
| GPU 验证 | 尚未执行本项目目标迁移、梯度、多 GPU 等价性或 2048 性能验证 |
| 本轮提交范围 | 仅交接文档、阅读入口与链接修正；push 由用户操作 |

新会话先运行以下只读检查（从工程根目录执行）：

```bash
git status --short --branch
git log -3 --oneline
git remote -v
git branch -vv
```

核查时 remote 名为 origin 和 github，本地 main 跟踪 origin/main；这仅是当前机器的配置，clone 后可能不同。不要把此前对话中示例 remote 名 gitlab 写死到脚本，也不要把含凭据的地址复制进文档。

## 2. 已确认要求与尚未确认的设计

**用户已明确：**基于修改版 Hi3D 做实验；目标生成参考 vae3d 的 prepare_and_train_t1.bat；工程名称为 hi3d-vae-field；形成可供决策、用于远程实验的文档；本轮补充后本地提交，由用户 push。

**技术事实：**目标生成包含 UE50 投票、门控背壳、几何拟合和渲染拟合；最终是配对的场与形变；两仓库编码布局/倍率不同；当前流式解码只能作为推理路径。

**方案建议，尚非用户逐项确认：**使用4通道场输入 encoder、调整为倍率8/latent32、采用连续 signed-tanh 输出头、先独立 VAE 后接 DiT、使用骨干热启动、首轮 B2/B4 对照。实施相关改动前结合当前用户指令确定范围，不能以“文档写了”当作已批准的最终架构。

**尚缺运行信息：**远程主机与工程目录、GPU/显存/数量、CPU/RAM/磁盘、数据集路径/划分、权重位置、GPU小时预算、是否必须保留旧 DiT。不要从历史推理任务推断当前训练资源仍可用。

## 3. 下一项可交付任务

建议首个实现任务是“独立目标生成入口与 G0 表示验证”，暂不改网络骨干。

1. 固定 vae3d 源码版本，列出移植依赖与许可；设计独立 CLI，输入 mesh、输出目录、阶段和参数。
2. 提取目标流水线，支持 A1（投票）、A2（背壳）、A3（几何拟合）、A4（渲染拟合），可靠闭合数据另做 A0 欧氏 SDF 基准。
3. 保存每阶段目标场、delta、索引/支撑、归一化、完整配置、耗时、失败原因与 raw mesh。阶段裁剪前后规则明确。
4. 用同一批固定资产和共同支撑做目标表示对照；保留各方法原生支撑作为补充。
5. 输出 G0 报告；之后进行 G0b 中心点/角点适配，再进入梯度和训练任务。原模型流式一致性 G0c 可独立验证，新输出头实现后需重测。

这是后续任务规格，不表示已有可运行入口。现有 convert 不支持直接用 render_steps=0 跳过渲染；需要真正实现阶段开关，不能写出未实现的命令。

首个交付验收：CLI帮助与配置可审查；阶段与缓存隔离；固定小样本能导出预期数据与 raw mesh；字段 shape/单位/共享点一致性通过；记录失败且不静默改参数。数值阈值在基准运行前依据分辨率和误差定义锁定，当前尚无实测阈值。

## 4. vae3d 获取与依赖

vae3d 不是当前工程已纳入的代码或 submodule。只 clone 本工程无法执行其预处理。当前机器位置为 `/Users/changle/code/3dgen/vae3d`，入口为仓库相对路径 `prepare_and_train_t1.bat`；该绝对路径仅用于定位，远程不要写死。

首选获取方式：从用户提供的、可访问的 vae3d 仓库地址获取指定 commit。此仓库的远程访问地址与权限未在交接中验证，故不编造 clone URL。若远程无法访问，可从现有本地 checkout 导出固定提交快照并传输；不用复制 .vscode 或整个开发目录。

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

## 5. 远程运行记录模板

此表尚待填写；凭据、令牌和 SSH 私钥不写入仓库。

| 字段 | 当前状态 |
|---|---|
| 主机/调度环境、工作目录 | 未指定 |
| GPU/显存/数量，CPU/RAM/空闲磁盘 | 未测量 |
| OS、driver、CUDA、PyTorch 与依赖版本 | 未验证组合 |
| mesh manifest、训练/验证/测试划分 | 未建立 |
| Hi3D 权重路径及哈希 | 未记录 |
| vae3d 源码/预处理缓存路径 | 远程未部署 |
| 输出根目录、run_id、配额/最长时长 | 未确定 |
| 射线/距离/渲染后端 | 未验证；Linux 不直接照搬 DXR |
| 运行命令与断点恢复命令 | 实现后填写，不能用方案伪代码代替 |

首轮离线生成目标后再训练。记录标签拟合和网络训练两类成本；4096 渲染标签拟合不是一次普通 SDF 查询。原预处理内部有 cuda:0 假设；多卡映射和进程配置须实测。

## 6. 验证证据与后续更新规则

| 类型 | 当前已有结论 | 不代表什么 |
|---|---|---|
| 静态源码分析 | 调用链、数据格式、流式 inference_mode 等已核对，见方案与分析 | 不证明生成质量提升 |
| 上次迁移检查 | 文件内容/Git历史保留、Python AST 检查曾通过 | 不证明依赖可安装或 GPU 可运行 |
| 本轮文档检查 | 提交前检查相对链接、标题/代码块、git diff --check | 不等于实验测试 |
| G0/G0b/G0c/G1–G4 | 均未完成，无实验报告 | README 性能声明不能作为本项目实测 |

每完成一个阶段，记录：日期、实验代码 commit、两套源码版本、resolved_config、数据/权重哈希、环境、执行命令、产物位置、指标、失败清单和下一步。可在 docs/reports/ 放轻量报告并引用外部结果目录；模型权重和大型缓存不直接加入普通 Git。

维护分工：本文件管状态和任务；experiment-plan-v0.3.md 管实验设计；modified-repo-analysis.md 管固定基线的事实。若设计发生实质变化，增加决策记录/修订说明，保留 v0.2 历史归档。
