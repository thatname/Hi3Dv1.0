# 本地浏览远端 G0 mesh、A1/A2 场与四路线拟合

入口：[view_g0_meshes.command](../view_g0_meshes.command)。在 macOS Finder 双击，或在本工程根目录运行：

```bash
./view_g0_meshes.command
```

脚本启动只监听 `127.0.0.1` 的本机 HTTP 服务，自动打开浏览器。默认从 `le.chang@124.156.97.173` 的 `/data/hi3d/g0-prep` 读取 20 个闭合候选、30 个非闭合 mesh 的 `manifest-balanced50.json`、加载审计，以及 `g0-a1a2-r128-cpu-v1/summary.json`。点击样本时才通过 SSH 调用远端只读辅助脚本。页面使用 WebGL 2，不需要本机安装 vae3d、Open3D 或 CUDA。按 `Ctrl+C` 停止服务。

默认 `balanced50` 清单还读取 `g0b-a4-r128-v1/summary.json`。主页面点 **“打开 G0 四路线 2×5 对比”**，或在本地服务地址后加 `/compare?index=7`。该页两行分别为 classic MC 与 FlexiCubes，五列为原 mesh、A1、A2、A3、A4；顶栏的 flat/smooth 选择会加载对应独立拟合缓存，A1/A2 保持共用字段。A1–A4 可按列隐藏，显示着色是单独选项。当前新结果位于 `g0-four-route-r128-v5`，固定 50 件已完成。旧 G0b 同一 A4 场换提取器的历史网格已于 2026-10-31 清理，指标仍见 G0b 报告；`/compare/legacy` 不再作为可用预览入口。其原两种视图模式为：

- **三视窗并排**：原 mesh 灰白、A4 classic MC 蓝色、A4 FlexiCubes 橙色。三个视窗使用源 mesh 的同一居中/缩放和同步相机；在任一视窗旋转、平移或缩放，其他视窗同步。
- **MC / FlexiCubes 叠加**：蓝色 MC 作为底图、橙色 FlexiCubes 以可调透明度覆盖。可切换线框以找轮廓、薄结构和局部缺失；这是显示效果，不是误差热图。

2026-10-30 网页 mesh 着色已改用逐三角面法线（flat shading），避免共享顶点法线平滑掩盖折面。三个视窗的几何坐标和网页相机同步；A4 优化器使用正交视角、逐面法线和轮廓/深度/法线损失，网页仍是透视交互预览，不等同于 A4 的渲染损失图。

左侧按零起点清单 index 或文件名搜索；优先检查 G0b Chamfer 回退较大的 index `7、46、11、40、0`。详情区列出 A0 标志、A4 倒置穿越 cell 数、面数、Chamfer、F-score 和边界边数。Chamfer 是原 mesh 坐标单位，差值为 FlexiCubes 减 MC。A0 标志及倒置 cell 数不等于人工质量结论。对比页按需读取远端 G0b 已缓存的两份 `*_raw_mesh.ply`，不重新生成场或运行网络。

两个对比页默认传输简化目标约 6000 面的显示预览（部分拓扑会保留更多面），以缩短慢速 SSH 连接上的等待时间；可切换到原始提取网格。预览由远端缓存临时简化并压缩传输，不覆盖缓存，也不参与页面列出的原精度评测指标。切换清单 index 后首次生成预览仍需等待十几秒到数十秒；原精度传输可能需要几分钟。单次打开一份服务即可，固定端口可用 `./view_g0_meshes.command --no-browser --port 8765 --four-route-root /data/hi3d/g0-prep/g0-four-route-r128-v5`。

固定 50 件可一次性建立本地压缩预览快照，之后重启浏览器时加 `--preview-cache`，启动和 G0b 快速预览均不需要 SSH；原精度模式、主页面的 A1/A2 场及未缓存输入仍按需通过 SSH 读取。缓存目录默认在 `.local-cache/`，已加入 `.gitignore`，不影响远端结果：

2026-10-31 起，G0b 原 PLY 已清理，旧 `--preview-cache .local-cache/g0b-preview-v1` 启动方式不可用。当前四路线页面用独立的 `.local-cache/four-route-preview-v1` 按需持久缓存简化网格，不要求 G0b 快照。启动命令为 `python3 -m experiments.vae_sdf.view_meshes --dataset balanced50 --port 8765 --no-browser --four-route-root /data/hi3d/g0-prep/g0-four-route-r128-v5`；首次打开新资产较慢，之后本地命中。以下旧 G0b 快照命令仅作历史记录。

```bash
./view_g0_meshes.command --no-browser --port 8765
python3 -m experiments.vae_sdf.cache_g0b_previews --workers 3
# 缓存建完后停止上一份服务，再启动：
./view_g0_meshes.command --no-browser --port 8765 \
  --preview-cache .local-cache/g0b-preview-v1
```

2026-10-30 未缓存预览的本地 HTTP 首字节实测分别为 42.62 秒（index 46，369328 字节）和 15.22 秒（index 47，273100 字节），响应体传输不足 1 毫秒；同一 index 命中进程缓存为 0.0009 秒。远端 index 47 的 cProfile 为 5.19 秒，其中 Python 模块导入累计约 3.87 秒，预览简化函数累计约 2.87 秒（含导入）；SSH 空命令约 3.15 秒。余下等待存在连接/磁盘/并发波动，不能只归因于链路带宽。本地快照将此开销移到一次性准备阶段。50 件压缩快照共约 8.2 MB，当前 8765 服务已切换为 `--preview-cache .local-cache/g0b-preview-v1`；服务首次读取 index 48 的 1,044,316 字节预览总耗时 0.0038 秒、首字节 0.0031 秒，目录接口 0.0018 秒。网页解析/绘制耗时未单独量化。

左侧选 1–50 号资产，顶部可切换 **输入 mesh、A1/A2 未清理零等值面、A1/A2 有符号场切片**。切片支持 X/Y/Z 方向和 0–127 层滑块；红为负、蓝为正、浅色接近零，颜色在 ±4 细体素处饱和。鼠标悬停切片可读原始 `field` 值及 `s=R·field`。三维视图的 A1/A2 零面显示该资产的 Chamfer 和 F-score，便于与输入几何对照。A1/A2 是目标有符号场，**不是严格欧氏 SDF**；切片也不是完整体渲染。

视角操作：左键按住拖动旋转，中键按住拖动平移，滚轮缩放；“重置视角”同时重置旋转、平移和缩放。

可选参数：

```bash
./view_g0_meshes.command --no-browser --port 8000
./view_g0_meshes.command --dataset usable50  # 查看最初补足的 50 个可加载 mesh
./view_g0_meshes.command --dataset first50  # 查看保留失败项的原始首 50 清单
./view_g0_meshes.command --host user@server --remote-dir /path/to/g0-prep \
  --remote-python /path/to/python
./view_g0_meshes.command --batch-root /path/to/g0-a1a2-r128-cpu-v1
./view_g0_meshes.command --g0b-root /path/to/g0b-a4-r128-v1
./view_g0_meshes.command --four-route-root /path/to/g0-four-route-r128-v5
```

远端辅助脚本包括 `experiments/vae_sdf/remote_mesh.py`、`remote_target.py`、`remote_g0b_mesh.py`、`remote_g0b_bundle.py` 与 `remote_four_route.py`。四路线辅助脚本已复制到远端 `/data/hi3d/hi3d-vae-field/experiments/vae_sdf/`，未提交；其余历史辅助脚本按此前部署在 `/data/hi3d/g0-prep/experiments/vae_sdf/`。服务按固定清单索引读取，浏览器不能传入任意远端路径。`--dataset usable50/first50` 仍只显示输入 mesh，因为这两份清单没有配套的固定批量缓存。

此工具参考 vae3d 的 `view_train_t1.bat` 的本地浏览器工作方式，但**不加载 T1 checkpoint、不评估 VAE、不重新生成场**。主页面查看已缓存的 G0 输入、A1/A2 场与零面；新对比页按需查看四路线 A3/A4 网格，历史对比页查看同一 A4 场经 G0b 接口提取的两份网格。A0、A3 场及 A4 场切片尚无完整批量视图。

预览所用几何由 vae3d `load_mesh` 读取并验证，随后恢复到输入坐标系；浏览器只为显示自动居中缩放。它不是完全未经处理的原始文件字节，也不应用于数值评测。默认清单含 20 个 `is_watertight` 闭合候选与 30 个非闭合 mesh；闭合仅为拓扑预筛，尚未证明 A0 符号可靠。`--dataset usable50` 保留原始首 50 个文件中成功的 42 个与补入的 8 个；原始首 50 的 8 个失败仍保留在 `--dataset first50` 和原加载审计中。
