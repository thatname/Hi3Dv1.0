# 本地浏览远端 G0 mesh、A1/A2 场与 G0b 提取对比

入口：[view_g0_meshes.command](../view_g0_meshes.command)。在 macOS Finder 双击，或在本工程根目录运行：

```bash
./view_g0_meshes.command
```

脚本启动只监听 `127.0.0.1` 的本机 HTTP 服务，自动打开浏览器。默认从 `le.chang@124.156.97.173` 的 `/data/hi3d/g0-prep` 读取 20 个闭合候选、30 个非闭合 mesh 的 `manifest-balanced50.json`、加载审计，以及 `g0-a1a2-r128-cpu-v1/summary.json`。点击样本时才通过 SSH 调用远端只读辅助脚本。页面使用 WebGL 2，不需要本机安装 vae3d、Open3D 或 CUDA。按 `Ctrl+C` 停止服务。

默认 `balanced50` 清单还读取 `g0b-a4-r128-v1/summary.json`。主页面点 **“打开 G0b MC / FlexiCubes 对比”**，或在本地服务地址后加 `/compare?index=7`，可直接查看指定的零起点清单 index。对比页有两种模式：

- **三视窗并排**：原 mesh 灰白、A4 classic MC 蓝色、A4 FlexiCubes 橙色。三个视窗使用源 mesh 的同一居中/缩放和同步相机；在任一视窗旋转、平移或缩放，其他视窗同步。
- **MC / FlexiCubes 叠加**：蓝色 MC 作为底图、橙色 FlexiCubes 以可调透明度覆盖。可切换线框以找轮廓、薄结构和局部缺失；这是显示效果，不是误差热图。

左侧按零起点清单 index 或文件名搜索；优先检查 G0b Chamfer 回退较大的 index `7、46、11、40、0`。详情区列出 A0 标志、A4 倒置穿越 cell 数、面数、Chamfer、F-score 和边界边数。Chamfer 是原 mesh 坐标单位，差值为 FlexiCubes 减 MC。A0 标志及倒置 cell 数不等于人工质量结论。对比页按需读取远端 G0b 已缓存的两份 `*_raw_mesh.ply`，不重新生成场或运行网络。

G0b 对比页默认传输简化目标约 6000 面的显示预览（部分拓扑会保留更多面），以缩短慢速 SSH 连接上的等待时间；顶部“显示精度”可切换到原始提取网格。预览由远端缓存临时简化并压缩传输，不覆盖缓存，也不参与页面列出的原精度评测指标。切换清单 index 后首次生成预览仍需等待十几秒到数十秒；原精度传输可能需要几分钟。单次打开一份服务即可，固定端口可用 `./view_g0_meshes.command --no-browser --port 8765`，页面入口为 `http://127.0.0.1:8765/compare?index=7`。

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
```

远端辅助脚本是 `experiments/vae_sdf/remote_mesh.py`、`remote_target.py`、`remote_g0b_mesh.py` 和 `remote_g0b_bundle.py`，副本位于 `/data/hi3d/g0-prep/experiments/vae_sdf/`；不会修改远端 hi3d-vae-field 仓库。它们按固定清单索引读取输入 mesh、A1/A2 缓存或 G0b 原始提取网格，并核对批量目录及资产身份。服务不接受网页传入任意远端路径。`--dataset usable50/first50` 仍只显示输入 mesh，因为这两份清单没有配套的 A1/A2 或 G0b 固定批量缓存。

此工具参考 vae3d 的 `view_train_t1.bat` 的本地浏览器工作方式，但**不加载 T1 checkpoint、不评估 VAE、不重新生成场**。主页面查看已缓存的 G0 输入、A1/A2 场与零面；对比页查看 A4 场经 G0b 接口提取的两份网格。A0、A3 场及 A4 场切片尚无完整批量视图。

预览所用几何由 vae3d `load_mesh` 读取并验证，随后恢复到输入坐标系；浏览器只为显示自动居中缩放。它不是完全未经处理的原始文件字节，也不应用于数值评测。默认清单含 20 个 `is_watertight` 闭合候选与 30 个非闭合 mesh；闭合仅为拓扑预筛，尚未证明 A0 符号可靠。`--dataset usable50` 保留原始首 50 个文件中成功的 42 个与补入的 8 个；原始首 50 的 8 个失败仍保留在 `--dataset first50` 和原加载审计中。
