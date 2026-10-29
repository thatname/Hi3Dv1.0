# 本地浏览远端 G0 mesh 与 A1/A2 场

入口：[view_g0_meshes.command](../view_g0_meshes.command)。在 macOS Finder 双击，或在本工程根目录运行：

```bash
./view_g0_meshes.command
```

脚本启动只监听 `127.0.0.1` 的本机 HTTP 服务，自动打开浏览器。默认从 `le.chang@124.156.97.173` 的 `/data/hi3d/g0-prep` 读取 20 个闭合候选、30 个非闭合 mesh 的 `manifest-balanced50.json`、加载审计，以及 `g0-a1a2-r128-cpu-v1/summary.json`。点击样本时才通过 SSH 调用远端只读辅助脚本。页面使用 WebGL 2，不需要本机安装 vae3d、Open3D 或 CUDA。按 `Ctrl+C` 停止服务。

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
```

远端辅助脚本是 `experiments/vae_sdf/remote_mesh.py` 和 `remote_target.py`，副本位于 `/data/hi3d/g0-prep/experiments/vae_sdf/`；不会修改远端 hi3d-vae-field 仓库。前者按清单索引加载输入 mesh，后者依据固定批量结果读取目标缓存并核对 manifest、源哈希、阶段、分辨率。服务不接受网页传入任意远端路径。`--dataset usable50/first50` 仍只显示输入 mesh，因为这两份清单没有配套的 50 资产 A1/A2 批量缓存。

此工具参考 vae3d 的 `view_train_t1.bat` 的本地浏览器工作方式，但**不加载 T1 checkpoint、不评估 VAE、不重新生成场**。当前查看的是已缓存的 G0 输入、A1/A2 场与零面；A0、A3、A4 尚无完整批量视图。

预览所用几何由 vae3d `load_mesh` 读取并验证，随后恢复到输入坐标系；浏览器只为显示自动居中缩放。它不是完全未经处理的原始文件字节，也不应用于数值评测。默认清单含 20 个 `is_watertight` 闭合候选与 30 个非闭合 mesh；闭合仅为拓扑预筛，尚未证明 A0 符号可靠。`--dataset usable50` 保留原始首 50 个文件中成功的 42 个与补入的 8 个；原始首 50 的 8 个失败仍保留在 `--dataset first50` 和原加载审计中。
