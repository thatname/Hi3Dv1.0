# 本地浏览远端 G0 mesh

入口：[view_g0_meshes.command](../view_g0_meshes.command)。在 macOS Finder 双击，或在本工程根目录运行：

```bash
./view_g0_meshes.command
```

脚本启动只监听 `127.0.0.1` 的本机 HTTP 服务，自动打开浏览器。默认从 `le.chang@124.156.97.173` 的 `/data/hi3d/g0-prep` 读取 20 个闭合候选、30 个非闭合 mesh 的 `manifest-balanced50.json` 与加载审计；点击样本时才通过 SSH 调用远端只读辅助脚本，回传三角网格。页面本身使用 WebGL 2，不需要本机安装 vae3d、Open3D 或 CUDA。按 `Ctrl+C` 停止服务。

视角操作：左键按住拖动旋转，中键按住拖动平移，滚轮缩放；“重置视角”同时重置旋转、平移和缩放。

可选参数：

```bash
./view_g0_meshes.command --no-browser --port 8000
./view_g0_meshes.command --dataset usable50  # 查看最初补足的 50 个可加载 mesh
./view_g0_meshes.command --dataset first50  # 查看保留失败项的原始首 50 清单
./view_g0_meshes.command --host user@server --remote-dir /path/to/g0-prep \
  --remote-python /path/to/python
```

远端辅助脚本是 `experiments/vae_sdf/remote_mesh.py`，当前副本位于 `/data/hi3d/g0-prep/experiments/vae_sdf/remote_mesh.py`；不会修改远端 hi3d-vae-field 仓库。它按清单索引加载资产，并要求加载审计中的索引、ID 和路径吻合。服务不接受任意远端路径请求。

此工具参考 vae3d 的 `view_train_t1.bat` 的本地浏览器工作方式，但**不加载 T1 checkpoint、不评估 VAE、不生成 SDF**。当前仅查看 G0 的输入 mesh；将来 A0–A4 raw mesh 生成后，可在独立迭代中增加同屏对照。

预览所用几何由 vae3d `load_mesh` 读取并验证，随后恢复到输入坐标系；浏览器只为显示自动居中缩放。它不是完全未经处理的原始文件字节，也不应用于数值评测。默认清单含 20 个 `is_watertight` 闭合候选与 30 个非闭合 mesh；闭合仅为拓扑预筛，尚未证明 A0 符号可靠。`--dataset usable50` 保留原始首 50 个文件中成功的 42 个与补入的 8 个；原始首 50 的 8 个失败仍保留在 `--dataset first50` 和原加载审计中。
