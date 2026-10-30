# G1 本地训练监控页

本地入口：`http://127.0.0.1:8767/`。这是独立的只读服务；原 G0/G0b mesh 浏览器继续使用 `http://127.0.0.1:8765/`。两个服务的路由、进程和端口互不覆盖。

在本地工程根目录启动：

```bash
python3 -m experiments.vae_sdf.view_g1 --port 8767 --no-browser
```

默认监控远端 `/data/hi3d/g0-prep/g1-a4-fifty-v2/`。换一轮训练时指定新的远端运行目录：

```bash
python3 -m experiments.vae_sdf.view_g1 --port 8767 --no-browser \
  --run /data/hi3d/g0-prep/新运行目录
```

2026-10-30 当前本地 8767 进程指向 `--run /data/hi3d/g0-prep/g1-a4-eight-train-step400-v1`；状态 API 返回 `completed`、400 条 trace 和可用 checkpoint。原始默认运行仍可通过不带 `--run` 的命令查看。

下一轮先确定唯一的远端输出目录，**启动并打开 8767 页面，再启动训练命令**；页面在 `resolved_config.json` 尚未出现时显示“等待训练目录”，目录出现后自动切换为训练状态。这样从第一步起就能看曲线。开训前核对页面顶部路径与训练命令的 `--output` 完全一致，避免看错运行。

页面每 5 秒经 SSH 读取 `resolved_config.json` 和逐步 `trace.json`，显示训练进度、总损失、场/位移/支撑/有效性损失、KL 诊断、梯度、逐资产首末变化及 checkpoint 是否存在。浏览器和本地服务不加载权重、不启动训练、不改远端缓存。当前默认运行已完成，因此页面显示历史记录；下一轮运行写入同样的日志格式时，可在训练进行中持续刷新。网络暂时断开时保留最近一次画面并重试。

表格中的 **A4** 链接打开原有的 G0b 网格对照页，显示原 mesh、A4 classic MC 与 A4 FlexiCubes。它们是训练目标的提取结果，**不是 G1 网络重建**。G1 评测已另存 raw PLY 与 `report.json`，见 [8+2 报告](reports/g1-eight-train-two-val-2026-10-30.md)；当前训练监测页不展示这些网格。训练曲线记录每次参数更新前的前向损失，单次运行中不同样本的数值不应直接当作验证集指标。

2026-10-30 实测：本地 `/api/state` 返回 200，含当前 8 资产、400 条 trace 与 checkpoint 状态；独立历史运行的指标见对应阶段报告。服务是否仍在运行以当前端口检查为准，不依赖历史 PID。
