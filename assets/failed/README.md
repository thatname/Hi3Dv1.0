# G3 被剔除资产样例

每个文件对应一类被 G3 数据池剔除的资产，均从远端 `/data/TRELLIS/datasets/ObjaverseXL_github/raw/github/objects/`
（或转换产物 `/data/hi3d/g3/converted/glb-v1/`）原样拷贝，可以直接用 Blender 打开。

| 文件 | 剔除环节 | 原因 | 远端原路径 |
|---|---|---|---|
| `01-convert-failed.dae` | 池级转换 | Blender 5.2.2 转换失败（21,911 请求里 1,209 失败 / 9 超时） | `…/objects/0002da6f3766240a67919c8ccd4816b216ee20b354d350e1d43eecdc2f251540.dae` |
| `02-load-failed-missing-bin.gltf` | 清单加载 | 引用外部 `scene.bin`，但该文件在池里不存在 → trimesh 加载失败（1,561 件的主要成因） | `…/objects/0005eb0f4de373ad6a0cff49ac3beeb00e9ed9c9a833e45702e3c13af2189586.gltf` |
| `03-degenerate-20-faces.obj` | 退化剔除 | 面数 20（阈值 <50）或包围盒零 extent（509 件） | `…/objects/003bb7cbb9f5648c7752e2a882f51acd00b7e7bbac3eb318ffe12c01aa0488a7.obj` |
| `04-capacity-excluded-1.7B-cells.obj` | 容量策略 | 预估 R256 support cell 17.35 亿，远超 4.0M 阈值（512 件，1.62%）；资产本身正常 | `…/objects/0e6688b88bde2658c45df3254793c3e82faa594a5db3cd6ca465b110e5530dcd.obj` |
| `05-a1a2-support-over-maxsites.gltf` | A1/A2 | `Support has 6,218,242 points, exceeds max-sites 6,000,000`（44 件） | `…/objects/014759e992279a9a289ccd3a22309da7ff7ed84dacefe231bd3b937c3c73c531.gltf` |
| `06-a1a2-no-crossing-cells.glb` | A1/A2 | `Initial field has no crossing cells`——退化网格，场里没有零交叉面（14 件） | `/data/hi3d/g3/converted/glb-v1/14e0fa509d10d70aed576c9030cc8a30f99e9010d5fee092614cf06679acf253.glb` |
| `07-a4-cuda700.glb` | A4 | `RuntimeError: Cuda error: 700` in nvdiffrast `rasterize_fwd_cuda`；2048 分辨率重试仍失败 | `…/objects/086107ffa8ac79ff0390ba968e4196bbbf57d56d4881c429923989ecaf9b8bc7.glb` |
| `08-a4-normals-not-converged.stl` | A4 | `RuntimeError: GPU corner fan labels did not converge`（`four_route_normals.py:99`） | `…/objects/09bc88b9b06334c8569bfee8cb5c03d5d247e1765df7dbc2332f1aabd9997ebb.stl` |
| `09-a4-timeout-12.7M-faces.glb` | A4 | 1,268 万面，单独重试 900 s 超时（原始 7,889 万面的 3721 已单独留在 `assets/g3-index-3721/`） | `/data/hi3d/g3/converted/glb-v1/0f7169057a02183e4fe190d4c9dad5fce65ad3dc6e4a59468c31a9029c4b540c.glb` |

## 完整漏斗（截至 2026-11-03 14:50，脚本自动生成）

数据在远端 `reports/asset-funnel.json`，由
`python -m experiments.vae_sdf.summarize_g3_funnel --experiment /data/hi3d/g3/exp-35k-r256-e4-v1` 生成；
跑完后重跑一次即可冻结最终数字。

- 源池 34,847 文件 →（−1,226 格式不可用：转换失败 1,209、超时 9、abc 6、无扩展名 2）→ **33,621 候选**
- 33,621 →（−1,561 加载失败、−509 退化）→ **31,551 可用**
- 31,551 →（−512 容量策略）→ **31,039 训练池**（24,818 / 3,106 / 3,115）
- 运行期缺口：A1/A2 58（44 support 超 max-sites + 14 无零交叉）、A4 账本 22 → 合计 **80**
- 预期最终可用：**30,959**（另有 1 件 deferred 待重跑，未计入缺口）

## 注意

- `.gltf` 样例缺 `scene.bin` 是剔除原因本身，Blender 打开会同样报缺文件——这是预期现象。
- `04-` 那个 69 MB 的 obj 是"好资产但太大"，用它对比可以直观看到容量策略拦下了什么样的模型。
- 这些文件只用于人工检查，未纳入任何提交（`.gitignore` 已忽略 `*.glb`，其余小文件如不想入库请自行处理）。