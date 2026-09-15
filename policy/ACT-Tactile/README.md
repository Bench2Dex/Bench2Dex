# Tactile Policy

这个目录提供一套独立的 Tactile-Flat + ACT 训练、离线验证和 Isaac Sim
在线评测流程。它读取
Dex2Bench HDF5 中的四路 RGB、机器人关节状态、未来动作块，以及
`robot/tactile/tacmap` 下由 `site_names` 定义的单通道 TacMap。

现有 `policy/ACT` 和 `run_policy.py` 没有被修改。在线评测使用策略目录内独立的
`policy/ACT-Tactile/tactile_run_policy.py`，本地加载 tactile checkpoint，并复用原评测的任务重置、
action chunk、成功率、metric 和结果输出逻辑。

## 模型输入

- RGB：默认四路 ACT 相机，形状为 `[B, 4, 3, 480, 640]`；
- qpos：默认启用当前 robot key 对应的 active DOF；
- TacMap：按 HDF5 的 `robot/tactile/meta/site_names` 排序，形状为
  `[B, N, 1, 240, 240]`。站点数 `N` 来自 HDF5 元数据，例如 Allegro/LEAP
  当前为 8，其他已配置手型可为 10；
- action：未来 30 步关节目标，尾部和 `action_valid=False` 的位置使用
  `is_pad` 屏蔽。

RGB 使用共享 ResNet-18 生成空间 token。全部 TacMap 合并到 batch 维后通过
共享的单通道 ResNet-18，每个触觉站点经全局池化得到一个特征并加入 site
identity embedding。一个可学习 query 通过多头 attention pooling 聚合所有
站点，最终无论固定配置中有多少个站点，都只生成一个触觉 token；该 token
再加入 tactile modality embedding。随后，视觉、单个触觉 token、qpos 和
cVAE latent token 一起进入 ACT Transformer，预测 action chunk。

单 token attention pooling 从 checkpoint format version 4 开始使用。version 3
及更早的触觉 checkpoint 与当前结构不兼容，需要重新训练。

## 环境

依赖项目现有的 PyTorch、torchvision、h5py、NumPy 和 OpenCV 环境。本机可用
的 IsaacLab Python 示例为：

```powershell
$python = ".\.venv\Scripts\python.exe"
```

所有命令均从 `dex2scene` 根目录执行。

## 最小 smoke test

下面的命令只检查数据、前向、反向和 checkpoint 是否能跑通。它使用小模型和
缩小后的图像，不代表正式训练配置：

```powershell
& $python -m policy.tactile_policy.train `
  --dataset-dir "..\tactile_leverage" `
  --ckpt-dir ".\outputs\logs\tactile_policy\smoke" `
  --epochs 1 `
  --batch-size 1 `
  --num-workers 0 `
  --val-ratio 0 `
  --max-steps-per-epoch 1 `
  --backbone tiny `
  --hidden-dim 128 `
  --dim-feedforward 256 `
  --nheads 4 `
  --enc-layers 2 `
  --dec-layers 2 `
  --latent-dim 16 `
  --image-height 120 `
  --image-width 160 `
  --tactile-height 60 `
  --tactile-width 60 `
  --device cpu
```

## 正式配置示例

```powershell
& $python -m policy.tactile_policy.train `
  --dataset-dir "..\tactile_leverage" `
  --ckpt-dir ".\outputs\logs\tactile_policy\vision_tactile" `
  --epochs 100 `
  --batch-size 2 `
  --num-workers 4 `
  --chunk-size 30 `
  --backbone resnet18 `
  --robot-key multi_panda_with_allegro `
  --device cuda
```

`--robot-key` 是可选参数。省略时训练程序从 HDF5 自动检测 robot key；提供时会
严格校验所有 episode 的元数据，发生不一致即报错，不会覆盖 HDF5 中的值。同一
训练目录仍只能包含一种手型。

该策略只接受视觉与 TacMap 联合输入，不提供 `vision_only` 训练开关。纯视觉基线请直接使用未修改的 `policy/ACT`。

Linux 也可以使用包装脚本：

```bash
bash policy/tactile_policy/train.sh \
  ../tactile_leverage \
  outputs/logs/tactile_policy/vision_tactile \
  --epochs 100 --batch-size 2
```

## 离线触觉消融

```powershell
& $python -m policy.tactile_policy.offline_eval `
  --checkpoint ".\outputs\logs\tactile_policy\vision_tactile\policy_best.ckpt" `
  --dataset-dir "..\tactile_leverage" `
  --batch-size 2 `
  --num-workers 0 `
  --output ".\outputs\logs\tactile_policy\vision_tactile\ablation.json"
```

报告包含：

- `normal`：正常 TacMap；
- `zero`：全部 TacMap 置零；
- `shuffle`：将 TacMap 在同一 episode 内做确定性时间打乱。

比较三组 L1 loss 可以检查模型是否依赖触觉。它不能替代任务成功率实验。

## Isaac Sim 在线评测

在线输入为 **4 RGB + checkpoint-defined TacMaps**：左右腕相机、左右双目相机，
以及 checkpoint `site_names` 中保存的动态数量触觉站点。相机、触觉站点、robot key、
active indices 和 state dimension 会在 rollout 前校验。模型使用 prior-only
推理，输出的归一化 action chunk 会恢复到实际关节值，并以 20 Hz 逐项执行。

TacMap ray casting 要求 Isaac Sim 使用 CUDA 设备，不能使用 CPU simulation。
当前脚本只支持本地 checkpoint，不经过 REMOTE server/client。

直接命令示例：

```bash
python policy/ACT-Tactile/tactile_run_policy.py \
  --policy-type TACTILE \
  --task scenes/06_fruit_bowl_loading.yaml \
  --ckpt-dir outputs/logs/tactile_policy/vision_tactile \
  --ckpt-name policy_best.ckpt \
  --robot-key multi_panda_with_allegro \
  --enable-rgb --active-dof --device cuda:0 \
  --temporal-agg --temporal-agg-k 0.1 \
  --num-episodes 10 --episode-steps 400 \
  --enable-generalization --generalization-profile none \
  --anchor-hdf5 /path/to/episode_000000.hdf5 \
  --headless
```

上面的直接命令显式启用 temporal aggregation。启用后，策略会在每个 20 Hz 控制步
重新推理，并对覆盖当前时刻的重叠 action chunk 做指数加权融合，而不是一次性执行完整
chunk。`--temporal-agg-k` 控制融合权重的衰减程度；设为 `0` 时使用均匀平均。

也可以使用包装脚本：

在线 wrapper 使用 `mapfile` 和数组参数转发，要求 **Bash 4+**；目标运行环境为
Isaac/Linux。macOS 自带的 Bash 3 不支持该 wrapper，请在 Isaac/Linux 环境运行。

```bash
TACTILE_TASK=06_fruit_bowl_loading \
TACTILE_CKPT_DIR=outputs/logs/tactile_policy/vision_tactile \
TACTILE_ROBOT_KEY=multi_panda_with_allegro \
TACTILE_ANCHOR_HDF5=/path/to/episode_000000.hdf5 \
bash policy/ACT-Tactile/eval_direct.sh
```

`eval_direct.sh` 默认启用 temporal aggregation，默认衰减系数为 `0.1`。可以通过
`TACTILE_TEMPORAL_AGG_K` 调整，例如
`TACTILE_TEMPORAL_AGG_K=0.05 bash policy/ACT-Tactile/eval_direct.sh`。

可选环境变量包括 `TACTILE_CKPT_NAME`、`TACTILE_GPU_ID`、
`TACTILE_NUM_EPISODES`、`TACTILE_EPISODE_STEPS`、`TACTILE_SEED`、
`TACTILE_GEN_PROFILE`、`TACTILE_OUTPUT_DIR`、`TACTILE_ROBOT_KEY` 和
`ISAAC_PYTHON`。wrapper 仅在 `TACTILE_ROBOT_KEY` 非空时转发 `--robot-key`，
用于显式校验手型；省略时由 checkpoint 自动选择 robot key。

## 输出文件

- `resolved_config.json`：本次模型、数据和维度配置；
- `dataset_stats.pkl`：仅由训练帧计算的 qpos/action 归一化统计；
- `policy_last.ckpt`：最后一轮 checkpoint；
- `policy_best.ckpt`：按 validation loss 选择的 checkpoint。

checkpoint 同时保存 camera/site 顺序、robot key、active joint indices、按语义排序的
active joint names、TacMap resolution/max-distance 编码参数、模型配置、归一化统计和
optimizer 状态。在线评测拒绝缺少这些运行时元数据或旧格式的 checkpoint，
以免策略静默忽略触觉或错配关节顺序；这类旧 checkpoint 需要重新训练/保存。
checkpoint 是 embodiment-specific 的，只能用于训练时对应的手型，不能跨手型复用。

## 当前数据的限制

`../tactile_leverage/episode_000000.hdf5` 只有一条轨迹，并且标记为
`success=False`、`demo_eligible=False`。代码会允许它进行 smoke test，但会打印
明确警告。它可以验证 TacMap 的读取、融合和梯度链路，不能单独训练出可靠策略，
也不能据此证明触觉提高了任务成功率。
