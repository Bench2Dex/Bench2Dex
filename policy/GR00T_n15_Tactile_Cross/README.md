# GR00T N1.5 TacMap Cross-Attention

新训练默认使用 `post_dit + grid=2 + dropout=0.3`，评测默认预测 16 步、执行前 4 步后重新规划。优化分析、论文比较与消融命令见 [触觉方案分析](TACTILE_DESIGN_REVIEW.md)。

独立于 `GR00T_n15_Tactile` 的触觉后训练分支。它加载预训练 GR00T N1.5，冻结 Eagle 视觉塔和语言模型，并联合训练原 Action Head、DiT 与 TacMap residual correction 分支。

## 模型

输入原始深度：`[B, N, 1, H, W]`，每个 site 来自 HDF5：

```text
robot/tactile/distance_along_normal_m/<site>
```

深度为 `float32`、单位米。读取时使用 HDF5 `robot/tactile/meta/max_distance_m`（默认 `0.015`）统一计算：

```python
depth = clip(depth_m, 0, d_max_m) / d_max_m
```

每个 site 经共享 CNN 生成 2×2 的四个 token，加上传感器和空间嵌入。默认在 DiT 之后、action decoder 之前融合：DiT 输出的动作 token 作为 Query，TacMap tokens 作为 Key/Value，Q/K 使用 LayerNorm：

```text
action_features += CrossAttention(LN_Q(action_features), LN_K(tactile_tokens), tactile_tokens)
```

触觉 token 不会拼接到 Eagle VLM token。Cross-Attention 的 `out_proj` 零初始化，因此未训练时与原 GR00T 输出一致。训练时有 0.3 的整模态 tactile dropout；被 drop 的样本同时将 tactile residual 精确门控为零。损失仍只使用原 Flow Matching loss。

冻结：Eagle 视觉塔和语言模型。

训练：VLLN、VL self-attention、state/action encoder、action decoder、future/position embedding、DiT、TacMap CNN、投影、传感器/空间 embedding 和单层 8-head cross-attention。

训练和推理默认使用 `240×240`。Checkpoint 保存 site 顺序、图像尺寸、`resolution_step`、深度单位和 `d_max`；推理端从 checkpoint `config.json` 读取并校验这些参数。

## 训练

保留旧脚本的调用方式，改为 Cross 目录：

```bash
bash policy/GR00T_n15_Tactile_Cross/train.sh 03 --tag tactile_cross --gpu 6
```

训练输出为：

```text
../policy_ckpt/03/<robot_key>/gr00t_n15_trunc_tactile_cross_tactile_cross
```

默认 batch size 为 64，训练 20000 steps，无需额外触觉参数。该目录由 `eval_double_env.sh` 自动发现；也可显式传入模型目录。

新默认仅在从非触觉基座创建触觉分支时生效。加载已有 Cross checkpoint 或 `--resume` 时保留保存的结构；旧 checkpoint 缺少融合字段时仍按 `pre_dit` 加载。使用新结构需要重新训练，不能靠更改默认值转换旧权重。已有同名实验应换 tag，避免混用新旧结果。

## 评估

```bash
bash policy/GR00T_n15_Tactile_Cross/eval_double_env.sh 03 all --sii --headless \
  --model-path ../policy_ckpt/03/multi_xarm7_with_ability/gr00t_n15_trunc_tactile_cross_tactile_cross
```

无需传入融合参数，推理从 checkpoint 读取结构。`deploy_policy.yml` 的 `execution_horizon: 4` 控制执行前缀，`action_horizon: 16` 保持服务器预测长度；可加 `--chunk-size 16` 恢复旧执行长度。执行 4 步会增加策略查询频率，并不保证实时吞吐。

运行时 `TacMapRig.capture()` 已提供 `distance_along_normal_m`，Cross 策略会直接使用它；旧量化字段 `tacmap` 不会进入策略。

旧 VLM-fused tactile checkpoint 与本分支不兼容。必须从 GR00T N1.5 基座重新训练，或加载本分支产生的 Cross checkpoint。
