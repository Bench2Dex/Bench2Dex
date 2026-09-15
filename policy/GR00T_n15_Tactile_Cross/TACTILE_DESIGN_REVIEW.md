# 触觉方案分析与优化实验

日期：2026-09-08。分析对象：本目录 GR00T N1.5 Cross 分支。

## 判断

现有方案适合作为低侵入的触觉融合基线：共享 CNN、site/spatial embedding、动作查询触觉、零初始化残差，以及整模态 dropout 都有明确作用。但它尚不具备执行动作段期间的快速触觉纠错能力。仅加深 cross-attention 不能解决反馈迟滞；更复杂的网络是否提高成功率，必须通过闭环评测确认。

当前代码可确认的限制：

| 项目 | 优化前实现与影响 | 优先级 |
| --- | --- | --- |
| 反馈间隔 | `run_policy.py` 按 20 Hz 消耗动作队列，默认 16 步后才重新生成，约 0.8 s 仿真时间；采集或发送新观测不会修改已排队动作 | 最高 |
| 融合上下文 | `pre_dit` 的 Q 来自带噪动作和时间编码，尚未经 DiT 融合视觉、语言、状态；动作特征层面的残差不等于物理动作纠偏 | 高 |
| 时间信息 | 数据集只读当前触觉帧，推理每次生成动作段也只编码一次；无法直接从多帧估计接触变化趋势 | 高 |
| 空间压缩 | 每 site 的 240×240 深度最终平均池化为 2×2，可能弱化稀疏局部接触；需要 2×2/4×4 消融来验证 | 中 |
| 信号语义 | TacMap 导出 CPD 过滤后的法向几何距离和 contact mask；模型只用距离，不能据此宣称测量了法向力、剪切力或摩擦 | 高 |
| 梯度来源 | 只有 action flow-matching loss；视觉/状态分支可能已足够拟合演示，需要触觉干预测试确认网络真的依赖触觉 | 高 |
| Dropout | 固定 0.3 的整模态 dropout 提供缺失模态鲁棒性，但会减少触觉分支的有效训练样本，最优值未知 | 中 |

零初始化输出投影能让新残差在初始化时为零；首个 backward 中上游触觉 CNN 的梯度可为零，这是预期行为。仅检查 `grad is not None` 无法证明 CNN 已在学习。输出投影更新后应检查 CNN 的非零梯度。本次测试覆盖了这一点。

零输入也不能当作严格的“无触觉模型”：CNN bias、site/spatial embedding 仍会产生 token。严格消融需要关掉 residual 或训练匹配的无触觉基线；shuffle/delay tactile 则测试策略是否使用正确且及时的接触信息。

## 与相关工作比较

检索尚未找到能够核实方法细节的 TouchRefine 原始论文。以下不将相近名称的工作视为 TouchRefine 的同义名称，也不推测它的网络结构；拿到准确论文后再补充。

| 工作 | 原文可确认的方法 | 对本项目的启发与边界 |
| --- | --- | --- |
| [VLA-Touch](https://arxiv.org/html/2507.17294v1) | 保持基础 VLA 不微调，使用 interpolant controller 将基础动作细化为专家动作；执行短段后更新观测；还包含触觉语言反馈用于规划 | 可以借鉴短段闭环和基础动作条件化。当前特征残差不是其 interpolant controller；TacMap 深度也不是其 GelSight marker-derived force |
| [Reactive Diffusion Policy](https://www.roboticsproceedings.org/rss21/p052.html) | 慢速 latent diffusion 生成动作计划，快速 asymmetric tokenizer 使用触觉/力反馈形成闭环 | 长期目标应是低频视觉规划与高频触觉执行解耦；把整个 VLA 重跑得更频繁只是先做的基线 |
| [VT-Refine](https://proceedings.mlr.press/v305/huang25b.html) | 视觉和触觉演示初始化 diffusion policy，再在带触觉仿真的数字孪生中用 RL 微调；使用法向力触觉传感器 | 提醒我们补充接触偏差和恢复行为的数据。其改进包含 RL 和数据分布改变，不能归因于一个融合层 |
| [ReTouch](https://arxiv.org/html/2608.01824v2) | 结构化触觉 patch 表征、预测未来触觉，并用执行期反馈在线修正触觉预测与动作 | 保留手指身份、接触空间结构和时间变化值得研究；实现同类预测需要额外训练目标和严格的因果数据处理 |

本项目跨多种手型，传感器 site 数量、布局、任务都与上述实验不同。论文中成功率不能作为这里预期增益的数值估计。

## 已实现的改动

1. `tactile_fusion_stage=post_dit`：Q 取 DiT 输出的动作 token，K/V 取触觉 token；对 Q/K 做 LayerNorm，V 保留原始编码幅度，使用零初始化输出投影和相同的 modality dropout gate。只细化动作 token，然后进入 action decoder。训练和每个去噪步使用相同路径。它仍属于条件 flow matching，不是高频独立 residual controller。
2. `tactile_grid_size`：从固定 2×2 改为可配置，2 保持旧 checkpoint 形状，4 对应每 site 16 个 token。更细网格会增加 attention 开销，不默认认为越大越好。
3. `tactile_dropout_prob`：允许训练参数覆盖，并保存到 checkpoint。
4. 修复 remote deployment 中 execution prefix 未落实的问题：`--chunk-size 4` 会只返回服务器动作段的前四步，队列消耗完后读取当前观测重新生成。模型训练和服务器预测仍可保持 16 步。
5. Cross 训练的 tmux 启动显式传入 GPU、Python PATH、PYTHONPATH 和相关缓存环境，避免已有 tmux server 使用过期环境。

新训练默认结构为 `post_dit + grid=2 + dropout=0.3`，评测默认执行前 4 步后重新规划，服务器预测仍为 16 步。原有命令无需增加触觉参数或 `--chunk-size 4`。旧 checkpoint 可按原结构加载，缺少融合字段时使用 `pre_dit`。切换融合位置或网格改变参数结构，需从非触觉基座训练一个新实验；训练脚本拒绝悄悄转换已有 tactile checkpoint。`--resume` 从输出目录最新的 checkpoint 加载结构，再恢复训练状态。

## 建议的实验顺序

先选 03、08 等接触变化明显的任务和一个装箱对照任务，固定数据划分、相同训练步数、batch、学习率以及评测 seed，至少使用多个训练 seed。优先做单因素消融：

| 组 | 融合 | grid | dropout | 执行前缀 | 目的 |
| --- | --- | --- | --- | --- | --- |
| A | pre_dit | 2 | 0.3 | 16 | 原始基线 |
| B | pre_dit | 2 | 0.3 | 4 | 复用 A 的 checkpoint，隔离反馈频率影响 |
| C | post_dit | 2 | 0.3 | 4 | 隔离上下文融合位置影响 |
| D | post_dit | 4 | 0.3 | 4 | 检查空间细节 |
| E | post_dit | 4 | 0.1 | 4 | 检查 dropout 强度 |

不要只比较 A 与 E 后就把所有收益归因于 cross-attention。无触觉基线也要匹配执行前缀长度，区分更多视觉重规划和触觉本身的收益。

在仓库根目录训练 C：

```bash
bash policy/GR00T_n15_Tactile_Cross/train.sh 03 \
  --gpu 3 --steps 20000 --batch 64 \
  --tactile-fusion-stage post_dit \
  --tactile-grid-size 2 --tactile-dropout-prob 0.3 \
  --tag post_dit_grid2_drop03_bs64_step20000
```

评测 C（显式绑定此次 checkpoint 和独立输出目录）：

```bash
bash policy/GR00T_n15_Tactile_Cross/eval_double_env.sh 03 all --sii --headless \
  --model-path ../policy_ckpt/03/multi_xarm7_with_ability/gr00t_n15_trunc_tactile_cross_post_dit_grid2_drop03_bs64_step20000 \
  --chunk-size 4 --output-dir ../output_tactile_ablation/03_post_dit_grid2_exec4
```

也可以在已有 A checkpoint 上先只测试 `--chunk-size 4`，不需要重新训练。20 Hz 动作下，4 步相当于 0.2 s 仿真时间；这不是已验证的 5 Hz 实时推理吞吐。相对执行 16 步，完整策略查询频率约增至四倍。必须记录端到端延迟、RPC 开销及真实 wall-clock 时间。

评价至少包含 stable SR、接触后成功率/失败原因、任务时间、滑落与安全指标、推理延迟，以及触觉 residual 禁用、错配和延迟干预。四个泛化通道应使用同一任务的同一个已选 checkpoint 和一致的 seed 方案；在验证集上选 checkpoint，测试集不逐通道挑分数。报告成功次数和分母以及跨 seed 波动。

## 后续优先级

短段重规划有效之后，再加入因果触觉历史，例如 `[t-3,t-2,t-1,t]`，共享 CNN 后做小型 temporal encoder。训练和推理必须使用相同采样间隔、episode 起始填充、reset 规则；未来触觉只可作为训练标签，不能放入当前输入。几何距离的时差能描述接触变化，但不能自动等价于滑移速度或剪切力。

如果触觉分支仍被忽略，可先增加接触阶段覆盖、轻微错位与恢复行为演示，再研究 contact-state/变化预测的辅助损失。需要从真实记录的 contact mask 或定义清晰的几何信号构造目标，不能用未经标定的深度替代力标签。

如果全模型每四步重算的延迟不允许部署，再实现 RDP/VLA-Touch 类慢快结构：低频缓存视觉上下文和基础动作，高频读取当前 tactile/state 修正短期动作。该方案需要训练与执行协议一起设计，不能只靠在现有 DiT 中多插几层 attention 得到。

本次完成代码级兼容性、梯度和执行前缀检查；尚未训练新模型或运行 Isaac Sim 闭环实验，因此不声称成功率已提升。

已验证：54 项单元测试通过，覆盖新训练默认值、旧 checkpoint 不被默认值覆盖、默认执行前缀及 CLI 覆盖；真实 diffusers DiT 的 `pre_dit/grid2`、`post_dit/grid2`、`post_dit/grid4` 三组 CPU 小模型均通过两次优化步、推理和配置/权重重载检查；训练命令 dry-run、Python CLI 参数解析、shell 语法和 diff 空白检查通过。真实 DiT 检查不加载完整 GR00T/VLM；完整训练恢复和实际机器人运行仍需实验验证。

```bash
../miniconda3/envs/groot/bin/python -m unittest discover -s policy/GR00T_n15_Tactile_Cross/tests
../miniconda3/envs/groot/bin/python policy/GR00T_n15_Tactile_Cross/tests/smoke_real_dit.py
```
