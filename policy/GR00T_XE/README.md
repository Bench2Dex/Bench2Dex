# GR00T XE (Cross-Embodiment)

GR00T N1.5 的跨灵巧手微调，基于 [NVIDIA Isaac GR00T N1.5](https://github.com/NVIDIA/Isaac-GR00T)。

## 概述

GR00T XE 将 12 款灵巧手 × 26 个任务的遥操作数据统一到一个 64 维 action 空间：

| 部分 | 维度 | 说明 |
|------|------|------|
| 右手 arm ee pose | 6 | 末端位置 + 欧拉角 (机器人基座坐标系) |
| 左手 arm ee pose | 6 | 同上 |
| 右手 hand | 22 | 语义槽 (拇指/食指/中指/无名指/小指/手腕) |
| 左手 hand | 22 | 镜像 |
| padding | 8 | 补到 64 |

## 训练

### 阶段 1: Pretrain (全任务)

```bash
bash policy/GR00T_XE/pretrain.sh
```

### 阶段 2: Per-task Finetune

```bash
bash policy/GR00T_XE/finetune.sh <TASK_ID> --tag <TAG>
```

## 评测

```bash
bash policy/GR00T_XE/eval_double_env.sh <TASK_ID> [none|cov_only|inv_only|inv_cov]
```

## 环境

与 GR00T_n15 共用 `groot` conda 环境。首次使用需运行:

```bash
bash policy/GR00T_n15/setup_env.sh
```

## 文件结构

```
policy/GR00T_XE/
├── pretrain.py / pretrain.sh      # 全任务 pretrain
├── finetune.py / finetune.sh      # 单任务 finetune
├── deploy_policy.py               # 部署: 加载模型 + FK/IK 转换
├── deploy_policy.yml              # 部署配置
├── ik_arm_converter.py            # FK/IK + 手部映射转换器
├── embodiment_mapping.yml         # 12 手 → 44 语义槽映射表
├── xe_config.py                   # 数据配置
├── dataset.py                     # 数据集
├── gr00t_hdf5_dataset.py          # HDF5 数据加载
├── eval_double_env.sh             # 评测脚本
├── src/                           # GR00T 模型扩展
├── LICENSE                        # Apache 2.0
├── requirements.txt               # 依赖
└── setup_env.sh                   # 环境安装
```

## 许可证

Apache 2.0 — 同上游 NVIDIA Isaac GR00T。

## 引用

```bibtex
@inproceedings{gr00tn1_2025,
  title  = {{GR00T} {N1}: An Open Foundation Model for Generalist Humanoid Robots},
  author = {NVIDIA et al.},
  year   = {2025},
  booktitle = {ArXiv Preprint},
}
```