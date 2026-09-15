# GR00T N1.5 Tactile Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a standalone GR00T N1.5 tactile policy fork that consumes synchronized TacMap images during training and inference without modifying `policy/GR00T_n15`.

**Architecture:** Fork the existing GR00T N1.5 policy into `policy/GR00T_n15_Tactile`. Read per-site TacMap frames as `[N,1,H,W]`, encode all sites with a shared CNN, add site identity embeddings, pool them with a learned attention query, and append one projected tactile token to the frozen Eagle vision-language features before the flow-matching action head. Store the tactile site schema in the model/checkpoint configuration and enforce the same schema during inference.

**Tech Stack:** Python, PyTorch, torchvision, h5py, NumPy, Isaac-GR00T N1.5

---

### Task 1: Fork the GR00T baseline

**Files:**
- Copy: `policy/GR00T_n15/**`
- Create: `policy/GR00T_n15_Tactile/**`

1. Copy the baseline implementation without changing `policy/GR00T_n15`.
2. Update package paths and default data-config references to `policy.GR00T_n15_Tactile`.
3. Verify the copied Python sources compile.

### Task 2: Define and test the tactile HDF5 contract

**Files:**
- Create: `policy/GR00T_n15_Tactile/tactile_io.py`
- Create: `policy/GR00T_n15_Tactile/tests/test_tactile_io.py`
- Modify: `policy/GR00T_n15_Tactile/gr00t_hdf5_dataset.py`

1. Write failing tests for site-order loading, uint8 validation, resizing, normalization, and cross-episode schema mismatch.
2. Implement pure NumPy/h5py tactile readers.
3. Add current-frame tactile tensors to HDF5-native samples.

### Task 3: Add the tactile token encoder

**Files:**
- Create: `policy/GR00T_n15_Tactile/src/gr00t/model/action_head/tactile_token_encoder.py`
- Create: `policy/GR00T_n15_Tactile/tests/test_tactile_token_encoder.py`
- Modify: `policy/GR00T_n15_Tactile/src/gr00t/model/action_head/flow_matching_action_head.py`

1. Test expected input validation and `[B,1,D]` output shape.
2. Implement a shared single-channel CNN, site embeddings, learned-query attention pooling, projection, and modality embedding.
3. Append the tactile token and attention-mask entry before vision-language self-attention in both training and inference.

### Task 4: Carry tactile through GR00T transforms and model construction

**Files:**
- Modify: `policy/GR00T_n15_Tactile/gr00t_dex2bench_config.py`
- Modify: `policy/GR00T_n15_Tactile/scripts/gr00t_finetune.py`
- Modify: `policy/GR00T_n15_Tactile/train.sh`

1. Preserve the tactile tensor through `GR00TTransform`.
2. Discover the dataset tactile schema before model construction.
3. Recreate the action head with tactile configuration and randomly initialized tactile weights while preserving compatible pretrained weights.
4. Keep tactile modules trainable with the action projector and diffusion head.

### Task 5: Add tactile-aware inference

**Files:**
- Modify: `policy/GR00T_n15_Tactile/deploy_policy.py`
- Create: `policy/GR00T_n15_Tactile/tactile_eval_client.py`
- Modify: `policy/GR00T_n15_Tactile/eval_double_env.sh`

1. Validate checkpoint-defined site names and tactile dimensions.
2. Encode online TacMap frames in the same order and scale as training.
3. Reuse the project TacMap rig to capture tactile observations during rollout.
4. Send RGB, qpos, language, and tactile fields through the GR00T inference path.

### Task 6: Document and verify

**Files:**
- Create or replace: `policy/GR00T_n15_Tactile/README.md`

1. Document environment setup, dataset requirements, training, inference, schema checks, and ablations.
2. Run pure data tests, PyTorch tests when a GR00T environment is available, compile all Python files, and run shell syntax checks.
3. Verify all new tactile implementation remains scoped to `policy/GR00T_n15_Tactile`.
