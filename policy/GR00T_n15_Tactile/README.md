# GR00T N1.5 Tactile

This directory is a standalone tactile fork of `policy/GR00T_n15`. The original
GR00T policy remains unchanged. The tactile fork consumes four RGB cameras,
active-DOF proprioception, a task instruction, and synchronized TacMap images.

## Tactile fusion

For each policy query, the dataset supplies TacMap as `[N, 1, H, W]`, where
`N` is the checkpoint-defined tactile-site count. All sites share a
single-channel CNN. Each pooled site feature receives a learned site identity
embedding, and a learned query performs multi-head attention pooling over the
sites. The resulting single tactile token is projected to the GR00T backbone
dimension and appended to the Eagle vision-language tokens before the action
head's vision-language self-attention and cross-attention.

The Eagle language and vision towers keep their normal GR00T tuning settings.
The tactile encoder is initialized from scratch and trained together with the
flow-matching action head.

## Dataset requirements

Training directories must contain `episode_*.hdf5` files with:

- `robot/qpos`
- `action/commanded`
- the four RGB streams used by GR00T
- `robot/tactile/meta/site_names`
- `robot/tactile/tacmap/{site_name}` as `[T,H,W]` `uint8`
- `time/sim_step` and `meta/homing_start_sim_step`

Every episode in one training run must use exactly the same tactile-site order
and native TacMap shape. The standard launcher excludes the homing frame and all
later return-to-home frames. TacMap is resized to `120x120` by default and
normalized to `[0,1]`.

The trained checkpoint stores the site order, native TacMap shape, encoder input
shape, and tactile encoder configuration. Inference rejects missing, extra, or
reordered sites.

## Environment setup

The dependencies are the same as the vendored GR00T policy:

```bash
bash policy/GR00T_n15_Tactile/setup_env.sh
```

For inference, use two environments:

- the GR00T environment runs the policy server;
- the dex2bench/Isaac Lab environment runs simulation, RGB capture, and TacMap
  ray casting.

## Training

Basic task-specific training:

```bash
bash policy/GR00T_n15_Tactile/train.sh 06 \
  --dataset ../teleopdata/dataset/06_fruit_bowl_loading/replay-generalization \
  --gpu 0 --batch 64 --steps 20000 --no-tmux
```

The launcher validates homing markers, loads RGB/qpos/action/language/TacMap
directly from HDF5, validates the tactile schema, recreates the action head with
the tactile token encoder, copies compatible pretrained weights, and stores the
tactile schema in the checkpoint configuration.

`train.sh` currently writes to a `gr00t_n15_trunc*` directory, which the tactile
evaluation launcher does not auto-discover. For automatic discovery, deploy the
final checkpoint as
`../policy_ckpt/<TASK_ID>/<robot_key>/gr00t_n15_tactile`. If it remains at the
training output path, pass that actual checkpoint (including a nested
`checkpoint-*` directory when applicable) with `--model-path`.

Tactile input size is configured in `deploy_policy.yml`:

```yaml
tactile_height: 120
tactile_width: 120
```

## Evaluation

### One-command double-environment evaluation

The primary supported evaluation entry point is:

```bash
bash policy/GR00T_n15_Tactile/eval_double_env.sh 03 all --sii --headless --tmux
```

Run it from the repository root. The launcher starts the tactile policy server
in the GR00T/`groot` environment, waits for it to become ready, then runs the
tactile Isaac client in the dex2bench/Isaac environment. With `--tmux`, it
manages both sides from one terminal and cleans up the server between Isaac
processes.

By default, the launcher discovers the checkpoint at:

```text
../policy_ckpt/<TASK_ID>/<robot_key>/gr00t_n15_tactile
```

It searches only `gr00t_n15_tactile*`, preferring the exact directory name;
tactile-named variants and their nested `checkpoint-*` directories are also
supported. To bypass discovery, pass the checkpoint explicitly:

```bash
bash policy/GR00T_n15_Tactile/eval_double_env.sh 03 all \
  --model-path ../policy_ckpt/03/multi_ur5_rh56dfx_with_flange/gr00t_n15_tactile \
  --sii --headless --tmux
```

`all` evaluates the four channels `none`, `cov_only`, `inv_only`, and
`inv_cov`, then aggregates their results. Isaac restarts every 25 episodes by
default; change this with `--episodes-per-process N`. Recording is enabled by
default, and interrupted runs can continue with `--resume` (or
`--resume-dir PATH`) while preserving per-channel output and aggregation.

### Two-terminal manual debugging alternative

For server/client debugging, the two sides can still be started manually. Run
both commands from the repository root.

Terminal 1, GR00T/`groot` environment:

```bash
source ../miniconda3/bin/activate groot
export PYTHONPATH="$PWD/policy/GR00T_n15_Tactile/src:${PYTHONPATH:-}"
python script/policy_model_server.py \
  --host 127.0.0.1 --port 9000 \
  --config policy/GR00T_n15_Tactile/deploy_policy.yml \
  --overrides \
  --model_path ../policy_ckpt/03/multi_ur5_rh56dfx_with_flange/gr00t_n15_tactile \
  --robot_key multi_ur5_rh56dfx_with_flange \
  --data_config policy.GR00T_n15_Tactile.gr00t_dex2bench_config:Dex2BenchGR00TDataConfig
```

Terminal 2, dex2bench Isaac Lab environment:

```bash
source ../isaaclab_setup/env_new.sh
start_xvfb
activate_isaaclab
setup_nvidia_libs

python policy/GR00T_n15_Tactile/run_policy.py \
  --task scenes/03_wine_glass_plate_balance.yaml \
  --robot-key multi_ur5_rh56dfx_with_flange \
  --host 127.0.0.1 --port 9000 \
  --enable-rgb --active-dof --device cuda:0 \
  --num-episodes 10 --episode-steps 800 \
  --headless
```

The client constructs TacMap online with `TacMapRig`, sends RGB/qpos/language/
TacMap observations to the server, and executes returned action chunks. The
server orders and resizes tactile sites according to the checkpoint schema.

Do not use the standard `policy/GR00T_n15` client for tactile evaluation: it does
not capture or transmit tactile observations.

## Verification and ablations

Pure HDF5/schema and wiring tests:

```bash
python3 -m unittest discover -s policy/GR00T_n15_Tactile/tests -v
```

In the GR00T environment, additionally run a short training smoke test. To test
whether the policy uses touch, compare normal TacMap input with all-zero and
within-episode temporally shuffled TacMap while keeping RGB and qpos fixed.

## Current limitations

- A checkpoint has a fixed ordered site set. Cross-embodiment training requires
  a canonical site vocabulary plus padding and masking.
- Only quantized TacMap images are consumed; `contact_mask` and raw metric depth
  are not yet fused.
- Full PyTorch forward/backward verification requires the GR00T environment and
  a compatible base checkpoint.
