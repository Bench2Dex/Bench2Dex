# Evaluation Devices

All policy evaluation runners default to **CPU physics**. This does not make
RGB evaluation GPU-free: camera rendering, model inference, and TacMap compute
still use the GPU by default.

| Component | Default | Override |
| --- | --- | --- |
| Physics simulation | `cpu` | `--device cpu` / `--device cuda:0` |
| Local ACT/DP/ACT-Tactile inference | `cuda:0` | `--policy-device cuda:0` on the direct runner |
| Remote model server | Existing policy configuration | Model configuration `device` |
| TacMap compute | `cuda:0` | `--tactile-device cuda:0` on tactile evaluation |
| RGB rendering | GPU | Isaac Sim rendering configuration |

`--sim-device` is an alias of `--device`. `--gpu` (or the wrapper's GPU
positional argument) selects visible GPUs; it does not select CPU/GPU physics.
`cuda:0` refers to the first GPU visible to that process.

The main `eval_double_env.sh` wrappers and ACT-Tactile's `eval_direct.sh`
default to CPU physics without an extra flag. Explicit `--device cuda:0`
restores GPU physics. The pi05 four-channel wrapper forwards this choice to
each evaluation process.

Config-driven clients use **`sim_device`** for physics and leave the model's
**`device`** untouched. A client CLI `--device` override is interpreted as
`sim_device`. A trailing `run_policy_args` device flag takes precedence.
The standalone Python runners also default to CPU physics.

TacMap has a separate `compute_device`: physics views remain on the simulation
device, while live sensor/target poses are copied to the tactile device before
ray casting. Rigid bodies and articulated target links both use live physics
poses. Existing teleoperation/collection callers that do not specify
`compute_device` retain their previous behavior. Ray-casting sensors sharing
mesh paths in the same scene must use the same compute device.

Use a **new output directory** when switching physics backends; CPU and GPU
physics results should not be appended into the same evaluation run. Start
with one episode before launching a large evaluation. Device-routing and
tensor-path tests do not replace an Isaac Sim integration run with real
cameras, meshes and robot assets.

Tests (run from the repository root with the simulation environment's Python):

```bash
python -m unittest discover -s policy/tests -p test_eval_devices.py -v
python -m unittest discover -s collector/tests -p test_tacmap_devices.py -v
```

The CPU-to-CUDA tensor test skips when no CUDA device is visible.
