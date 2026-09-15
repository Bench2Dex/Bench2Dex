"""
Usage:
    python train.py --config-name=robot_dp_36_dex2scene_pretrained.yaml \
        training.device="cuda:0" dataloader.batch_size=4 logging.mode=offline
"""

import os
import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import pathlib

import hydra
import torch
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("diffusion_policy", "config")),
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    # --- DDP initialization via torchrun ---
    local_rank = os.environ.get("LOCAL_RANK", None)
    if local_rank is not None:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        cfg.training.local_rank = int(local_rank)
        cfg.training.rank = dist.get_rank()
        cfg.training.world_size = dist.get_world_size()
        cfg.training.multi_gpu = True
        # each rank must use its own GPU
        cfg.training.device = f"cuda:{cfg.training.local_rank}"
        torch.cuda.set_device(cfg.training.local_rank)
    else:
        cfg.training.local_rank = 0
        cfg.training.rank = 0
        cfg.training.world_size = 1
    # -----------------------------------------

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    dataset_path = cfg.task.dataset.get("zarr_path", cfg.task.dataset.get("dataset_dir", ""))
    if cfg.training.rank == 0:
        print(dataset_path, cfg.task_name)
    workspace.run()

    # cleanup DDP
    if cfg.training.multi_gpu:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
