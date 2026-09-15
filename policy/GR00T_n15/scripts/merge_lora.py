#!/usr/bin/env python3
"""Merge LoRA adapter into base model and resize action head to match data dim.

Usage:
    python policy/GR00T_n15/scripts/merge_lora.py \
        --base-model ../groot/groot_baseline_ckpt \
        --checkpoint ../groot/06_lora/checkpoint-10000 \
        --output ../groot/06_lora/merged \
        --action-dim 36
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch

# Ensure GR00T source is importable
REPO_ROOT = Path(__file__).resolve().parents[3]
GR00T_SRC = REPO_ROOT / "policy" / "GR00T_n15" / "src"
sys.path.insert(0, str(GR00T_SRC))

from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead


COMPUTE_DTYPE = torch.bfloat16


def copy_partial_action_expert_weights(old_dict, new_dict, old_dim, new_dim):
    """Copy weights with partial dimension matching for action_dim changes."""
    total_params = copied_params = random_params = 0

    for key, old_tensor in old_dict.items():
        if key not in new_dict:
            continue
        new_tensor = new_dict[key]
        total_params += new_tensor.numel()

        if old_tensor.shape == new_tensor.shape:
            new_tensor.copy_(old_tensor)
            copied_params += new_tensor.numel()
        elif "action_encoder" in key and "W1.weight" in key:
            new_tensor[:, :old_dim] = old_tensor
            copied_params += old_tensor.numel()
            random_params += new_tensor.numel() - old_tensor.numel()
        elif "action_decoder" in key and ("weight" in key or "bias" in key):
            if old_tensor.dim() == 1:
                new_tensor[:old_dim] = old_tensor
            elif old_tensor.dim() == 2:
                new_tensor[:, :old_dim] = old_tensor
            elif old_tensor.dim() == 3:
                new_tensor[:, :, :old_dim] = old_tensor
            copied_params += old_tensor.numel()
            random_params += new_tensor.numel() - old_tensor.numel()
        else:
            random_params += new_tensor.numel()

    assert total_params == copied_params + random_params, "Parameter count mismatch"
    random_percentage = (random_params / total_params) * 100 if total_params > 0 else 0
    print(
        f"Weight copy stats: {copied_params:,} copied, {random_params:,} random "
        f"({random_percentage:.1f}% randomly initialized)"
    )
    print(f"Action dimensions {old_dim+1}-{new_dim} will be learned from scratch")
    return new_dict


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter and resize action head")
    parser.add_argument("--base-model", required=True, help="Path to base GR00T model")
    parser.add_argument("--checkpoint", required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--output", required=True, help="Output path for merged model")
    parser.add_argument("--action-dim", type=int, default=36, help="Target action dimension")
    parser.add_argument("--action-horizon", type=int, default=16, help="Action horizon")
    args = parser.parse_args()

    base_model_path = Path(args.base_model)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    # Verify checkpoint has experiment_cfg (required for inference)
    experiment_cfg = checkpoint_path / "experiment_cfg"
    if not experiment_cfg.is_dir():
        print(f"ERROR: checkpoint missing experiment_cfg/ at {checkpoint_path}")
        sys.exit(1)

    print(f"Loading base model from: {base_model_path}")
    model = GR00T_N1_5.from_pretrained(str(base_model_path), torch_dtype=COMPUTE_DTYPE)
    old_action_dim = model.action_head.config.action_dim
    old_action_horizon = model.action_head.config.action_horizon
    print(f"Base model action_dim={old_action_dim}, action_horizon={old_action_horizon}")

    # Load and merge LoRA adapter
    lora_config_path = checkpoint_path / "adapter_config.json"
    if lora_config_path.is_file():
        print(f"Loading LoRA adapter from: {checkpoint_path}")
        from peft import PeftModel
        peft_model = PeftModel.from_pretrained(model, str(checkpoint_path))
        print("Merging LoRA adapter into base model...")
        model = peft_model.merge_and_unload()
        print("LoRA merged successfully.")
    else:
        print("No LoRA adapter found, using base model as-is.")

    # Resize action head if needed
    need_resize = (
        old_action_dim != args.action_dim
        or old_action_horizon != args.action_horizon
    )

    if need_resize:
        print(f"\nResizing action head: dim {old_action_dim}→{args.action_dim}, "
              f"horizon {old_action_horizon}→{args.action_horizon}")

        new_config = copy.deepcopy(model.action_head.config)
        new_config.action_dim = args.action_dim
        new_config.action_horizon = args.action_horizon

        new_action_head = FlowmatchingActionHead(new_config)

        # Partial weight copy
        old_sd = model.action_head.state_dict()
        new_sd = new_action_head.state_dict()
        new_sd = copy_partial_action_expert_weights(
            old_sd, new_sd, old_action_dim, args.action_dim
        )
        new_action_head.load_state_dict(new_sd)

        # Replace action head
        model.action_head = new_action_head

        # Update config
        model.config.action_horizon = args.action_horizon
        model.config.action_head_cfg["action_horizon"] = args.action_horizon
        model.config.action_head_cfg["action_dim"] = args.action_dim
        model.config.action_dim = args.action_dim
        model.action_dim = args.action_dim
        model.action_horizon = args.action_horizon

        print("Action head resized successfully.")

    # Save merged model
    print(f"\nSaving merged model to: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))

    # Copy experiment_cfg for inference
    import shutil
    target_exp = output_path / "experiment_cfg"
    if not target_exp.is_dir():
        shutil.copytree(str(experiment_cfg), str(target_exp))
        print(f"Copied experiment_cfg to merged model.")

    # Save config
    model.config.save_pretrained(str(output_path))

    print(f"\nDone! Merged model saved to {output_path}")
    print(f"Use this path as MODEL_PATH for eval:")
    print(f"  --model_path {output_path}")


if __name__ == "__main__":
    main()
