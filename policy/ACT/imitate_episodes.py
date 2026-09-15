import os
import gc
import ctypes

import torch
import numpy as np
import pickle
import argparse
import glob

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_dex2bench_data
from utils import compute_dict_mean, set_seed, detach_dict
from act_policy import ACTPolicy, CNNMLPPolicy


def main(args):
    set_seed(1)
    policy_class = args["policy_class"]
    batch_size_train = args["batch_size"]
    batch_size_val = args["batch_size"]
    num_epochs = args["num_epochs"]

    task_scene = args.get("task")
    if task_scene is None:
        raise ValueError("--task is required, e.g. 06_fruit_bowl_loading")
    task_name = f"dex2bench-{task_scene}"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dex2bench_root = os.path.join(script_dir, "..", "..")

    dataset_dir_override = args.get("dataset_dir")
    if dataset_dir_override:
        dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir_override))
    else:
        dataset_dir = os.path.normpath(os.path.join(
            dex2bench_root, "outputs", "ur5_rh56dfx", "scenes", task_scene, "replay"
        ))

    hdf5_files = glob.glob(os.path.join(dataset_dir, "*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {dataset_dir}")
    episode_len = 1200
    camera_names = [
        "cam_right_wrist", "cam_left_wrist",
        "cam_stereo_left", "cam_stereo_right",
    ]

    if not args.get("ckpt_dir"):
        from datetime import datetime as _dt
        exp_tag = _dt.now().strftime("exp_%Y%m%d_%H%M%S")
        args["ckpt_dir"] = os.path.normpath(os.path.join(
            dex2bench_root, "outputs", "logs", "act", task_scene, exp_tag
        ))
    ckpt_dir = args["ckpt_dir"]

    state_dim = args["state_dim"]
    lr_backbone = 1e-5
    backbone = "resnet18"
    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "state_dim": state_dim,
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": args["lr"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "num_queries": 1,
            "state_dim": state_dim,
            "camera_names": camera_names,
        }
    else:
        raise NotImplementedError

    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": ckpt_dir,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        "step_delay": args.get("step_delay", 0.0),
    }

    train_dataloader, val_dataloader, stats, _ = load_dex2bench_data(
        dataset_dir,
        camera_names,
        batch_size_train,
        batch_size_val,
        use_active_dof=bool(args.get("use_active_dof", True)),
        robot_key=args.get("robot_key"),
    )
    data_state_dim = int(stats["action_mean"].shape[0])
    if data_state_dim != int(state_dim):
        print(f"[ACT] Overriding state_dim from {state_dim} to data dim {data_state_dim}")
        state_dim = data_state_dim
        policy_config["state_dim"] = data_state_dim
        config["state_dim"] = data_state_dim

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, best_loss = best_ckpt_info
    metric_name = "val loss" if val_dataloader is not None else "train loss"
    print(f"Best ckpt, {metric_name} {best_loss:.6f} @ epoch{best_epoch}")


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        return ACTPolicy(policy_config)
    if policy_class == "CNNMLP":
        return CNNMLPPolicy(policy_config)
    raise NotImplementedError


def make_optimizer(policy_class, policy):
    if policy_class in ("ACT", "CNNMLP"):
        return policy.configure_optimizers()
    raise NotImplementedError


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data = image_data.cuda(non_blocking=True).float()
    qpos_data = qpos_data.cuda(non_blocking=True)
    action_data = action_data.cuda(non_blocking=True)
    is_pad = is_pad.cuda(non_blocking=True)
    return policy(qpos_data, image_data, action_data, is_pad)


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    step_delay = config.get("step_delay", 0.0)

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, "tb_logs"))
    except ImportError:
        writer = None
        print("[WARN] tensorboard not available; install tensorboard to enable loss curves")

    train_history = []
    validation_history = []
    min_best_loss = np.inf
    best_ckpt_info = None
    has_validation = val_dataloader is not None
    best_ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
    last_ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt")
    last_ckpt_save_freq = 1000
    if not has_validation:
        print("[ACT] Validation disabled; using all episodes for training.")
    print(f"[ACT] policy_last.ckpt will be overwritten every {last_ckpt_save_freq} epochs.")

    for epoch in tqdm(range(num_epochs), disable=not os.isatty(1)):
        # validation
        epoch_val_loss = None
        val_epoch_summary = None
        if has_validation:
            with torch.inference_mode():
                policy.eval()
                epoch_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy)
                    epoch_dicts.append(detach_dict(forward_dict))
                    del forward_dict, data
                val_epoch_summary = compute_dict_mean(epoch_dicts)
                validation_history.append(val_epoch_summary)
                del epoch_dicts

            epoch_val_loss = val_epoch_summary["loss"]
            val_summary_string = " ".join(
                f"{k}: {v.item():.3f}" for k, v in val_epoch_summary.items()
            )
            if epoch_val_loss < min_best_loss:
                min_best_loss = epoch_val_loss.detach().cpu()
                best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in policy.state_dict().items()
                }
                torch.save(best_state_dict, best_ckpt_path)
                best_ckpt_info = (epoch, min_best_loss)
            tqdm.write(f"Epoch {epoch:4d} | val_loss={epoch_val_loss:.5f}", end="")
            print(f"Epoch {epoch:4d} | val_loss={epoch_val_loss:.5f} | {val_summary_string}")

        # training
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_train_dicts = []
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_train_dicts.append(detach_dict(forward_dict))
            del loss, forward_dict, data
            if step_delay > 0:
                import time; time.sleep(step_delay)
        epoch_summary = compute_dict_mean(epoch_train_dicts)
        train_history.append(epoch_summary)
        del epoch_train_dicts
        epoch_train_loss = epoch_summary["loss"]
        train_summary_string = " ".join(
            f"{k}: {v.item():.3f}" for k, v in epoch_summary.items()
        )
        if not has_validation and epoch_train_loss < min_best_loss:
            min_best_loss = epoch_train_loss.detach().cpu()
            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in policy.state_dict().items()
            }
            torch.save(best_state_dict, best_ckpt_path)
            best_ckpt_info = (epoch, min_best_loss)
        if has_validation:
            tqdm.write(f" | train_loss={epoch_train_loss:.5f}")
        else:
            tqdm.write(f"Epoch {epoch:4d} | train_loss={epoch_train_loss:.5f}")
        print(f"Epoch {epoch:4d} | train_loss={epoch_train_loss:.5f} | {train_summary_string}")

        if writer is not None:
            writer.add_scalar("Loss/train_action", epoch_train_loss.item(), epoch)
            if "l1" in epoch_summary:
                writer.add_scalar("Loss/train_l1", epoch_summary["l1"].item(), epoch)
            if "kl" in epoch_summary:
                writer.add_scalar("Loss/train_kl", epoch_summary["kl"].item(), epoch)
            if epoch_val_loss is not None:
                writer.add_scalar("Loss/val_action", epoch_val_loss.item(), epoch)
            if val_epoch_summary is not None and "l1" in val_epoch_summary:
                writer.add_scalar("Loss/val_l1", val_epoch_summary["l1"].item(), epoch)
            if val_epoch_summary is not None and "kl" in val_epoch_summary:
                writer.add_scalar("Loss/val_kl", val_epoch_summary["kl"].item(), epoch)

        del epoch_summary
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        should_save_last = (epoch + 1) % last_ckpt_save_freq == 0 or epoch == num_epochs - 1
        if should_save_last:
            torch.save(policy.state_dict(), last_ckpt_path)

    best_epoch, best_loss = best_ckpt_info
    metric_name = "val loss" if has_validation else "train loss"
    print(f"Training finished:\nSeed {seed}, {metric_name} {best_loss:.6f} at epoch {best_epoch}")

    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)
    if writer is not None:
        writer.close()

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    if not train_history:
        return

    has_validation = bool(validation_history)
    for key in train_history[0]:
        prefix = "train_val" if has_validation else "train"
        plot_path = os.path.join(ckpt_dir, f"{prefix}_{key}_seed_{seed}.png")
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        plt.plot(
            np.linspace(0, num_epochs - 1, len(train_history)),
            train_values,
            label="train",
        )
        if has_validation:
            val_values = [summary[key].item() for summary in validation_history]
            plt.plot(
                np.linspace(0, num_epochs - 1, len(validation_history)),
                val_values,
                label="validation",
            )
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        plt.close()
    print(f"Saved plots to {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="Checkpoint dir. Auto-set when omitted.")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Directory containing dex2bench episode_*.hdf5 replay files. "
                             "Overrides the --task default path.")
    parser.add_argument("--policy_class", type=str, required=True, help="ACT or CNNMLP")
    parser.add_argument("--task", type=str, required=True,
                        help="Task scene name, e.g. 06_fruit_bowl_loading")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)

    parser.add_argument("--kl_weight", type=float, required=False)
    parser.add_argument("--chunk_size", type=int, required=False)
    parser.add_argument("--hidden_dim", type=int, required=False)
    parser.add_argument("--state_dim", type=int, required=True)
    parser.add_argument("--robot_key", type=str, default=None)
    parser.add_argument("--use_active_dof", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--step_delay", type=float, default=0.0,
                        help="Sleep seconds after each train step (rate-limit GPU)")
    parser.add_argument("--dim_feedforward", type=int, required=False)
    parser.add_argument("--temporal_agg", action="store_true")

    main(vars(parser.parse_args()))
