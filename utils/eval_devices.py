"""Keep evaluation physics, inference and tactile compute devices independent."""

import argparse


def normalize_eval_device_overrides(overrides):
    """Interpret a client CLI --device override as physics, not model config."""
    return [
        "--sim_device" if index % 2 == 0 and value == "--device" else value
        for index, value in enumerate(overrides or [])
    ]


def add_eval_device_arguments(parser, *, tactile=False):
    # AppLauncher already registers --device. Only change its evaluation default.
    parser.set_defaults(device="cpu")
    parser.add_argument("--sim-device", dest="device", default=argparse.SUPPRESS,
                        help="Physics device (alias of --device; default: cpu).")
    parser.add_argument("--policy-device", default="cuda:0",
                        help="Inference device for locally loaded models (default: cuda:0).")
    if tactile:
        parser.add_argument("--tactile-device", default="cuda:0",
                            help="TacMap compute device, independent of physics (default: cuda:0).")


def append_eval_device_arguments(command, config, *, tactile=False):
    # config['device'] belongs to the model, not the physics process.
    command.extend(["--device", str(config.get("sim_device") or "cpu")])
    if tactile and config.get("tactile_device"):
        command.extend(["--tactile-device", str(config["tactile_device"])])
