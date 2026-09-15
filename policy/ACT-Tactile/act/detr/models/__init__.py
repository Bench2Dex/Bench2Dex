# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr_vae import build as build_vae
from .detr_vae import build_cnnmlp as build_cnnmlp


def build_ACT_model(args, tactile_cfg=None):
    return build_vae(args, tactile_cfg=tactile_cfg)


def build_CNNMLP_model(args):
    return build_cnnmlp(args)
