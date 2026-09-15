# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
from collections.abc import Mapping
from .backbone import build_backbone, build_tactile_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, tactile_cfg=None):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(state_dim, hidden_dim)  # project action to embedding
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)  # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1 + 1 + num_queries,
                                                                      hidden_dim))  # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)  # learned position embedding for proprio and latent

        self.tactile_cfg = self._validate_tactile_cfg(tactile_cfg)
        if self.tactile_cfg is not None:
            if self.backbones is None or len(self.backbones) < 2:
                raise ValueError("tactile_cfg requires a dedicated tactile backbone (len(backbones) >= 2)")
            self.tactile_input_proj = nn.Conv2d(backbones[1].num_channels, hidden_dim, kernel_size=1)
            self.tactile_site_embedding = nn.Embedding(self.tactile_cfg["num_sites"], hidden_dim)
            self.tactile_modality_embedding = nn.Embedding(1, hidden_dim)
            self.tactile_pool_query = nn.Embedding(1, hidden_dim)
            self.tactile_attention_pool = nn.MultiheadAttention(
                hidden_dim,
                transformer.nhead,
                batch_first=True,
            )
            self.tactile_pool_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _validate_tactile_cfg(tactile_cfg):
        if not isinstance(tactile_cfg, Mapping):
            raise ValueError("tactile_cfg must be a mapping")
        required = ("num_sites", "height", "width", "site_names")
        missing = [key for key in required if key not in tactile_cfg]
        if missing:
            raise ValueError(f"tactile_cfg is missing required keys: {missing}")
        validated = {}
        for key in ("num_sites", "height", "width"):
            value = tactile_cfg[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"tactile_cfg.{key} must be a positive integer")
            validated[key] = value
        site_names = tactile_cfg["site_names"]
        if isinstance(site_names, str) or not isinstance(site_names, (tuple, list)):
            raise ValueError("tactile_cfg.site_names must be a sequence of names")
        if len(site_names) != validated["num_sites"] or len(set(site_names)) != len(site_names):
            raise ValueError("tactile_cfg.site_names must be unique and match num_sites")
        validated["site_names"] = tuple(site_names)
        return validated

    def _encode_tactile(self, tactile, batch_size):
        if not isinstance(tactile, torch.Tensor):
            raise ValueError("tactile input is required and must be a torch.Tensor")
        expected = (batch_size, self.tactile_cfg["num_sites"], 1, self.tactile_cfg["height"], self.tactile_cfg["width"])
        if tuple(tactile.shape) != expected:
            raise ValueError(f"tactile input must have shape {expected}, got {tuple(tactile.shape)}")
        if tactile.device != next(self.backbones[1].parameters()).device:
            raise ValueError("tactile input and tactile backbone must be on the same device")
        if tactile.dtype == torch.uint8:
            tactile = tactile.float().div(255.0)
        elif torch.is_floating_point(tactile):
            if not torch.isfinite(tactile).all() or torch.any(tactile < 0) or torch.any(tactile > 1):
                raise ValueError("floating tactile input must be finite and in [0, 1]")
        else:
            raise ValueError("tactile input must be uint8 or floating point")
        tactile = tactile.to(dtype=next(self.backbones[1].parameters()).dtype)
        tactile = tactile.reshape(-1, 1, self.tactile_cfg["height"], self.tactile_cfg["width"])
        features, _ = self.backbones[1](tactile)
        tokens = self.tactile_input_proj(features[0]).mean(dim=(-2, -1))
        tokens = tokens.reshape(batch_size, self.tactile_cfg["num_sites"], -1)
        site_ids = torch.arange(self.tactile_cfg["num_sites"], device=tokens.device)
        site_tokens = tokens + self.tactile_site_embedding(site_ids).unsqueeze(0)
        query = self.tactile_pool_query.weight.unsqueeze(0).expand(batch_size, -1, -1)
        need_weights = bool(getattr(self, "capture_tactile_attention", False))
        attended, attn_weights = self.tactile_attention_pool(
            query,
            site_tokens,
            site_tokens,
            need_weights=need_weights,
            average_attn_weights=True,
        )
        if need_weights:
            # attn_weights: [B, 1, num_sites] averaged over heads
            self.last_tactile_attn_weights = attn_weights.detach()
        pooled = self.tactile_pool_norm(query + attended)
        positions = self.tactile_modality_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
        return pooled, positions

    def _modality_attention_from_cross_attn(self, visual_src: torch.Tensor):
        """Aggregate decoder cross-attention into [B, 4] modality proportions.

        Memory layout is [latent, proprio, visual*HW, tactile] (see
        Transformer.forward); weights are averaged over decoder layers and
        queries, then summed per modality group.
        """
        layer_weights = []
        for layer in self.transformer.decoder.layers:
            weights = getattr(layer, "last_cross_attn_weights", None)
            if weights is not None:
                layer_weights.append(weights)
        if not layer_weights:
            return None
        stacked = torch.stack(layer_weights)                      # [n_layers, B, L, S]
        per_pos = stacked.mean(dim=0).mean(dim=1)                 # [B, S]
        hw = int(visual_src.shape[2]) * int(visual_src.shape[3])
        latent = per_pos[:, 0]
        proprio = per_pos[:, 1]
        visual = per_pos[:, 2:2 + hw].sum(dim=1)
        tactile = per_pos[:, 2 + hw:].sum(dim=1)
        return torch.stack([latent, proprio, visual, tactile], dim=1).detach()  # [B, 4]

    def forward(self, qpos, image, env_state, actions=None, is_pad=None, tactile=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None  # train or val
        bs, _ = qpos.shape
        tactile_features, tactile_pos = self._encode_tactile(tactile, bs)
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)  # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1)  # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device)  # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0]  # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            # print("image.shape in detr_vae", image.shape,"camera_names", self.camera_names)
            for cam_id, cam_name in enumerate(self.camera_names):
                # print("cam_id", cam_id, "cam_name", cam_name)
                features, pos = self.backbones[0](image[:, cam_id])  # HARDCODED
                features = features[0]  # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            _capture_mod = bool(getattr(self, "capture_modality_attention", False))
            if _capture_mod:
                for _layer in self.transformer.decoder.layers:
                    _layer.capture_cross_attn = True
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input,
                                  self.additional_pos_embed.weight, tactile_features, tactile_pos)[0]
            if _capture_mod:
                self._last_modality_attention = self._modality_attention_from_cross_attn(src)
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1)  # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]


class CNNMLP(nn.Module):

    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim)  # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                                          nn.Conv2d(128, 64, kernel_size=5), nn.Conv2d(64, 32, kernel_size=5))
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + 14
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=state_dim, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None  # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0]  # take the last layer feature
            pos = pos[0]  # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1)  # 768 each
        features = torch.cat([flattened_features, qpos], axis=1)  # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim  # 256
    dropout = args.dropout  # 0.1
    nhead = args.nheads  # 8
    dim_feedforward = args.dim_feedforward  # 2048
    num_encoder_layers = args.enc_layers  # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm  # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args, tactile_cfg=None):
    state_dim = getattr(args, 'state_dim', 14)  # configurable, default 14 for legacy tasks

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image: separate backbones for RGB (3-ch) and tactile (1-ch)
    backbones = []
    backbones.append(build_backbone(args))          # [0] RGB backbone
    if tactile_cfg is not None:
        backbones.append(build_tactile_backbone(args))  # [1] tactile backbone

    transformer = build_transformer(args)

    encoder = build_encoder(args)

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.chunk_size
        camera_names=args.camera_names,
        tactile_cfg=tactile_cfg,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6, ))

    return model


def build_cnnmlp(args):
    state_dim = getattr(args, 'state_dim', 16)  # configurable, default 16 for legacy tasks

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6, ))

    return model
