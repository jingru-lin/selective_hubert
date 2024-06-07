# --------------------------------------------------------
# WavLM: Large-Scale Self-Supervised  Pre-training  for Full Stack Speech Processing (https://arxiv.org/abs/2110.13900.pdf)
# Github source: https://github.com/microsoft/unilm/tree/master/wavlm
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------
# Note: this implementation is modified based on
# https://github.com/microsoft/unilm/tree/master/wavlm
# and adpated to fit the fairseq module definition.
# --------------------------------------------------------

import logging
from typing import Dict, List, Optional, Tuple

from dataclasses import dataclass, field
import numpy as np
from omegaconf import II
import torch
import torch.nn as nn
from torch.nn import LayerNorm

from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules import GradMultiply

from selective_hubert.tasks.shubert_pretraining import (
    SHubertPretrainingConfig,
    SHubertPretrainingTask
)
from selective_hubert.models.wavlm_extraction.wavlm_encoder import (
    ConvFeatureExtractionModel, TransformerEncoder
)


logger = logging.getLogger(__name__)

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])


@dataclass
class SWavLMConfig(FairseqDataclass):
    label_rate: int = II("task.label_rate")

    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group "
            "norm with d groups in the first conv block, whereas layer_norm "
            "has layer norms in every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )

    # dropouts
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the transformer"},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN"},
    )
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a tarnsformer layer"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many "
            "dimensions. set to encoder_embed_dim is <= 0"
        },
    )
    untie_final_proj: bool = field(
        default=False,
        metadata={"help": "use separate projection for each target"},
    )
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"},
    )
    conv_feature_layers: str = field(
        default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        metadata={
            "help": "string describing convolutional feature extraction "
            "layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply feature extractor var grads by this"},
    )

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )

    # loss computation
    skip_masked: bool = field(
        default=False,
        metadata={"help": "skip computing losses over masked frames"},
    )
    skip_nomask: bool = field(
        default=False,
        metadata={"help": "skip computing losses over unmasked frames"},
    )

    normalize: bool = field(
        default=False,
        metadata={"help": "for compatibility with official WavLM models"},
    )

    ####################################################
    # new parameters (in addition to HuBERT) for WavLM #
    ####################################################
    relative_position_embedding: bool = field(
        default=False, metadata={"help": "apply relative position embedding"}
    )
    num_buckets: int = field(
        default=320,
        metadata={"help": "number of buckets for relative position embedding"},
    )
    max_distance: int = field(
        default=1280,
        metadata={"help": "maximum distance for relative position embedding"},
    )
    gru_rel_pos: bool = field(
        default=False, metadata={"help": "apply gated relative position embedding"}
    )
    expand_attention_head_size: int = field(default=-1, metadata={"help": "not used"})

    pretrained_ckpt_path: str = field(
        default="",
        metadata={"help": "pretrained hubert checkpoint"},
    )
    # speaker
    speaker_injection_layers: List[int] = field(
        default_factory=lambda: [5,6,7],
        metadata={"help": "layers to inject speaker embedding"},
    )
    speaker_dim: int = field(
        default=192,
        metadata={"help": "speaker embedding dimension"},
    )

    # ctr loss
    ctr_layer: int = field(
        default=-1,
        metadata={"help": "contrastive layers in the transformer"},
    )


@register_model("swavlm", dataclass=SWavLMConfig)
class SWavLM(BaseFairseqModel):
    def __init__(
        self,
        cfg: SWavLMConfig,
        task_cfg: Optional[SHubertPretrainingConfig] = None,
        dictionaries: List[Dictionary] = [None],
    ) -> None:
        super().__init__()
        logger.info(f"SWavLM Config: {cfg}")
        is_finetuning_or_inference = any([d is None for d in dictionaries])

        self.cfg = cfg
        feature_enc_layers = eval(cfg.conv_feature_layers)  # noqa
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )
        if not is_finetuning_or_inference:
            feature_ds_rate = np.prod([s for _, _, s in feature_enc_layers])
            self.feat2tar_ratio = (
                cfg.label_rate * feature_ds_rate / task_cfg.sample_rate
            )

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        if not is_finetuning_or_inference:
            self.logit_temp = cfg.logit_temp
            self.skip_masked = cfg.skip_masked
            self.skip_nomask = cfg.skip_nomask

            final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        self._load_weights(cfg.pretrained_ckpt_path)

        if not is_finetuning_or_inference:
            self.target_glu = None
            if cfg.target_glu:
                self.target_glu = nn.Sequential(
                    nn.Linear(final_dim, final_dim * 2), nn.GLU()
                )

            self.untie_final_proj = cfg.untie_final_proj
            if self.untie_final_proj:
                self.final_proj = nn.Linear(
                    cfg.encoder_embed_dim, final_dim * len(dictionaries)
                )
            else:
                self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)
        self.linear_block = LinearBlock(cfg.encoder_embed_dim)
        self.ctr_layer = cfg.ctr_layer

        # modules below are not needed during fine-tuning
        if is_finetuning_or_inference:
            logger.info("cannot find dictionary. assume will be used for fine-tuning")
        else:
            self.num_classes = [len(d) for d in dictionaries]
            self.label_embs_concat = nn.Parameter(
                torch.FloatTensor(sum(self.num_classes), final_dim)
            )
            nn.init.uniform_(self.label_embs_concat)

    def _load_weights(self, pretrained_ckpt_path):
        pretrained_model_dict = torch.load(pretrained_ckpt_path)['model']
        model_dict = self.state_dict()
        for name in model_dict.keys():
            if name in pretrained_model_dict.keys():
                model_dict[name] = pretrained_model_dict[name]
            # FIXME: munually setting the layers
            elif 'encoder.layers.' in name:
                if '_layer_norm.ln_weight_' in name:
                    logger.info(f"{name} is from Conditional Layer Norm, skipping")
                    pass
                else:
                    logger.info(f"{name} not in the model")
                    raise NotImplementedError
            else:
                logger.info(f"{name} not in the model")
                raise NotImplementedError
        self.load_state_dict(model_dict, strict=True)
        logger.info("Loaded pretrained pretrained parameters successfully")

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: SWavLMConfig, task: SHubertPretrainingTask):
        """Build a new model instance."""

        model = SWavLM(cfg, task.cfg, task.dictionaries)
        return model

    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices
    
    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits

    def forward_features(self, source: torch.Tensor) -> torch.Tensor:
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        return features

    def forward_targets(
        self,
        features: torch.Tensor,
        target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(2)
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
        target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
        target_list = [t[:, target_inds.long()] for t in target_list]
        return features, target_list

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def forward(
        self,
        source: torch.Tensor,
        spk_emb: torch.Tensor,
        paired_source: torch.Tensor = None,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        source = torch.cat([source, paired_source], dim=0) 
        if padding_mask is not None:
            padding_mask = torch.cat([padding_mask, padding_mask], dim=0)

        """output layer is 1-based"""
        features, features_pen, target_list, padding_mask = self.forward_conv(
            source, target_list=target_list, padding_mask=padding_mask
        )
        if target_list is not None:
            for j, t in enumerate(target_list): # jr: use same targets for both batches
                target_list[j] = t.repeat(2, 1)

        return self.forward_transformer(
            features,
            features_pen,
            spk_emb,
            target_list=target_list,
            padding_mask=padding_mask,
            mask=mask,
            features_only=features_only,
            output_layer=output_layer,
        )

    def forward_conv(
        self,
        source: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        features = self.forward_features(source)
        if target_list is not None:
            features, target_list = self.forward_targets(features, target_list)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        # unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        # unmasked_features = self.dropout_features(unmasked_features)
        return features, features_pen, target_list, padding_mask

    def forward_transformer(
        self,
        features: torch.Tensor,
        features_pen: torch.Tensor,
        spk_emb: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask)
        else:
            x = features
            mask_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, layer_results = self.encoder(
            x,
            spk_emb.repeat(2, 1) if spk_emb is not None else None, 
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )

        if features_only:
            raise NotImplementedError("Please use extract_features() to get features")


        def compute_pred(proj_x, target, label_embs):
            # compute logits for the i-th label set
            y = torch.index_select(label_embs, 0, target.long())
            negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
            if self.target_glu:
                y = self.target_glu(y)
                negs = self.target_glu(negs)
            # proj_x: (S, D)
            # y: (S, D)
            # negs: (Neg, S, D)
            return self.compute_nce(proj_x, y, negs)

        score_list = []
        if self.training:
            if self.ctr_layer == -1:
                B = x.shape[0]
                x_1, x_2 = torch.split(self.linear_block(x), B // 2, dim=0)
                score_list.append(self.barlow_twin_loss(x_1, x_2))
            else:
                raise NotImplementedError

        label_embs_list = self.label_embs_concat.split(self.num_classes, 0)

        if not self.skip_masked:
            masked_indices = torch.logical_and(~padding_mask, mask_indices)
            proj_x_m = self.final_proj(x[masked_indices])
            if self.untie_final_proj:
                proj_x_m_list = proj_x_m.chunk(len(target_list), dim=-1)
            else:
                proj_x_m_list = [proj_x_m for _ in range(len(target_list))]
            logit_m_list = [
                compute_pred(proj_x_m, t[masked_indices], label_embs_list[i])
                for i, (proj_x_m, t) in enumerate(zip(proj_x_m_list, target_list))
            ]
        else:
            logit_m_list = [None for _ in target_list]

        if not self.skip_nomask:
            nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
            proj_x_u = self.final_proj(x[nomask_indices])
            if self.untie_final_proj:
                proj_x_u_list = proj_x_u.chunk(len(target_list), dim=-1)
            else:
                proj_x_u_list = [proj_x_u for _ in range(len(target_list))]

            logit_u_list = [
                compute_pred(proj_x_u, t[nomask_indices], label_embs_list[i])
                for i, (proj_x_u, t) in enumerate(zip(proj_x_u_list, target_list))
            ]
        else:
            logit_u_list = [None for _ in target_list]

        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
            "score_list": score_list
        }
        return result

    def extract_features(
        self,
        source: torch.Tensor,
        spk_emb: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """output layer is 1-based"""
        features = self.forward_features(source) # B x D x T

        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask) # Get padding mask according to the shape of features

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)

        if mask: # FIXME NOT NEEDED NOW: use identical masking for both branches
            B, T, _ = features.shape
            mask_indices = self.get_mask(B, T, padding_mask)
            mask_indices = torch.from_numpy(mask_indices).to(features.device)
            features[mask_indices] = self.mask_emb
            x = features
        else:
            x = features
            mask_indices = None

        x, layer_results = self.encoder(
            x,
            spk_emb, # same spk embeddings for source and paired_source
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )

        return {"x": x, "padding_mask": padding_mask, "features": features, "layer_results": layer_results}

    def get_logits(self, net_output, is_masked=True):
        if is_masked:
            logits_list = net_output["logit_m_list"]
        else:
            logits_list = net_output["logit_u_list"]
        logits_list = [x.float() for x in logits_list if x is not None]
        return logits_list

    def get_targets(self, net_output, is_masked=True):
        logits_list = self.get_logits(net_output, is_masked)
        targets_list = [x.new_zeros(x.size(0), dtype=torch.long) for x in logits_list]
        return targets_list

    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []

        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        return extra_losses, names

    def remove_pretraining_modules(self):
        self.target_glu = None
        self.final_proj = None

    def barlow_twin_loss(self, z1, z2, num_samples=640, bt_lambda=0.005):
        """
        Feature dim size (z1, z2): (bs , t , p_dim)
           z1, z2 to be random samples of features along temporal frame

        Arguments
        ---------
        z1, z2: tensor, Embedded representation of wav
        num_samples: int, Determine number of token-frames to be sampled
        bt_lambda: float, Determine the penalization value of off-diagonal element to 0
        """
        B, T, C = z1.size()
        eye = torch.eye(C, dtype=bool, requires_grad=False)

        if num_samples:  # perform frame-token sampling
            ## get non zeros frames index (remove zero padding tokens from batch representation)
            # FIXME: the next line is not tested zero padding tokens as the pre-training does not use padding
            non_zeros = [T - (z1[i] == 0).sum(0).min() for i in range(B)]
            sample_indices = [[torch.randint(i, size=(int(i / sum(non_zeros) * num_samples),))] for i in non_zeros]
            z1 = torch.cat([z1[bs][i] for bs, i in enumerate(sample_indices)])
            z2 = torch.cat([z2[bs][i] for bs, i in enumerate(sample_indices)])
            T = num_samples

        z1_norm = (z1 - z1.mean(0)) / z1.std(0)
        z2_norm = (z2 - z2.mean(0)) / z2.std(0)

        c = torch.mm(z1_norm.T, z2_norm) / T

        c_diff = (c - eye.float().to(c.device)).pow(2)  # DxD
        invariance = c_diff * eye.float().to(c_diff.device)
        redundant = c_diff * (~eye).float().to(c_diff.device) * bt_lambda

        loss = torch.sum(invariance + redundant)

        return loss


class LinearBlock(nn.Module):
    def __init__(self, in_dim) -> None:
        super().__init__()
        self.up1 = nn.Linear(in_dim, 2048)
        self.norm1 = LayerNorm(2048)
        self.act1 = nn.SiLU()
        self.dropout1 = nn.Dropout(0.05)
        self.down1 = nn.Linear(2048, in_dim)
        self.norm2 = LayerNorm(in_dim)
        self.act2 = nn.SiLU()
        self.dropout2 = nn.Dropout(0.05)
        self.up2 = nn.Linear(in_dim, 2048)

    def forward(self, x):
        x = self.up1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.down1(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.dropout2(x)

        x = self.up2(x)
        return x
