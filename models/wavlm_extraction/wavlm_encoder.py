# --------------------------------------------------------
# WavLM: Large-Scale Self-Supervised  Pre-training  for Full Stack Speech Processing (https://arxiv.org/abs/2110.13900.pdf)
# Github source: https://github.com/microsoft/unilm/tree/master/wavlm
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------

import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from fairseq import utils
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    SamePad,
    TransposeLast,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from selective_hubert.models.wavlm_extraction.modules import GLU_Linear, MultiheadAttention
from selective_hubert.modules import CondLayerNorm

def manual_layer_norm(x, layer_weight, layer_bias, layer_eps):
    mean = x.mean(-1, keepdim=True)
    std = (x.var(-1, keepdim=True, unbiased=False) + layer_eps).sqrt()
    return layer_weight * (x - mean) / std + layer_bias


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
        conv_type: str = "default",
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        self.conv_type = conv_type
        if self.conv_type == "default":
            in_d = 1
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3, "invalid conv definition: " + str(cl)
                (dim, k, stride) = cl

                self.conv_layers.append(
                    block(
                        in_d,
                        dim,
                        k,
                        stride,
                        is_layer_norm=mode == "layer_norm",
                        is_group_norm=mode == "default" and i == 0,
                        conv_bias=conv_bias,
                    )
                )
                in_d = dim
        elif self.conv_type == "conv2d":
            in_d = 1
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3
                (dim, k, stride) = cl

                self.conv_layers.append(torch.nn.Conv2d(in_d, dim, k, stride))
                self.conv_layers.append(torch.nn.ReLU())
                in_d = dim
        elif self.conv_type == "custom":
            in_d = 1
            idim = 80
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3
                (dim, k, stride) = cl
                self.conv_layers.append(
                    torch.nn.Conv2d(in_d, dim, k, stride, padding=1)
                )
                self.conv_layers.append(torch.nn.LayerNorm([dim, idim]))
                self.conv_layers.append(torch.nn.ReLU())
                in_d = dim
                if (i + 1) % 2 == 0:
                    self.conv_layers.append(
                        torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
                    )
                    idim = int(math.ceil(idim / 2))
        else:
            pass

    def forward(self, x, mask=None):
        # BxT -> BxCxT
        x = x.unsqueeze(1)
        if self.conv_type == "custom":
            for conv in self.conv_layers:
                if isinstance(conv, nn.LayerNorm):
                    x = x.transpose(1, 2)
                    x = conv(x).transpose(1, 2)
                else:
                    x = conv(x)
            x = x.transpose(2, 3).contiguous()
            x = x.view(x.size(0), -1, x.size(-1))
        else:
            for conv in self.conv_layers:
                x = conv(x)
            if self.conv_type == "conv2d":
                b, c, t, f = x.size()
                x = x.transpose(2, 3).contiguous().view(b, c * f, t)
        return x


def rel_positional_encoding(feature_dim, kernel_size, groups):
    pos_conv = nn.Conv1d(
        feature_dim,
        feature_dim,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        groups=groups,
    )
    dropout = 0
    std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * feature_dim))
    nn.init.normal_(pos_conv.weight, mean=0, std=std)
    nn.init.constant_(pos_conv.bias, 0)

    pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
    pos_conv = nn.Sequential(pos_conv, SamePad(kernel_size), nn.GELU())
    return pos_conv


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        self.pos_conv = rel_positional_encoding(
            self.embedding_dim, args.conv_pos, args.conv_pos_groups
        )

        if hasattr(args, "relative_position_embedding"):
            self.relative_position_embedding = args.relative_position_embedding
            self.num_buckets = args.num_buckets
            self.max_distance = args.max_distance
        else:
            self.relative_position_embedding = False
            self.num_buckets = 0
            self.max_distance = 0

        self.speaker_injection_layers = args.speaker_injection_layers
        layers = []
        for i in range(args.encoder_layers):
            if i in self.speaker_injection_layers:
                layers.append(
                    TransformerSentenceEncoderLayer_1(
                        embedding_dim=self.embedding_dim,
                        ffn_embedding_dim=args.encoder_ffn_embed_dim,
                        num_attention_heads=args.encoder_attention_heads,
                        dropout=self.dropout,
                        attention_dropout=args.attention_dropout,
                        activation_dropout=args.activation_dropout,
                        activation_fn=args.activation_fn,
                        layer_norm_first=args.layer_norm_first,
                        has_relative_attention_bias=(
                                self.relative_position_embedding and i == 0
                        ),
                        num_buckets=self.num_buckets,
                        max_distance=self.max_distance,
                        gru_rel_pos=args.gru_rel_pos,
                        speaker_dim=args.speaker_dim,
                    )
                )
            else:
                layers.append(
                    TransformerSentenceEncoderLayer(
                        embedding_dim=self.embedding_dim,
                        ffn_embedding_dim=args.encoder_ffn_embed_dim,
                        num_attention_heads=args.encoder_attention_heads,
                        dropout=self.dropout,
                        attention_dropout=args.attention_dropout,
                        activation_dropout=args.activation_dropout,
                        activation_fn=args.activation_fn,
                        layer_norm_first=args.layer_norm_first,
                        has_relative_attention_bias=(
                            self.relative_position_embedding and i == 0
                        ),
                        num_buckets=self.num_buckets,
                        max_distance=self.max_distance,
                        gru_rel_pos=args.gru_rel_pos,
                    )
                )
        self.layers = nn.ModuleList(layers)

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(
        self,
        x,
        spk_emb,
        padding_mask=None,
        streaming_mask=None,
        layer=None,
    ):
        x, layer_results = self.extract_features(
            x,
            spk_emb,
            padding_mask,
            streaming_mask,
            layer,
        )

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(
        self,
        x,
        spk_emb,
        padding_mask=None,
        streaming_mask=None,
        tgt_layer=None,
    ):
        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        z = None
        if tgt_layer is not None:
            layer_results.append((x, z))
        r = None
        pos_bias = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop) or i in self.speaker_injection_layers:
                if i not in self.speaker_injection_layers:
                    x, z, pos_bias = layer(
                        x,
                        self_attn_padding_mask=padding_mask,
                        need_weights=False,
                        self_attn_mask=streaming_mask,
                        pos_bias=pos_bias,
                    )
                else:
                    x, z, pos_bias = layer(
                        x,
                        spk_emb,
                        self_attn_padding_mask=padding_mask,
                        need_weights=False,
                        self_attn_mask=streaming_mask,
                        pos_bias=pos_bias,
                    )
            if tgt_layer is not None:
                layer_results.append((x, z))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results

class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        has_relative_attention_bias: bool = False,
        num_buckets: int = 0,
        max_distance: int = 0,
        rescale_init: bool = False,
        gru_rel_pos: bool = False,
    ) -> None:
        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_name = activation_fn
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            has_relative_attention_bias=has_relative_attention_bias,
            num_buckets=num_buckets,
            max_distance=max_distance,
            rescale_init=rescale_init,
            gru_rel_pos=gru_rel_pos,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)

        if self.activation_name == "glu":
            self.fc1 = GLU_Linear(self.embedding_dim, ffn_embedding_dim, "swish")
        else:
            self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        pos_bias=None
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
                position_bias=pos_bias
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                attn_mask=self_attn_mask,
                position_bias=pos_bias
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn, pos_bias


class TransformerSentenceEncoderLayer_1(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        has_relative_attention_bias: bool = False,
        num_buckets: int = 0,
        max_distance: int = 0,
        rescale_init: bool = False,
        gru_rel_pos: bool = False,
        speaker_dim: int = 512,
    ) -> None:
        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_name = activation_fn
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            has_relative_attention_bias=has_relative_attention_bias,
            num_buckets=num_buckets,
            max_distance=max_distance,
            rescale_init=rescale_init,
            gru_rel_pos=gru_rel_pos,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        # self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.self_attn_layer_norm = CondLayerNorm(self.embedding_dim, speaker_dim)

        if self.activation_name == "glu":
            self.fc1 = GLU_Linear(self.embedding_dim, ffn_embedding_dim, "swish")
        else:
            self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        # self.final_layer_norm = LayerNorm(self.embedding_dim)
        self.final_layer_norm = CondLayerNorm(self.embedding_dim, speaker_dim)

    def forward(
        self,
        x: torch.Tensor,
        spk_emb: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        pos_bias=None
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x, spk_emb)
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
                position_bias=pos_bias
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x, spk_emb)
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                attn_mask=self_attn_mask,
                position_bias=pos_bias
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x, spk_emb)

            residual = x
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x, spk_emb)

        return x, attn, pos_bias