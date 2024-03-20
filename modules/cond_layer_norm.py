# """
# Adapt from https://github.com/auspicious3000/contentvec/blob/main/contentvec/modules/cond_layer_norm.py
# """
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Module
from torch.nn import Linear

# class CondLayerNorm(Module):
#
#     def __init__(self, dim_last, eps=1e-5, dim_spk=256, elementwise_affine=True):
#         super(CondLayerNorm, self).__init__()
#         self.dim_last = dim_last
#         self.eps = eps
#         self.dim_spk = dim_spk
#         self.elementwise_affine = elementwise_affine
#         if self.elementwise_affine:
#             self.weight_ln = Linear(self.dim_spk,
#                                     self.dim_last,
#                                     bias=False)
#             self.bias_ln = Linear(self.dim_spk,
#                                   self.dim_last,
#                                   bias=False)
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         if self.elementwise_affine:
#             init.ones_(self.weight_ln.weight)
#             init.zeros_(self.bias_ln.weight)
#
#     def forward(self, input, spk_emb):
#         weight = self.weight_ln(spk_emb)
#         bias = self.bias_ln(spk_emb)
#         return F.layer_norm(
#             input, input.size()[1:], weight, bias, self.eps)
#
#     def extra_repr(self):
#         return '{dim_last}, eps={eps}, ' \
#             'elementwise_affine={elementwise_affine}'.format(**self.__dict__)

import torch
import torch.nn as nn
import numbers

class CondLayerNorm(nn.Module):
    def __init__(self, normalized_shape, embed_dim=192, modulate_bias=False, eps=1e-5):
        super(CondLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)

        self.embed_dim = embed_dim
        self.eps = eps

        self.weight = nn.Parameter(torch.empty(*normalized_shape))
        self.bias = nn.Parameter(torch.empty(*normalized_shape))
        assert len(normalized_shape) == 1
        self.ln_weight_modulation = FiLM(normalized_shape[0], embed_dim)
        self.modulate_bias = modulate_bias
        if self.modulate_bias:
            self.ln_bias_modulation = FiLM(normalized_shape[0], embed_dim)
        else:
            self.ln_bias_modulation = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input, embed):
        mean = torch.mean(input, -1, keepdim=True)
        var = torch.var(input, -1, unbiased=False, keepdim=True)
        if embed is None:
            res = (input - mean) / torch.sqrt(var + self.eps)
        else:
            weight = self.ln_weight_modulation(embed, self.weight.expand(embed.size(0), -1))
            if self.ln_bias_modulation is None:
                bias = self.bias
            else:
                bias = self.ln_bias_modulation(embed, self.bias.expand(embed.size(0), -1))
            res = (input - mean) / torch.sqrt(var + self.eps) * weight + bias
        return res

    def extra_repr(self):
        return '{normalized_shape}, {embed_dim}, modulate_bias={modulate_bias}, eps={eps}'.format(**self.__dict__)

class FiLM(nn.Module):
    """ Feature-wise Linear Modulation (FiLM) layer"""
    def __init__(self, feat_size, embed_size, num_film_layers=1, layer_norm=False):
        super(FiLM, self).__init__()
        self.feat_size = feat_size
        self.embed_size = embed_size
        self.num_film_layers = num_film_layers
        self.layer_norm = nn.LayerNorm(output_size) if layer_norm else None
        gamma_fcs, beta_fcs = [], []
        for i in range(num_film_layers):
            if i == 0:
                gamma_fcs.append(nn.Linear(embed_size, feat_size))
                beta_fcs.append(nn.Linear(embed_size, feat_size))
            else:
                gamma_fcs.append(nn.Linear(feat_size, feat_size))
                beta_fcs.append(nn.Linear(feat_size, feat_size))
        self.gamma_fcs = nn.ModuleList(gamma_fcs)
        self.beta_fcs = nn.ModuleList(beta_fcs)
        self.init_weights()

    def init_weights(self):
        for i in range(self.num_film_layers):
            nn.init.zeros_(self.gamma_fcs[i].weight)
            nn.init.zeros_(self.gamma_fcs[i].bias)
            nn.init.zeros_(self.beta_fcs[i].weight)
            nn.init.zeros_(self.beta_fcs[i].bias)

    def forward(self, embed, x):
        gamma, beta = None, None
        for i in range(len(self.gamma_fcs)):
            if i == 0:
                gamma = self.gamma_fcs[i](embed)
                beta = self.beta_fcs[i](embed)
            else:
                gamma = self.gamma_fcs[i](gamma)
                beta = self.beta_fcs[i](beta)
        gamma = gamma.expand_as(x)
        beta = beta.expand_as(x)
        x = (1 + gamma) * x + beta
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x

