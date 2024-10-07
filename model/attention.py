"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import torch
import numpy as np


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, emb_dim, num_heads, use_biases=True, scale_dot_att=True, clip_value=None, extended_version=False):
        super().__init__()
        assert (emb_dim % num_heads == 0)
        self.extended_version = extended_version
        self.emb_size = emb_dim
        self.scale_dot_att = scale_dot_att
        self.num_heads = num_heads
        head_dim = emb_dim // num_heads
        self.head_dim = head_dim
        self.clip_value = clip_value
        self.use_biases = use_biases

        self.lambda_x = torch.nn.Parameter(torch.FloatTensor(num_heads, emb_dim, head_dim))
        self.lambda_y = torch.nn.Parameter(torch.FloatTensor(num_heads, emb_dim, head_dim))
        self.lambda_z1 = torch.nn.Parameter(torch.FloatTensor(num_heads, emb_dim, head_dim))
        self.lambda_z2 = torch.nn.Parameter(torch.FloatTensor(num_heads, emb_dim, head_dim))

        self.bias_lambda_x = torch.nn.Parameter(torch.FloatTensor(num_heads, head_dim)) if use_biases else None
        self.bias_lambda_y = torch.nn.Parameter(torch.FloatTensor(num_heads, head_dim)) if use_biases else None

        self.theta1 = torch.nn.Parameter(torch.FloatTensor(num_heads, emb_dim, head_dim))
        self.theta2 = torch.nn.Parameter(torch.FloatTensor(num_heads, emb_dim, head_dim))
        self.bias_theta = torch.nn.Parameter(torch.FloatTensor(num_heads, head_dim)) if use_biases else None

        self._reset_params()

    def _reset_params(self):
        torch.nn.init.xavier_uniform_(self.lambda_x)
        torch.nn.init.xavier_uniform_(self.lambda_y)
        torch.nn.init.xavier_uniform_(self.lambda_z1)
        torch.nn.init.xavier_uniform_(self.lambda_z2)

        torch.nn.init.zeros_(self.bias_lambda_x)
        torch.nn.init.zeros_(self.bias_lambda_y)

        torch.nn.init.xavier_uniform_(self.theta1)
        torch.nn.init.xavier_uniform_(self.theta2)
        torch.nn.init.zeros_(self.bias_theta)

    def forward(self, x, y, z, mask=None):
        """
            x: <B, N, Q>
            y: <B, M, Q>
            z: <B, N, M, Q> (edge features)
        """

        r_x = torch.einsum("bnq,hqd->bnhd", x, self.lambda_x)
        r_y = torch.einsum("bmq,hqd->bmhd", y, self.lambda_y)
        if z is not None:
            r_z1 = torch.einsum("bnmq,hqd->bnmhd", z, self.lambda_z1)
            r_z2 = torch.einsum("bnmq,hqd->bnmhd", z, self.lambda_z2)

        if self.use_biases:
            r_x = r_x + self.bias_lambda_x
            r_y = r_y + self.bias_lambda_y

        if z is None:
            att_scores = torch.einsum("bnhd,bmhd->bhnm", r_x, r_y)
        else:
            att_scores = torch.einsum("bnmhd,bnmhd->bhnm", r_x[:, :, None, :, :] + r_z1,
                                      r_y[:, None, :, :, :] + r_z2)

        if self.scale_dot_att:
            att_scores *= self.head_dim ** -0.5

        if self.clip_value is not None:
            att_scores = self.clip_value * torch.tanh(att_scores)

        if mask is not None:
            # repeat over num_heads
            mask = mask.unsqueeze(1).repeat(1, att_scores.shape[1], 1, 1)
            att_scores[mask == 1] = -np.inf

        att_weights = torch.softmax(att_scores, dim=-2)

        r = torch.einsum("bhnm,bnq,hqd->bmhd", att_weights, x, self.theta1)
        if self.use_biases:
            r = r + self.bias_theta
        output = torch.einsum("bmhd,hqd->bmq", r, self.theta2)

        return output
