"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import torch
from torch import nn
from torch.nn import Linear
from model.attention import MultiHeadAttention
from model.residual import ResidualNorm


class Layer(nn.Module):
    def __init__(self, dim_emb, dim_ff, num_heads, attention_clip_value, activation_ff):
        super(Layer, self).__init__()

        self.attn = MultiHeadAttention(dim_emb, num_heads, clip_value=attention_clip_value)

        self.linear1 = Linear(dim_emb, dim_ff)
        self.linear2 = Linear(dim_ff, dim_emb)

        assert activation_ff == "relu" or activation_ff == "gelu"

        if activation_ff == "relu":
            self.activation = torch.nn.ReLU()
        else:
            self.activation = torch.nn.GELU()

        self.res_norm1 = ResidualNorm()
        self.res_norm2 = ResidualNorm()

    def forward(self, state, edge_embs, mask=None, multitype=False, seq_len_per_type=None):
        # state: list of states     [s1, s2]
        # edge_features             [ef1, ef2, ef3, ef4]  (some of edge_features can be None)
        # computation flow is :
        # s1 = self_att(s1, ef1)
        # s2 = self_att(s2, ef2)
        # s1 = s1 + cross_att(s2, s1, ef3)
        # s2 = s2 + cross_att(s1, s2, ef4)

        if not multitype:
            state_rc = state
            # self attention
            state = self.attn(state, state, edge_embs, mask)
            # residual + norm
            state = self.res_norm1(state_rc, state)
            state_rc = state
            # FF
            state = self.linear2(self.activation(self.linear1(state)))
            # residual + norm
            state = self.res_norm2(state_rc, state)
        else:
            if edge_embs is None:
                edge_embs = [None, None, None, None]
            state_rc = state
            nb_of_type1, nb_of_type2 = seq_len_per_type
            input1, input2 = state[:, :nb_of_type1], state[:, nb_of_type1:]

            # self attention of first node type
            state1 = self.attn(input1, input1, edge_embs[0], mask)
            # self attention of second node type
            state2 = self.attn(input2, input2, edge_embs[1], mask)

            # first cross attention
            state1 = state1 + self.attn(input2, input1, edge_embs[2], mask)
            # second cross attention
            state2 = state2 + self.attn(input1, input2, edge_embs[3], mask)

            state = torch.cat([state1, state2], dim=1)
            state = self.res_norm1(state_rc, state)

            state_before = state
            # FF
            state = self.linear2(self.activation(self.linear1(state)))
            # residual + norm
            state = self.res_norm2(state_before, state)

        return state
