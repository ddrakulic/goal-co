"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import torch


class ResidualNorm(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(0.))  # 'ReZero' https://arxiv.org/abs/2003.04887)

    def forward(self, state_before, modified_after):
        # residual connection
        state_before = state_before + self.alpha * modified_after
        # batch/layer norm
        return state_before

