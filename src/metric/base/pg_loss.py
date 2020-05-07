# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from src import *
from src.metric import *


class _PGLoss_REINFORCE(Module):
    __slots__ = ('_gather',)

    def __init__(self, gather=True):
        super(_PGLoss_REINFORCE, self).__init__()
        self._gather = gather

    def forward(self, lprobs, target, seq_padding_mask, reward):
        lprobs = lprobs.reshape(-1, lprobs.size(-1))
        target = target.reshape(-1, 1)
        if self._gather:
            logprob_select = torch.gather(lprobs, 1, target)
        else:
            logprob_select = lprobs
        try:
            mask = seq_padding_mask.reshape(-1, 1).bool()
        except:
            mask = seq_padding_mask.reshape(-1, 1).byte()
        out = torch.masked_select(logprob_select, mask)
        reward = torch.masked_select(reward.reshape(-1, 1), mask)
        out = out * reward
        loss = -torch.sum(out) / torch.sum(seq_padding_mask).float()  # get the average loss.
        return loss


class PGLoss_REINFORCE(BaseLoss):

    def __init__(self, device: bool, gather=True) -> None:
        super(PGLoss_REINFORCE, self).__init__(_PGLoss_REINFORCE(gather), device, )
