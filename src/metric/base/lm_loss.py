# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from src import *
from src.metric import *
from src.utils.constants import PAD


class _LMLoss(Module):
    '''
    cross-entropy loss
    '''

    __slots__ = ('_gather',)

    def __init__(self, gather=True) -> None:
        super(_LMLoss, self).__init__()
        self._gather = gather

    def forward(self, log_probs: torch.Tensor, target: torch.Tensor, ) -> torch.Tensor:
        '''
        :param log_probs: [batch_size, seq_len, probability_size]
        :param target: [batch_size, seq_len]
        '''
        log_probs = log_probs.reshape(-1, log_probs.size(-1))
        target = target.reshape(-1, 1)
        if self._gather:

            log_probs_selected = torch.gather(log_probs, 1, target)
        else:
            log_probs_selected = log_probs
        mask = target.data.gt(0)  # generate the mask
        out = torch.masked_select(log_probs_selected, mask)
        loss = -torch.sum(out) / torch.sum(target.data.ne(PAD)).float()  # get the average loss.
        return loss


class LMLoss(BaseLoss):

    def __init__(self, device: bool, gather=True) -> None:
        super(LMLoss, self).__init__(_LMLoss(gather), device, )


