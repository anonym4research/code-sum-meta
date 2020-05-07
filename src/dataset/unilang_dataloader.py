# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from src import *

from joblib import Parallel, delayed
from multiprocessing import cpu_count

from multiprocessing import Pool
from src.data import *
from src.dataset.base import *
from src.utils.util import *
from src.utils.util_data import *
from src.utils.util_file import *


class UnilangDataloader(object):
    __slots__ = ('batch_size', 'modes', 'lng', 'token_dicts', 'data_loaders', 'LENGTH',)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._construct(*args, **kwargs)

    def _construct(self,
                   # base parameters
                   base_dataset: Union[sBaseDataset,],
                   collate_fn: Union[sbase_collate_fn,],
                   params: Dict,

                   # loader parameters
                   batch_size: int,
                   modes: List[str],
                   thread_num: int,
                   ) -> None:
        self.batch_size = batch_size
        self.modes = modes

        self.lng = params.get('data_lng')
        self.LENGTH = {}
        self.data_loaders = {}
        self._load_dict(params.get('token_dicts'))
        params['token_dicts'] = self.token_dicts
        self._load_data(thread_num, base_dataset, collate_fn, params)

    def _load_data(self,
                   thread_num: int,
                   base_dataset: Union[sBaseDataset,],
                   collate_fn: Union[sbase_collate_fn,],
                   params: Dict,
                   ) -> None:
        # 18.2358
        paralleler = Parallel(len(self.modes))
        datasets = paralleler(delayed(base_dataset)(**dict(params, **{'mode': mode})) for mode in self.modes)
        for mode, ds in zip(self.modes, datasets):
            assert mode == ds.mode
            self.LENGTH[mode] = ds.size
            data_loader = DataLoader(
                ds, batch_size=self.batch_size,
                shuffle=False if mode == 'test' else True,
                collate_fn=collate_fn,
                num_workers=thread_num,
            )
            self.data_loaders[mode] = data_loader

    def _load_dict(self, token_dicts: Union[TokenDicts, Dict], ) -> None:
        if isinstance(token_dicts, dict):
            self.token_dicts = TokenDicts(token_dicts)
        elif isinstance(token_dicts, TokenDicts):
            self.token_dicts = token_dicts
        else:
            raise NotImplementedError('{}} token_dicts is wrong'.format(self.__class__.__name__))

    def __getitem__(self, key: str) -> Any:
        return self.data_loaders[key]

    @property
    def size(self, ) -> Dict:
        return self.LENGTH

    def __repr__(self):
        return str(self)

    def __str__(self):
        return '{}: {} - {}, batch_size({})'.format(
            self.__class__.__name__, self.lng, self.LENGTH, self.batch_size
        )
