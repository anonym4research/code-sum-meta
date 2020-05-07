# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from src import *
from src.data import *

import ujson

from joblib import Parallel, delayed
from multiprocessing import cpu_count
from src.dataset.base import *
from src.utils.util_file import *
from src.utils.util import *
from src.utils.util_data import *
from src.dataset.unilang_dataloader import UnilangDataloader


class XlangDataloader:
    __slots__ = ('src_datasets', 'trg_datasets', 'token_dicts',)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._construct(*args, **kwargs)

    def _construct(self,
                   # base parameters
                   base_dataset: Union[sBaseDataset,],
                   collate_fn: Union[sbase_collate_fn,],
                   params: Dict,

                   # loader parameters
                   src_dataset_info: Union[None, Dict], trg_dataset_info: Union[None, Dict],
                   batch_size: int,
                   thread_num: int,
                   ) -> None:
        self.token_dicts = TokenDicts(params.get('token_dicts'))
        params['token_dicts'] = self.token_dicts

        # source dataset
        if src_dataset_info is not None:
            # TimeAnalyzer(self._load_dataset)(
            #     base_dataset, collate_fn, file_dir, code_modalities, src_dataset_info,
            #     batch_size, token_dicts, thread_num,
            #     portion, merge_xlng, leaf_path_k,
            # )
            # TimeAnalyzer.print_stats()
            LOGGER.info("load src_datasets")
            self.src_datasets = self._load_dataset(
                base_dataset, collate_fn, params,
                src_dataset_info, batch_size, thread_num,
            )
        else:
            self.src_datasets = None
        # target dataset
        if trg_dataset_info is not None:
            LOGGER.info("load trg_datasets")
            self.trg_datasets = self._load_dataset(
                base_dataset, collate_fn, params,
                trg_dataset_info, batch_size, thread_num,
            )
        else:
            self.trg_datasets = None

    def _load_dataset(self,
                      base_dataset: Union[sBaseDataset,],
                      collate_fn: Union[sbase_collate_fn,],
                      params: Dict,

                      # parameters
                      dataset_info: Dict,
                      batch_size: int, thread_num: int,
                      ) -> Dict:
        data_lngs, modes = dataset_info['dataset_lng'], dataset_info['mode']
        dataset_dict = {}

        # 53.6876
        paralleler = Parallel(len(data_lngs))
        datasets = paralleler(delayed(UnilangDataloader)
                              (base_dataset, collate_fn, dict(params, **{'data_lng': lng}), batch_size, modes,
                               thread_num, )
                              for lng in data_lngs)
        for lng, ds in zip(data_lngs, datasets):
            assert lng == ds.lng
            dataset_dict[lng] = ds

        return dataset_dict

    def __getitem__(self, key: str) -> Any:
        if key in ['src', 'source', ]:
            return self.src_datasets
        elif key in ['trg', 'target', ]:
            return self.trg_datasets
        else:
            raise NotImplementedError

    @property
    def size(self):
        return {
            'src': {lng: unilng_dataset.size for lng, unilng_dataset in self.src_datasets.items()},
            'trg': {lng: unilng_dataset.size for lng, unilng_dataset in self.trg_datasets.items()},
        }

    def __str__(self):
        return 'TLDataloader(\nsrc dataset: {}\ntrg dataset: {}\ntoken_dicts: {}\n)'.format(
            self.src_datasets, self.trg_datasets, self.token_dicts
        )

    def __repr__(self):
        return str(self)
