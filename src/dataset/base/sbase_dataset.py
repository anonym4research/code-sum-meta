# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from src import *
from src.data import *

import glob
import random
import ujson
import math

from src.utils.util import *
from src.utils.util_data import *
from src.utils.util_file import *
from src.data import _Dict
from torch.utils.data import Dataset


def load_file(filename: str) -> List:
    with open(filename, 'r') as reader:
        return [ujson.loads(line.strip()) for line in reader.readlines() if len(line.strip()) > 0]


def load_data(dataset_files: Dict, mode: str) -> Dict:
    data = {key: [] for key in dataset_files.keys()}
    for key, filenames in dataset_files.items():
        for fl in filenames:
            LOGGER.debug("mode:{} keys: {}/{} file: {}/{}  ".format(mode, key, dataset_files.keys(),
                                                                    filenames.index(fl), len(filenames)))
            data[key].extend(load_file(fl))
    return data


class sBaseDataset(Dataset):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._construct(*args, **kwargs)

    @classmethod
    def multiprocessing(cls, **params: Dict, ) -> Any:
        instance = cls(
            file_dir=params.get('file_dir'),
            data_lng=params.get('data_lng'),
            code_modalities=params.get('code_modalities'),
            mode=params.get('mode'),
            token_dicts=params.get('token_dicts'),
            portion=params.get('portion', None),  # for maml
            leaf_path_k=params.get('leaf_path_k', None),  # for path
            pointer_gen=params.get('pointer_gen', False),  # pointer_gen for decoder
        )
        return instance

    def _construct(self, file_dir: str, data_lng: str, code_modalities: List[str], mode: str, token_dicts: TokenDicts,
                   portion=None, leaf_path_k=None,  # for path
                   pointer_gen=False,  # pointer_gen for decoder
                   ) -> None:
        # load data and save data info
        self.mode = mode.lower()
        self.lng = data_lng.lower()
        self.code_modalities = code_modalities
        self.pointer_gen = pointer_gen
        # LOGGER.debug(self.code_modalities)
        assert self.mode in MODES, \
            Exception('{} mode is not in {}'.format(self.mode, MODES, ))

        LOGGER.debug('read sbase dataset loader')
        load_keys = self.code_modalities + ['index', 'comment', ]
        # if test, load [code/docstring] as case-study info
        if self.mode == 'test':
            load_keys.extend(['code', 'docstring'])
        LOGGER.debug('{} load {}'.format(self.mode, load_keys))
        self.dataset_files = {
            key: sorted([filename for filename in
                         glob.glob('{}/*'.format(os.path.join(file_dir, data_lng, key)))
                         if self.mode in filename])
            for key in load_keys
        }

        self.data = load_data(self.dataset_files, mode=self.mode)
        # check all data are same size
        size = len(self.data['comment'])
        for key, value in self.data.items():
            assert size == len(value), Exception('{} data: {}, but others: {}'.format(key, len(value), size))

        if self.mode == 'test':
            self.data['case_study'] = list(zip(self.data['code'], self.data['docstring']))
            self.data.pop('code')
            self.data.pop('docstring')
        # sample portion
        # LOGGER.debug(portion)
        if self.mode == 'train' and self.lng.lower() == 'ruby':
            if portion is None or portion == 1.0:
                self.LENGTH = len(self.data['index'])
            elif 0.0 < portion < 1.0:
                self.LENGTH = max(int(len(self.data['index']) * portion), 1)  # at least 1 data
                random_indices = random.sample(range(len(self.data['index'])), self.LENGTH)
                for key, value in self.data.items():
                    self.data[key] = [value[ind] for ind in random_indices]
            else:
                raise NotImplementedError('portion can only be (0, 1]')
            # load dataset files
            LOGGER.info("load dataset: {}-{}-p{}".format(self.lng, self.mode, portion))
        else:
            self.LENGTH = len(self.data['index'])
            # load dataset files
            LOGGER.info("load dataset: {}-{}".format(self.lng, self.mode))

        if 'tok' in self.code_modalities:
            self.data['code'] = self.data['tok']
            code_key = 'tok'

        if code_key is not None:
            if self.pointer_gen:
                # oov
                # parse code for oov
                self.data['code_dict_comment'] = [None] * len(self.data['code'])
                self.data['code_oovs'] = [None] * len(self.data['code'])
                for ind, code in enumerate(self.data['code']):
                    extend_vocab, oovs = extend_dict(code, token_dicts['comment'])
                    self.data['code_dict_comment'][ind] = extend_vocab
                    self.data['code_oovs'][ind] = oovs
                    self.data['code'][ind] = token_dicts[code_key].to_indices(code, UNK_WORD)
            else:
                for ind, code in enumerate(self.data['code']):
                    self.data['code'][ind] = token_dicts[code_key].to_indices(code, UNK_WORD)

        # ast path mmodality
        if 'path' in self.code_modalities:
            for ind, path in enumerate(self.data['path']):
                if leaf_path_k is None:
                    head, center, tail = zip(*path)
                else:
                    head, center, tail = zip(*path[:leaf_path_k])
                self.data['path'][ind] = [
                    [token_dicts['border'].to_indices(subtoken, UNK_WORD) for subtoken in head],
                    [token_dicts['center'].to_indices(subtoken, UNK_WORD) for subtoken in center],
                    [token_dicts['border'].to_indices(subtoken, UNK_WORD) for subtoken in tail],
                ]
        else:
            self.data['path'] = None

        # comment
        self.data['raw_comment'] = deepcopy(self.data['comment'])  # time-consuming, if not necessary, pls delete
        if self.pointer_gen and 'code' in self.data:
            # pointer
            self.data['comment_extend_vocab'] = [None] * len(self.data['comment'])
            for ind, comment in enumerate(self.data['comment']):
                extend_vocab = extend_dict_with_oovs(comment, token_dicts['comment'], self.data['code_oovs'][ind])
                self.data['comment_extend_vocab'][ind] = extend_vocab
                self.data['comment'][ind] = token_dicts['comment'].to_indices(comment, UNK_WORD)
        else:
            for ind, comment in enumerate(self.data['comment']):
                self.data['comment'][ind] = token_dicts['comment'].to_indices(comment, UNK_WORD)

    @property
    def size(self):
        return self.LENGTH

    def __len__(self):
        return self.LENGTH

    def __getitem__(self, index: int) -> Dict:
        raw_comment = self.data['raw_comment'][index]
        comment = self.data['comment'][index]
        if self.pointer_gen and 'code' in self.data:
            comment_extend_vocab = self.data['comment_extend_vocab'][index]
        else:
            comment_extend_vocab = None
        # case study
        case_study_data = self.data['case_study'][index] if 'case_study' in self.data else None

        sample = {}
        if 'code' in self.data:
            # tok modal
            code = self.data['code'][index]
            if self.pointer_gen:
                code_dict_comment = self.data['code_dict_comment'][index]
                code_oovs = self.data['code_oovs'][index]
            else:
                code_dict_comment, code_oovs = None, None
            sample['tok'] = [code, code_dict_comment, code_oovs, ]
        if 'path' in self.code_modalities:
            # ast path
            sample['path'] = self.data['path'][index]

        sample['others'] = [comment, comment_extend_vocab, raw_comment,
                            case_study_data, index, self.code_modalities, self.pointer_gen, ]

        return sample


def sbase_collate_fn(batch_data: List) -> Dict:
    code_modalities, pointer_gen = batch_data[0]['others'][-2], batch_data[0]['others'][-1]
    batch_size = len(batch_data)

    # Build and return our designed batch (dict)
    store_batch = {}

    # release comment first for copy-generator
    comment, comment_extend_vocab, raw_comment, case_study_data, index, _, _ = \
        zip(*[batch['others'] for batch in batch_data])
    comment, comment_input, comment_target, comment_len = pad_comment_sum(comment)
    if comment_extend_vocab[0] is not None:  # tuple of None
        _, _, comment_extend_vocab, _ = pad_comment_sum(comment_extend_vocab)
    else:
        comment_extend_vocab = None
    # comment to tensor
    comment, comment_input, comment_target, comment_len, comment_extend_vocab, \
        = map(to_torch_long, (comment, comment_input, comment_target, comment_len, comment_extend_vocab,))
    # feed comment
    store_batch['comment'] = [comment, comment_input, comment_target, comment_len, raw_comment]

    if 'tok' in batch_data[0]:
        # release tok modal
        code, code_dict_comment, code_oovs = zip(*[batch['tok'] for batch in batch_data])
        code, code_len, code_mask = pad_seq(code, include_padding_mask=True)  # pad code
        code, code_len, code_mask = to_torch_long(code), to_torch_long(code_len), \
                                    torch.from_numpy(code_mask).float()
        store_batch['tok'] = [code, code_len, code_mask]
        if pointer_gen:
            # for pointer
            pointer_extra_zeros = torch.zeros(batch_size, max([len(x) for x in code_oovs]))
            code_dict_comment, _ = pad_seq(code_dict_comment)
            code_dict_comment = to_torch_long(code_dict_comment)
            store_batch['pointer'] = [code_dict_comment, comment_extend_vocab, pointer_extra_zeros, code_oovs]
        else:
            store_batch['pointer'] = [None, None, None, None]

    if 'path' in code_modalities:
        # release ast-path modal
        head, center, tail = map(
            lambda batch_list: list(itertools.chain(*batch_list)),
            zip(*[batch['path'] for batch in batch_data]),
        )
        head, head_len, head_mask = pad_seq(head, include_padding_mask=True)
        head, head_len, head_mask = to_torch_long(head), to_torch_long(head_len), \
                                    to_torch_long(head_mask)

        center, center_len, center_mask = pad_seq(center, include_padding_mask=True)
        center, center_len, center_mask = to_torch_long(center), to_torch_long(center_len), \
                                          to_torch_long(center_mask)

        tail, tail_len, tail_mask = pad_seq(tail, include_padding_mask=True)
        tail, tail_len, tail_mask = to_torch_long(tail), to_torch_long(tail_len), \
                                    to_torch_long(tail_mask)

        store_batch['path'] = [
            head, head_len, head_mask,
            center, center_len, center_mask,
            tail, tail_len, tail_mask,
        ]

    # other info
    store_batch['index'] = torch.Tensor(index).long()
    store_batch['case_study'] = case_study_data
    return store_batch
