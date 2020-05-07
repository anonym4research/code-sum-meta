# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from src import *

from src.utils.util_file import load_config
from src.dataset.base import *
from src.dataset import *
from src.utils.util_name import time_id, md5_id
from src.model.template import *


def CONST_TIME_CODE(debug):
    return time_id(debug)


CONST_MD5_CODE = md5_id()


def build_model(config: Dict, model: IModel, ) -> Module:
    def _init_param(model):
        for p in model.parameters():
            p.data.uniform_(-config['common']['init_bound'], config['common']['init_bound'])

    def _load_pretrained_weights():
        if config['common']['init_weights'] is None or not os.path.exists(config['common']['init_weights']):
            _init_param(model)
            LOGGER.info('Initialize model weights from scratch.')
        else:
            try:
                # print("init_weights: ", config['common']['init_weights'])
                assert os.path.exists(config['common']['init_weights'])
            except Exception as err:
                LOGGER.error(str(err))
                assert False
            LOGGER.info('Loading from {}'.format(config['common']['init_weights']))
            checkpoint = torch.load(config['common']['init_weights'], map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint)

    _load_pretrained_weights()

    if config['common']['device'] is not None:
        LOGGER.info('Model to GPU')
        model.cuda()
    else:
        LOGGER.info('Model to CPU')
    LOGGER.debug('model structure: {}'.format(str(model)))

    return model


def get_log_filename(args: namedtuple) -> str:
    if args.debug:
        log_filename = '{}_{}_{}_{}_{}.log'.format(
            str.upper(args.task), str.upper(args.lang_mode), str.upper(args.method_name), str.upper(args.train_mode),
            CONST_TIME_CODE(args.debug),
        )
    else:
        log_filename = '{}_{}_{}_{}_{}.{}.log'.format(
            str.upper(args.task), str.upper(args.lang_mode), str.upper(args.method_name), str.upper(args.train_mode),
            CONST_TIME_CODE(args.debug), CONST_MD5_CODE,
        )

    return log_filename


def run_init(yml_filename: str, config=None) -> Dict:
    LOGGER.info("Start {} ... =>  PID: {}".format(sys.argv[0], os.getpid()))

    yaml_file = os.path.join(sys.path[0], yml_filename)
    LOGGER.info('Load arguments in {}'.format(yaml_file))
    if config is None:
        config = load_config(yaml_file)
    else:
        config = load_config(yaml_file, config)

    LOGGER.info('# ------------ env init ------------ #')
    LOGGER.info('Init(seed {}): torch/random/numpy...'.format(config['common']['seed']))
    torch.manual_seed(config['common']['seed'])
    random.seed(config['common']['seed'])
    np.random.seed(config['common']['seed'])
    LOGGER.info("device: {}".format(config['common']['device']))
    if config['common']['device'] is not None:
        LOGGER.debug(config['common']['device'])
        torch.cuda.set_device(config['common']['device'])
    LOGGER.info('Device: {}'.format(torch.cuda.get_device_name()))
    LOGGER.info('# ------------ env init ------------ #')
    return config


def load_config_dataset(args: namedtuple,
                        dataloader: Union[XlangDataloader, UnilangDataloader],
                        base_dataset: Union[sBaseDataset,],
                        collate_fn: Union[sbase_collate_fn,], config=None) -> Tuple:
    if config is None:
        config = run_init(args.yaml)
    else:
        config = run_init(args.yaml, config=config)
    if dataloader == XlangDataloader:
        LOGGER.debug('data_loader: XlangDataloader')

        # because load data is time-consuming, we only load data we need
        src_dataset_info = config['dataset']['source'] \
            if (args.dataset_type == 'source') or (args.dataset_type == 'all') else None
        trg_dataset_info = config['dataset']['target'] \
            if (args.dataset_type == 'target') or (args.dataset_type == 'all') else None

        batch_size = config['training']['batch_size']
        thread_num = config['common']['thread_num']

        if 'leaf_path_k' in config['dataset']:
            leaf_path_k = config['dataset']['leaf_path_k']
        else:
            leaf_path_k = None

        params = {
            'file_dir': config['dataset']['dataset_dir'],
            'code_modalities': config['training']['code_modalities'],
            'token_dicts': config['dicts'],
            'portion': config['dataset']['portion'],
            'leaf_path_k': leaf_path_k,
            'pointer_gen': config['training']['pointer'],
        }

        dataset = dataloader(
            base_dataset, collate_fn, params,
            src_dataset_info, trg_dataset_info, batch_size, thread_num,
        )

    else:
        raise NotImplementedError
    LOGGER.info(dataset)

    config['training']['token_num'] = {key.split('_')[0]: token_size for key, token_size in
                                       dataset.token_dicts.size.items()}
    return config, dataset,


def get_save_dir(config: Dict, args: namedtuple) -> str:
    save_dir = os.path.join(
        config['dataset']['save_dir'],
        '{}_{}_{}.{}'.format(args.task, args.task_mode, CONST_TIME_CODE, CONST_MD5_CODE, ),
    )
    try:
        # os.makedirs(save_dir)
        pass
    except Exception as err:
        print(err)
        '''
        With time-stamp and md5 code, this error almost cannot be raised.
        If you indeed encounter, you are the one. You must win a lottery.
        '''
        LOGGER.error('Dir {} has already existed. Pls, wait and try again to avoid time/md5 conflict.')
    LOGGER.info('save dir: {}'.format(save_dir))
    return save_dir
