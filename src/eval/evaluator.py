# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from src import *
from src.data import *
from src.metric import *
from src.model.template import *
from src.utils.util_data import batch_to_cuda
from src.utils.util_eval import eval_metrics
from src.metric.base import *
from src.utils.util import save_json
from tabulate import tabulate
from src.utils.constants import METRICS


def load_data(model, datatype):
    import glob, ujson
    ast_files = sorted([filename for filename in glob.glob('{}/*'.format(os.path.join(
        model.config['dataset']['dataset_dir'], 'ruby', datatype
    ))) if 'test' in filename])

    len_list = []
    for fl in ast_files:
        with open(fl, 'r') as reader:
            line = reader.readline().strip()
            while len(line) > 0:
                line = ujson.loads(line)
                len_list.append(len(line))
                line = reader.readline().strip()
    return len_list


class Evaluator(object):
    def __init__(self, ) -> None:
        pass

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def summarization_eval(model: IModel, data_loader: DataLoader, token_dicts: TokenDicts, criterion: BaseLoss,
                           collate_func=None, model_filename=None, metrics=None) -> Any:

        with torch.no_grad():
            model.eval()
            data_iter = iter(data_loader)  # init

            total_loss = 0.0
            src_comment_all, trg_comment_all, pred_comment_all, src_code_all = \
                [], [], [], []
            if model.config['training']['pointer']:
                oov_vocab = []
            else:
                oov_vocab = None

            for iteration in range(1, 1 + len(data_loader)):  # 1 + len(data_loader)
                batch = data_iter.__next__()
                if collate_func is not None:
                    batch = collate_func(batch)
                if model.config['common']['device'] is not None:
                    batch = batch_to_cuda(batch)
                comment_pred, comment_logprobs, = model.eval_pipeline(batch)
                if model_filename is None:
                    src_comment_all.extend([None] * model.config['training']['batch_size'])
                    src_code_all.extend([None] * model.config['training']['batch_size'])
                else:
                    src_codes, src_comments, = zip(*batch['case_study'])
                    src_comment_all.extend(src_comments)
                    src_code_all.extend(src_codes)

                # comment
                trg_comment_all.extend(batch['comment'][4])
                pred_comment_all.extend(comment_pred)
                # oovs
                if model.config['training']['pointer']:
                    oov_vocab.extend(batch['pointer'][-1])

                # print(comment_logprobs.size())
                # print(comment_target_padded.size())
                if model.config['training']['pointer']:
                    comment_target = batch['pointer'][1][:, :model.config['training']['max_predict_length']]
                else:
                    comment_target = batch['comment'][2][:, :model.config['training']['max_predict_length']]
                # print('comment_logprobs: ', comment_logprobs.size())
                # print('comment_target: ', comment_target.size())
                comment_loss = criterion(comment_logprobs[:, :comment_target.size(1)], comment_target)
                total_loss += comment_loss.item()
            total_loss /= len(data_loader)
            LOGGER.info('Summarization test loss: {:.4}'.format(total_loss))

            if model_filename is None:
                pred_filename = None
            else:
                pred_filename = model_filename.replace('.pt', '.pred')

            bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, \
            cider, srcs_return, trgs_return, preds_return, = \
                eval_metrics(src_comment_all, trg_comment_all, pred_comment_all, src_code_all,
                             oov_vocab, token_dicts, pred_filename,
                             metrics=model.config['testing']['metrics'] if metrics is None else metrics, )
            bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider, = \
                map(lambda array: sum(array) / len(array), (bleu1, bleu2, bleu3, bleu4, meteor,
                                                            rouge1, rouge2, rouge3, rouge4, rougel, cider,))

            return bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider
