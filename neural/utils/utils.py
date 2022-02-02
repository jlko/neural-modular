from pathlib import Path
import os
import hydra

import torch
import torch.nn.functional as F
import collections


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)



def get_root_dir():
    if os.environ.get('SLURM_CLUSTER_NAME') == 'oatcloud':
        root = '/scratch-ssd/jansen/data'
    else:
        try:
            root = hydra.utils.get_original_cwd() + '/data'
        except Exception as e:
            print(e)
            root = '.'

    return Path(root)


def L2_loss(model, weights):

    penalty = 0

    # six different L2 params, one for each resnet block
    for n, p in model.named_parameters():
        if ('conv' not in n) or ('layer' not in n):
            # weirdly the first conv block does not have a 'layer'
            continue
        L = 'layer'
        # get layer number
        layer = int(n[n.find(L) + len(L)])
        # get L2 sum, layers start counting at 1
        penalty += weights[layer-1] * (p**2).sum()

    return penalty


"""Everything below from
https://github.com/wangleiofficial/label-smoothing-pytorch
/blob/main/label_smoothing.py
"""


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingNLLLoss(torch.nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, log_preds, target):
        n = log_preds.size()[-1]
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)
