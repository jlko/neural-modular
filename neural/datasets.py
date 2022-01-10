"""Datasets for active testing."""
import os
import pickle
import logging
import hydra
from pathlib import Path

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms, datasets
from torch.utils.data import Subset

from .utils import get_root_dir


def RandomLGSSM(cfg, seed):
    """From pruning code. Only used for debugging purposes."""

    # hydra.utils.get_original_cwd()

    # data/LGSSM
    pwd = hydra.utils.get_original_cwd()

    fn = 'random'
    fn = fn + f'_N={cfg.N}'
    fn = fn + f'_T={cfg.n_steps}'
    fn = fn + f'_zD={cfg.z_dim}'
    fn = fn + f'_yD={cfg.y_dim}'
    fn = fn + f'_noise={cfg.noise_scale}'
    fn = fn + f'_filter_tails={cfg.filter_tails}'
    fn = fn + '.pkl'
    filename = fn

    datapath = Path(pwd) / 'data/LGSSM' / filename

    logging.info(f'Loading RandomLGSSm from {datapath}.')

    # ** Load as Datasets **
    with open(datapath, 'rb') as file:
        dataset = pickle.load(file)

    # extract data for training a model
    # targets = np.log(-np.array(dataset['log_liks']))
    # don't need log anymore b/c distribution is now nice
    targets = np.array(dataset['log_liks'])
    if targets.ndim == 1:
        targets = targets[:, np.newaxis]

    # filter out Nones
    params = [
        [attribute for attribute in entry if attribute is not None]
        for entry in dataset['params']]

    # concatenate
    params = [np.concatenate(
        [np.reshape(attribute, -1) for attribute in entry])
        for entry in params]

    params = np.stack(params)

    ys = [np.concatenate(
        [np.reshape(attribute, -1) for attribute in entry])
        for entry in dataset['ys']]
    ys = np.stack(ys)

    # I currently do not look at these
    if cfg.observe_z:
        zs = [np.concatenate(
            [np.reshape(attribute, -1) for attribute in entry])
            for entry in dataset['zs']]
        zs = np.stack(zs)
        inputs = np.concatenate([
            params, ys, zs], 1).astype('float')

    else:
        inputs = np.concatenate([
            params, ys], 1).astype('float')

    N, D_in = inputs.shape
    N_out, D_out = targets.shape

    assert N == N_out

    test_size = 0.2
    val_size = 0.1

    train_end = round(N * (1-test_size-val_size))
    val_end = train_end + round(val_size * N)

    RS = np.random.default_rng(seed=seed)
    idxs = RS.permutation(N)
    # idxs = np.arange(0, N)
    train_idxs = idxs[:train_end]
    val_idxs = idxs[train_end:val_end]
    test_idxs = idxs[val_end:]

    train_inputs, train_targets = inputs[train_idxs], targets[train_idxs]
    val_inputs, val_targets = inputs[val_idxs], targets[val_idxs]
    test_inputs, test_targets = inputs[test_idxs], targets[test_idxs]

    # standardisation for targets
    def mse(a, b):
        return np.mean((np.mean(a) - b)**2)
    def log_random_perf(add):
        logging.info(
            f'MSE when guessing on {add}standardised data \n'
            f'\t train: {mse(train_targets, train_targets):.4f}\n'
            f'\t val: {mse(train_targets, test_targets):.4f}\n'
            f'\t test: {mse(train_targets, test_targets):.4f}\n')

    log_random_perf(add='un')
    ttm, tts = train_targets.mean(0), train_targets.std(0)
    train_targets = (train_targets - ttm) / tts
    val_targets = (val_targets - ttm) / tts
    test_targets = (test_targets - ttm) / tts
    log_random_perf(add='')

    # standardisation for inputs
    tim, tis = train_inputs.mean(0), train_inputs.std(0)
    # let's not divide by 0
    tis[np.isclose(tis, 0)] = 1

    train_inputs = (train_inputs - tim) / tis
    val_inputs = (val_inputs - tim) / tis
    test_inputs = (test_inputs - tim) / tis


    # TODO add dim-wise standardisation of inputs *and* targets
    def get_dataset(arrs):
        arrs = [torch.from_numpy(arr) for arr in arrs]
        dataset = torch.utils.data.TensorDataset(*arrs)
        return dataset

    train_data = get_dataset([train_inputs, train_targets])
    val_data = get_dataset([val_inputs, val_targets])
    test_data = get_dataset([test_inputs, test_targets])

    kwargs = {"num_workers": 4, "pin_memory": True, "persistent_workers": True}

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=128, shuffle=True, **kwargs)

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=1000, shuffle=False, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=1000, shuffle=False, **kwargs)

    out = dict(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        info=dict(N=N, D_in=D_in, D_out=D_out)
    )

    return out
