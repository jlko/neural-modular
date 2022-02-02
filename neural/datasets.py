"""Datasets for active testing."""
from cgi import test
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


def CIFAR10(cfg):

    dataset = datasets.CIFAR10
    name = 'CIFAR10'

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4739, 0.4739, 0.4739), (0.2517, 0.2517, 0.2517)),
            # (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4739, 0.4739, 0.4739), (0.2517, 0.2517, 0.2517)),
        #     (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    return get_pytorch_dataset(
        cfg, dataset, name, train_transform, test_transform)


def FashionMNIST(cfg):

    dataset = datasets.FashionMNIST
    name = 'FashionMNIST'

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.28652713), (0.3532744)),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.28652713), (0.3532744)),
        ])

    return get_pytorch_dataset(
        cfg, dataset, name, train_transform, test_transform)


def get_pytorch_dataset(cfg, dataset, name, train_transform, test_transform):

    root = get_root_dir()
    path = root / name
    logging.info(f'{name} dir is {root}.')

    # ** Load as Datasets **
    train_data = dataset(
        path, train=True, transform=train_transform, download=True)
    val_data = dataset(
        path, train=True, transform=test_transform, download=False)
    test_data = dataset(
        path, train=False, transform=test_transform, download=False)

    # Cut down to relevant subsets
    N = train_data.data.shape[0]
    V = cfg.val_size

    train_data = Subset(train_data, range(0, N - V))
    val_data = Subset(val_data, range(N - V, N))

    Bt = cfg.train_batch_size
    Be = cfg.test_batch_size
    # ** Create Dataloaders **
    kwargs = dict(
        num_workers=4,
        pin_memory=True,
        persistent_workers=True)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=Bt, shuffle=True, **kwargs)

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=Be, shuffle=False, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=Be, shuffle=False, **kwargs)

    D_in = train_data.dataset.data.shape[1:]
    D_out = np.size(np.unique(train_data.dataset.targets))

    out = dict(
        loaders=dict(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader),
        info=dict(
            N=N, D_in=D_in, D_out=D_out,
            type='classification')
    )

    return out
