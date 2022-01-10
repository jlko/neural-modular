# Restore model from path.

from ensembles.train import Trainer
from ensembles.utils import maps

import logging
from pathlib import Path
import omegaconf
from omegaconf import OmegaConf

import torch


def make_path(path):
    if not isinstance(path, Path):
        path = Path(path)
    return path


def restore_trainer(
        model_path: str,
        restore: bool = True,
        eval_train: bool = True,
        debug: bool = False,
        data_loader_kwargs: dict = None,
        ):
    """
    Args:
        model_path: Folder with hydra logs.
        restore: Load ckpt.
        eval_train: Evaluate loss on training set. This loads a training set
            with test set loader configs (disabled augmentations and larger
            batch size).
        debug: Load smaller train set for faster debugging.
    """

    model_path = make_path(model_path)

    cfg = restore_cfg(model_path)

    # ugly hotfix
    cfg.dataset.test_batch_size = 2

    kwargs = dict(train_eval=eval_train, debug=debug, n_eval_train=10000)
    dataloaders = restore_dataloaders(model_path, cfg, data_loader_kwargs)
    model = restore_model(cfg)

    trainer = Trainer(model, dataloaders, cfg)

    if restore:
        ckpt_path = model_path / 'model.pth'
        trainer.restore(ckpt_path)
        logging.info(f'Restored weights from {ckpt_path}')

    return trainer


def restore_cfg(model_path: Path):
    cfg_path = model_path / '.hydra' / 'config.yaml'

    return OmegaConf.load(cfg_path)


def restore_dataloaders(
        model_path: Path,
        cfg: omegaconf.dictconfig.DictConfig,
        data_loader_kwargs: dict,
        **kwargs):

    return maps.dataset[cfg.dataset.name](
        cfg.dataset, data_loader_kwargs=data_loader_kwargs, **kwargs)


def restore_model(
        cfg: omegaconf.dictconfig.DictConfig):

    model = maps.model[cfg.model.name](cfg)

    return model
