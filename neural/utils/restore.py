# Restore model from path.
from neural.train import Trainer
from hydra.utils import instantiate

import logging
from pathlib import Path
from omegaconf import OmegaConf
import torch


def restore_trainer(
        model_path: str,
        restore: bool = True,):
    """
    Args:
        model_path: Folder with hydra logs.
        restore: Load ckpt.

    Code example:
        from neural.utils.restore import restore_trainer`
        trainer = restore_trainer('outputs/scratch')

    """

    model_path = Path(model_path)

    cfg = restore_cfg(model_path)

    dataloaders = instantiate(cfg.dataset)
    model = instantiate(cfg.model, data_info=dataloaders['info'])

    trainer = Trainer(model, dataloaders, cfg)

    if restore:
        ckpt_path = model_path / 'model.pth'
        trainer.restore(ckpt_path)
        logging.info(f'Restored weights from {ckpt_path}')

    return trainer


def restore_cfg(model_path: Path):
    cfg_path = model_path / '.hydra' / 'config.yaml'

    return OmegaConf.load(cfg_path)

