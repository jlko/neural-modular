"""Main active testing loop."""
import os
import logging
import hydra
from hydra.utils import instantiate
import warnings

import numpy as np
import torch

from omegaconf import OmegaConf

from neural.utils import maps, flatten_dict
from neural.train import Trainer

@hydra.main(config_path='conf', config_name='config')
def main(cfg):
    """Train single model, evaluate on test data, save predictions and model.
    """

    if cfg.wandb.use_wandb:
        import wandb
        wandb_cfg = flatten_dict(dict(cfg))

        wandb_args = dict(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=wandb_cfg)
        wandb.init(**wandb_args)

    seed = cfg.experiment.random_seed
    if seed == -1:
        seed = np.random.randint(0, 1000)
    torch.torch.manual_seed(seed)
    np.random.seed(seed)

    logging.info(f'Setting random seed to {seed}.')
    logging.info(f'Logging to {os.getcwd()}.')

    if cuda := torch.cuda.is_available():
        logging.info(f'Still using cuda: {cuda}.')
    else:
        os.system('touch cuda_failure.txt')

    dataloaders = instantiate(cfg.dataset)

    model = instantiate(cfg.model, data_info=dataloaders['info'])

    trainer = Trainer(model, dataloaders, cfg)
    logging.info('Init complete.')

    # train with early stopping
    trainer.train_to_convergence()

    logging.info('Training completed sucessfully.')
    logging.info('Run completed sucessfully.')


if __name__ == '__main__':
    import os
    os.environ['HYDRA_FULL_ERROR'] = '1'

    def get_base_dir():
        return os.getenv('BASE_DIR', default='.')

    OmegaConf.register_resolver('BASE_DIR', get_base_dir)

    main()
