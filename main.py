"""Main active testing loop."""
import os
import logging
import hydra
import warnings

import numpy as np
import torch

from omegaconf import OmegaConf

from neural.utils import maps
from neural.train import Trainer


@hydra.main(config_path='conf', config_name='config')
def main(cfg):
    """Train single model, evaluate on test data, save predictions and model.
    """

    if cfg.wandb.use_wandb:
        import wandb
        # import pandas
        # wandb_cfg = pandas.json_normalize(dict(cfg), sep='_')
        # wandb_cfg = wandb_cfg.to_dict(orient='records')[0]
        from neural.utils import flatten_dict
        wandb_cfg = flatten_dict(dict(cfg))

        wandb_args = dict(
            project=cfg.wandb.project,
            entity="jlko",
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

    dataloaders = (
        maps.dataset[cfg.dataset.name](cfg.dataset, seed))

    data_in, data_out = [dataloaders['info'][i] for i in ['D_in', 'D_out']]

    model = maps.model[cfg.model.name](cfg.model, data_in, data_out)

    trainer = Trainer(model, dataloaders, cfg)
    logging.info('Init complete.')

    # train with early stopping
    trainer.train_to_convergence()

    # predict on val_model_selection set, predict on test set
    logging.info('Training completed sucessfully.')
    logging.info('Run completed suc essfully.')


if __name__ == '__main__':
    import os
    os.environ['HYDRA_FULL_ERROR'] = '1'

    def get_base_dir():
        return os.getenv('BASE_DIR', default='.')

    OmegaConf.register_resolver('BASE_DIR', get_base_dir)

    main()
