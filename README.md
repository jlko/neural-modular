# neural-modular


Simple codebase for flexible neural net training.

Allows for seamless exchange of models, dataset, and optimizers.

Uses hydra for config-building and logging.

Option to enable wandb for run-tracking and cloud-storage.

Run `python main.py` to train your model.

## Understanding the Code

* `main.py` is the main entry point
* `conf/config.yaml` is the default config in standard Hydra syntax:
    * by running `python main.py +experiments=blabla.yaml` you can overwrite and extend the config by whatever you put in `experiments/blabla.yaml`.
    * alternatively you can run `python main.py +new=arg` to add `new` to the config, or `python main.py new=arg` to overwrite key `new`

* using the config, we then instantiate a dataset from `neural.datasets` and a model from `neural.models`

* model and dataset are then given to the trainer `neural.train.Trainer` which further instantiates optimizers, schedulers, and the losses

* we then train the model to convergence and checkpoint the final model

* see `neural.utils.restore` for how to restore a model/trainer instance
