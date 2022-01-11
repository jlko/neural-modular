"""Make PyTorch models work with SKLearn interface."""
import logging
import pickle
from pathlib import Path
import copy
import hydra
import wandb
from omegaconf import OmegaConf

import math
import numpy as np
import torch
import torch.nn.functional as F

from .utils.early_stopping import EarlyStop
from .utils import LabelSmoothingNLLLoss, L2_loss


class Trainer():

    def __init__(self, model, dataloaders, cfg):

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.cfg = cfg
        self.t_cfg = cfg.trainer
        self.dtype = torch.double

        self.model = model.to(device=self.device).type(
            self.dtype)

        self.dataloaders = dataloaders

        self.train_loader, self.val_loader, self.test_loader = [
            dataloaders[i] for i in
            ['train_loader', 'val_loader', 'test_loader']]

        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler(self.optimizer)
        self.loss = self.get_loss()
        self.is_class = self.cfg.dataset.type == 'classification'
        self.eval_loss = F.nll_loss if self.is_class else F.mse_loss
        self.early_stop = EarlyStop(
            self.model, self.t_cfg['early_stopping_epochs'])

        self.with_wandb = self.cfg.wandb.use_wandb

        logging.info(
            f'Initialising model \n'
            f'\t on device: {self.device}\n'
            f'\t optimizer: {self.optimizer.__class__}\n'
            f'\t scheduler: {self.scheduler}\n'
            f'\t loss: {self.loss}\n'
            f'\t eval loss: {self.eval_loss}\n'
            f"\t early stop: {self.t_cfg['early_stopping_epochs']}\n"
        )

    def restore(self, path):
        self.model.load_state_dict(
            torch.load(path, map_location=self.device))

    @staticmethod
    def d(tensor):
        return tensor.detach()

    @staticmethod
    def dc(tensor):
        return tensor.detach().cpu()

    @staticmethod
    def dcn(tensor):
        return tensor.detach().cpu().numpy()

    @staticmethod
    def cn(tensor):
        return tensor.cpu().numpy()

    def get_optimizer(self):
        c = self.t_cfg.get('optimizer', False)

        if not c or c == 'adam':
            optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.t_cfg['learning_rate'],
                    weight_decay=self.t_cfg['weight_decay'],)

        elif c == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.t_cfg['learning_rate'],
                momentum=0.9, weight_decay=self.t_cfg['weight_decay'])
        else:
            raise ValueError

        return optimizer

    def get_scheduler(self, optimizer):
        c = self.t_cfg.get('scheduler', False)
        epochs = self.t_cfg['max_epochs']

        if not c:
            scheduler = None

        elif c == 'drop-steps':
            milestones = [int(epochs * 0.5), int(epochs * 0.75)]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=0.1)

        elif c == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs)

        elif c == 'devries':
            # https://arxiv.org/abs/1708.04552v2
            assert epochs == 200
            milestones = [60, 120, 160]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=0.2)

        return scheduler

    def get_loss(self):
        """Randomly draw L2 loss."""
        loss = self.t_cfg.get('loss')

        if loss == 'label_smoothing':
            lam = self.t_cfg.get('label_smoothing', False)
            loss_base = LabelSmoothingNLLLoss(lam)
        elif loss == 'nll':
            loss_base = F.nll_loss
        elif loss == 'mse':
            loss_base = F.mse_loss
        else:
            raise ValueError

        if weights := self.t_cfg.get('custom_L2_loss', False):
            def combined_loss(preds, targets):
                return loss_base(preds, targets) + L2_loss(self.model, weights)

            return combined_loss
        else:
            return loss_base

    def train_to_convergence(self):
        logging.info(
            f'Beginning training with {len(self.train_loader.dataset)} '
            f'training points and {len(self.val_loader.dataset)} '
            f'validation points.')

        m = self.t_cfg['max_epochs']
        log_every = int(.01 * m) if m >= 500 else 1

        for epoch in range(self.t_cfg['max_epochs']):
            log_epoch = epoch % log_every == 0

            train_loss = self.run_epoch(return_loss=log_epoch)
            val_results = self.evaluate(self.val_loader)

            if log_epoch:
                test_results = self.evaluate(self.test_loader)
                log_data = epoch, train_loss, val_results, test_results
                self.log_epoch(*log_data)

            val_loss = val_results['metrics'][self.eval_loss.__name__]

            if self.early_stop.update(val_loss):
                logging.info(
                    f'Patience reached - stopping training. '
                    f'Best was {self.early_stop.best}')
                break

            if self.scheduler is not None:
                self.scheduler.step()

        logging.info('Completed training for acquisition.')

        self.model = self.early_stop.best_model
        torch.save(self.model.state_dict(), 'model.pth')

    def run_epoch(self, return_loss=False):
        self.model.train()
        if return_loss:
            losses = []

        for data, target in self.train_loader:

            data = data.to(self.device)
            target = target.to(self.device)

            # supposed to be faster
            # careful if optim has more than model params
            self.model.zero_grad(set_to_none=True)
            # self.optimizer.zero_grad()

            prediction = self.model(data)

            loss = self.loss(prediction, target)

            if return_loss:
                losses.append(self.d(loss))

            loss.backward()

            self.optimizer.step()

        if return_loss:
            return self.cn(torch.mean(torch.stack(losses)))

    def evaluate(self, eval_loader, return_preds=False):

        self.model.eval()
        loss_fn = self.eval_loss

        total_loss = 0
        if self.is_class:
            correct = 0

        if return_preds:
            predictions = []

        with torch.no_grad():
            for data, target in eval_loader:

                data = data.to(self.device)
                target = target.to(self.device)
                prediction = self.model(data)
                loss = loss_fn(
                    prediction, target, reduction='sum')
                total_loss += loss

                if self.is_class:
                    correct += (prediction.argmax(1) == target).sum()
                if return_preds:
                    predictions.append(self.d(prediction))

        total_loss /= len(eval_loader.dataset)

        out_dict = dict()
        out_dict['metrics'] = dict()
        out_dict['metrics'][loss_fn.__name__] = self.dcn(total_loss)

        if self.is_class:
            percentage_correct = 100.0 * correct / len(eval_loader.dataset)
            out_dict['metrics']['accuracy'] = self.dcn(percentage_correct)

        if return_preds:
            out_dict['predictions'] = self.dcn(torch.cat(predictions, 0))

        return out_dict

    def log_epoch(self, epoch, train_loss, val_results, test_results):
        combo = list(zip(
            ['val', 'test'],
            [val_results, test_results]))

        log_str = f'Epoch {epoch:0>3d} eval: '
        ln = self.loss.__name__
        log_str = f'{log_str} train {ln}: {train_loss:.4f} '

        for type, result in combo:
            tmp_str = ', '.join([
                f'{type} {metric}: {value:.4f}' for metric, value
                in result['metrics'].items()])
            log_str = f'{log_str} -- {tmp_str}'

        logging.info(log_str)

        if not self.cfg.wandb.use_wandb:
            return True

        log_dict = dict()
        log_dict[f'train_{self.loss.__name__}'] = train_loss

        for type, result in combo:
            for metric, value in result['metrics'].items():
                log_dict[f'{type}_{metric}'] = value

        wandb.log(log_dict, step=epoch)
