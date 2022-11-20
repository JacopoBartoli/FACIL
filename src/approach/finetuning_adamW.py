import torch
from argparse import ArgumentParser

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing the finetuning baseline"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, all_outputs=False):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.all_out = all_outputs

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--all-outputs', action='store_true', required=False,
                            help='Allow all weights related to all outputs to be modified (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1 and not self.all_out:
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.wd)

    def pre_train_process(self, t, trn_loader):
        if self.warmup_epochs > 0:
            lr_max = self.warmup_lr
            n_steps = len(trn_loader) * self.warmup_epochs
            optim = self._get_optimizer()
            scheduler = WarmupScheduler(optimizer=optim, lr_max=lr_max, lr_min=self.lr, n_steps=n_steps)

            for epoch in range(self.warmup_epochs):
                """Runs a single epoch"""
                self.model.train()
                if self.fix_bn and t > 0:
                    self.model.freeze_bn()
                for images, targets in trn_loader:
                    # Forward current model
                    outputs = self.model(images.to(self.device))
                    loss = self.criterion(t, outputs, targets.to(self.device))
                    # Backward
                    scheduler.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
                    scheduler.step_and_update_lr()
                print('Loss:{}'.format(loss.item()))

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        if self.all_out or len(self.exemplars_dataset) > 0:
            return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])

class WarmupScheduler():
    '''A simple wrapper class for learning rate scheduling.
    Learning rate linear increase to a target value.'''

    def __init__(self, optimizer, lr_max, lr_min, n_steps):
        self._optimizer = optimizer
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.max_steps = n_steps
        self.n_steps = 0

        self.step = (lr_max - lr_min) /  self.max_steps


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._optimizer.step()
        self._update_learning_rate()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr(self):
        return (self.n_steps) * self.step


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        lr = self._get_lr()        
        self.n_steps += 1

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr