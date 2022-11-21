from random import shuffle
import torch
from torch import nn
from argparse import ArgumentParser

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ContrastiveExemplarsDataset
from datasets.memory_dataset import MemoryDataset


class Appr(Inc_Learning_Appr):
    """Class implementing the finetuning baseline"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000, momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, all_outputs=False, T=0.1, delta=2.0, contrastive_gamma=1.0):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.all_out = all_outputs
        self.T = T
        self.gamma = contrastive_gamma

        # 
        self._n_classes = 0
    @staticmethod
    def exemplars_dataset_class():
        return ContrastiveExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--all-outputs', action='store_true', required=False,
                            help='Allow all weights related to all outputs to be modified (default=%(default)s)')
        parser.add_argument('--T', default=0.1, required=False, type=float,
                            help='Temperature scaling (default=%(default)s)')        
        parser.add_argument('--delta', default=2.0, required=False, type=float,
                            help='Delta (default=%(default)s)')                
        parser.add_argument('--contrastive-gamma', default=1.0, required=False, type=float,
                            help='Contrastive gamma (default=%(default)s)')

        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1 and not self.all_out:
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = list(self.model.parameters()) 
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

                for x1, x2, targets in trn_loader:

                    x = torch.cat((x1,x2), dim=0)
                    labels = torch.cat((targets, targets), dim=0)
            
                    # Forward current model
                    out, gx = self.model(x.to(self.device), return_features=True)
            

                    contrastive = self.contrastive_loss(out, labels.to(self.device), self.T)

                    cross_entropy = self.criterion(t, out, labels.to(self.device))

                    loss = self.gamma * contrastive + cross_entropy

                    # Backward
                    scheduler.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
                    scheduler.step_and_update_lr()
                print('Loss:{}'.format(loss.item()))
    
    def _init_exemplars_loader(self, trn_loader):
        bs = 10 if trn_loader.batch_size < 100 else trn_loader.batch_size // 10
        self.exemplars_loader = torch.utils.data.DataLoader(self.exemplars_dataset,
                                                          batch_size=bs,
                                                          num_workers=trn_loader.num_workers,
                                                          pin_memory=trn_loader.pin_memory,
                                                          shuffle=True)
        self.exemplars_iter = iter(self.exemplars_loader)
    
    def _reset_exemplars_loader(self):
        self.exemplars_iter = iter(self.exemplars_loader)


    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            collect_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                          batch_size=trn_loader.batch_size,
                                                          num_workers=trn_loader.num_workers,
                                                          pin_memory=trn_loader.pin_memory,
                                                          shuffle=True)
            self._init_exemplars_loader(trn_loader)
        else:
            collect_loader=trn_loader

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset

        self.exemplars_dataset.collect_exemplars(self.model, collect_loader, val_loader.dataset.transform)

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        for x1, x2, targets in trn_loader:
            # Get the exemplars from the reharsal memory
            if len(self.exemplars_dataset) > 0 and t > 0:
                try:
                    ex1, ex2, ex_tgs = next(self.exemplars_iter)
                except StopIteration:                 
                    self._reset_exemplars_loader()
                    ex1, ex2, ex_tgs = next(self.exemplars_iter)
                ex = torch.cat((ex1, ex2), dim=0)
                ex_labels = torch.cat((ex_tgs, ex_tgs), dim=0)
                ex_out, ex_gx = self.model(ex.to(self.device), return_features=True)


            x = torch.cat((x1,x2), dim=0)
            labels = torch.cat((targets, targets), dim=0)
            
            # Forward current model
            out, gx = self.model(x.to(self.device), return_features=True)

            # Create the input for the contrastive loss.
            if len(self.exemplars_dataset) > 0 and t > 0:
                contrastive_features = torch.cat((gx, ex_gx), dim=0)
                contrastive_labels = torch.cat((labels,ex_labels))
            else:
                contrastive_features = gx
                contrastive_labels = labels
            

            contrastive = self.contrastive_loss(contrastive_features, contrastive_labels.to(self.device), self.T)
            
            cross_entropy = self.criterion(t, out, labels.to(self.device))

            if len(self.exemplars_dataset) > 0 and t > 0:
                cross_entropy += self.criterion(t, ex_out, ex_labels.to(self.device))

            loss = self.gamma * contrastive + cross_entropy

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
        print('Loss:{}'.format(loss.item()))

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()

            for x1, x2, targets in val_loader:
                # Forward current model
                outputs, gx = self.model(x1.to(self.device), return_features=True)

                
                contrastive = self.contrastive_loss(gx, targets.to(self.device), self.T)
                cross_entropy = self.criterion(t, outputs, targets.to(self.device))

                loss = self.gamma * contrastive + cross_entropy

                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        if self.all_out or len(self.exemplars_dataset) > 0:
            return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])


    def contrastive_loss(self, features, targets, temperature, delta_mask=None):
        """
        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """

        # It's possible to apply L2 norm to input vectors 
        features = torch.nn.functional.normalize(features, p=2, dim=0)

        dot_product_tempered = torch.mm(features, features.T) / temperature
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(self.device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(self.device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        if delta_mask is not None:
            log_prob = log_prob * delta_mask
        cardinality_per_samples[cardinality_per_samples == 0] = 1.0
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss

class WarmupScheduler():
    '''A simple wrapper class for learning rate scheduling'''

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
        