import torch
from argparse import ArgumentParser
import time
import numpy as np
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy

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

        self.wd = 2e-4
        self.all_out = True

        self.batch_size = 0
        self.new_classes = 0

        self.radius = 0
        self.model_old = None
        self.old_class = 0


        # Pass parameters
        self.kd_weight = 10
        self.protoAug_weight = 10
        self.temp = 0.1

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
        return torch.optim.Adam(params, lr=self.lr, weight_decay=self.wd)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        self.old_class += self.new_classes


        self.protoSave(t, self.model, trn_loader)

        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_loop(self, t, trn_loader, val_loader):
        self.batch_size = trn_loader.batch_size
        """Contains the epochs loop"""
        best_loss = np.inf

        self.optimizer = self._get_optimizer()
        self.scheduler = StepLR(self.optimizer, step_size=45, gamma=0.1)

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(t, trn_loader)
            clock1 = time.time()
            if self.eval_on_train:
                train_loss, train_acc, _ = self.eval(t, trn_loader)
                clock2 = time.time()
                print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
            else:
                print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

            # Valid
            clock3 = time.time()
            valid_loss, valid_acc, _ = self.eval(t, val_loader)
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc), end='')
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

            # Adapt learning rate - patience scheme - early stopping regularization
            if valid_loss < best_loss:
                best_loss = valid_loss
                print(' *', end='')

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        for img, target in trn_loader:

            # self-supervised learning based label augmentation
            images = torch.stack([torch.rot90(img, k, (2,3)) for k in range(4)], 1)
            images = images.view(-1, 3, 32, 32)
            target = torch.stack([target * 4 + k for k in range(4)], 1).view(-1)

            self.optimizer.zero_grad()

            loss = self._compute_loss(t, images.to(self.device), target.to(self.device), self.old_class)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
        self.scheduler.step()

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()

            for img, targets in val_loader:
                # Forward current model

                outputs = self.model(img.to(self.device))

                loss = self._compute_loss(t, img, targets.to(self.device), self.old_class)
                tmp = torch.cat(outputs, dim=1)
                tmp = tmp[:, ::4]
                tmp = list(tmp.reshape((1, tmp.shape[0],tmp.shape[1])))

                hits_taw, hits_tag = self.calculate_metrics(tmp, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw/total_num, total_acc_tag / total_num


    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        if self.all_out or len(self.exemplars_dataset) > 0:
            return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])

    def _compute_loss(self, t, imgs, target, old_class=0):       
        out, features = self.model(imgs.to(self.device), return_features = True)

        out = np.array(out)/self.temp

        loss_cls = self.criterion(t, out.tolist(), target)
        if self.model_old is None:
            return loss_cls
        else:
            _, features_old = self.model_old(imgs.to(self.device), return_features=True)
            loss_kd = torch.dist(features, features_old, 2)

            proto_aug = []
            proto_aug_label = []
            index = list(range(old_class))

            for _ in range(self.batch_size):
                np.random.shuffle(index)
                temp = self.prototype[index[0]] + np.random.normal(0,1,self.prototype[index[0]].shape[0]) * self.radius
                proto_aug.append(temp)
                proto_aug_label.append(4*self.class_label[index[0]])

            proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(self.device)
            proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(self.device)
            # Pass the prototype to the heads
            soft_feat_aug = []
            for head in self.model.heads:
                soft_feat_aug.append(head(proto_aug))
            soft_feat_aug = np.array(soft_feat_aug) / self.temp
            loss_protoAug = self.criterion(t, soft_feat_aug.tolist(), proto_aug_label)

            return loss_cls + self.protoAug_weight*loss_protoAug + self.kd_weight*loss_kd

    def protoSave(self, current_task, model, loader):
        features = []
        labels = []
        model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(loader):
                _, feature = model(images.to(self.device), return_features=True)
                if feature.shape[0] == self.batch_size:
                    labels.append(target.numpy())
                    features.append(feature.cpu().numpy())
        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
        feature_dim = features.shape[1]

        prototype = []
        radius = []
        class_label = []
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]
            prototype.append(np.mean(feature_classwise, axis=0))
            if current_task == 0:
                cov = np.cov(feature_classwise.T)
                radius.append(np.trace(cov) / feature_dim)

        if current_task == 0:
            self.radius = np.sqrt(np.mean(radius))
            self.prototype = prototype
            self.class_label = class_label
        else:
            self.prototype = np.concatenate((prototype, self.prototype), axis=0)
            self.class_label = np.concatenate((class_label, self.class_label), axis=0)
