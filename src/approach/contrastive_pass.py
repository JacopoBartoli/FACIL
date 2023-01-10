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
                 logger=None, exemplars_dataset=None, all_outputs=False, unsupervised=False, proto_con = True, is_cct=False):
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
        
        # Parameters to select the type of experiment
        self.unsupervised = unsupervised
        self.proto_con = proto_con
        # To implement.
        # We can consider 3 case, proto initialized by proto save and then learned and maybe distilled.
        self.save_and_learn = False
        # Proto learned at the end of a task and then not changed
        self.learn_and_freeze = False
        # Proto not initialized by proto save but learned and maybe distilled.
        self.full_learnable = False
        if self.full_learnable or self.save_and_learn or self.learn_and_freeze:
            self.learnable_proto = True
        else:
            self.learnable_proto = False
        if self.learnable_proto:
            self._init_learnable_prototypes

        # Enanble distillation of external parameter.(Key and Bias)
        self.ext_params = True
        self.is_cct = is_cct

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
        
        parser.add_argument('--unsupervised', action='store_true', required=False,
                            help='Allow to use unsupervised contrastive loss (default=%(default)s)')

        parser.add_argument('--proto-con', type=bool, required=False, default=True,
                            help='Allow to use unsupervised contrastive loss (default=%(default)s)')
        
        #parser.add_argument('--learnable-proto', action='store_true', required=False,
        #                    help='Allow to use unsupervised contrastive loss (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1 and not self.all_out:
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        if self.learnable_proto:   
            params.append(self.prototypes)
        return torch.optim.Adam(params, lr=self.lr, weight_decay=self.wd)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        self.old_class += self.new_classes


        self.protoSave(t, self.model, trn_loader)

        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def _init_learnable_prototypes(self, num_classes=100):
        self.prototypes = torch.nn.Parameter(torch.randn((num_classes, self.model.model.embedding_dim), device=self.device, requires_grad=True))
        
        self.proto_labels = torch.arange(0, 100)

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
                clock4 - clock3, valid_loss, 100 * valid_acc), end='\n')
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

            # Adapt learning rate - patience scheme - early stopping regularization
            if valid_loss < best_loss:
                best_loss = valid_loss
                print(' *', end='\n')

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
            if self.learnable_proto:
                torch.nn.utils.clip_grad_norm_(self.prototypes, self.clipgrad)
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

    def _compute_loss(self, t, imgs, target, old_class=0):       
        out, features = self.model(imgs.to(self.device), return_features = True)

        out = np.array(out)/self.temp
            
        if self.unsupervised:
            first_aug = torch.reshape(features[::4],(self.batch_size, 1, -1))
            second_aug = torch.reshape(features[1::4],(self.batch_size, 1, -1))
            third_aug = torch.reshape(features[2::4],(self.batch_size, 1, -1))
            forth_aug = torch.reshape(features[3::4],(self.batch_size, 1, -1))
            con_feat = torch.cat((first_aug, second_aug, third_aug, forth_aug), dim=1)
        else:
            con_feat = features

        loss_cls = self.criterion(t, out.tolist(), target)
        if self.model_old is None:
            if self.unsupervised:
                loss_con = self.unsupervised_contrastive_loss(con_feat)
            else:
                loss_con = self.contrastive_loss(con_feat, target, temperature=1)
            return loss_cls + loss_con
        else:
            _, features_old = self.model_old(imgs.to(self.device), return_features=True)
            loss_kd = torch.dist(features, features_old, 2)

            if self.ext_params and not self.is_cct:
                ext_k, ext_b, ext_k_old, ext_b_old = [], [], [], []
                # Distillation of external bias and external key.
                for item, item_old in zip(self.model.model.transformer.layers, self.model_old.model.transformer.layers):
                    ext_k.append(item[0].fn.ext_k) 
                    ext_b.append(item[0].fn.ext_bias)

                    ext_k_old.append(item_old[0].fn.ext_k) 
                    ext_b_old.append(item_old[0].fn.ext_bias)

                ext_k = torch.cat(ext_k, dim=1)
                ext_b = torch.cat(ext_b, dim=1)
                ext_k_old = torch.cat(ext_k_old, dim=1)
                ext_b_old = torch.cat(ext_b_old, dim=1)
                loss_ext_dist = torch.dist(ext_k, ext_k_old, 2) + torch.dist(ext_b, ext_b_old, 2)
            elif self.ext_params and self.is_cct:
                ext_k, ext_b, ext_k_old, ext_b_old = [], [], [], []
                for item, item_old in zip(self.model.model.classifier.blocks, self.model_old.model.classifier.blocks):
                    ext_k.append(item.self_attn.ext_k) 
                    ext_b.append(item.self_attn.ext_bias)

                    ext_k_old.append(item_old.self_attn.ext_k) 
                    ext_b_old.append(item_old.self_attn.ext_bias)

                ext_k = torch.cat(ext_k, dim=1)
                ext_b = torch.cat(ext_b, dim=1)
                ext_k_old = torch.cat(ext_k_old, dim=1)
                ext_b_old = torch.cat(ext_b_old, dim=1)
                loss_ext_dist = torch.dist(ext_k, ext_k_old, 2) + torch.dist(ext_b, ext_b_old, 2)
            else:
                loss_ext_dist = 0

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

            if self.unsupervised:
                loss_con = self.unsupervised_contrastive_loss(con_feat)
            else:
                con_feat = torch.cat((features, proto_aug), dim=0)
                con_target = torch.cat((target, proto_aug_label), dim=0)
                loss_con = self.contrastive_loss(con_feat, con_target, temperature=1)

            return loss_cls + self.protoAug_weight*loss_protoAug + self.kd_weight*loss_kd + loss_con + loss_ext_dist

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

    def unsupervised_contrastive_loss(self, features):
        """ SimCLR unsupervised loss:
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
        Returns:
            A loss scalar.
        """
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)        
         
        features = torch.nn.functional.normalize(features, p=2, dim=0)

        batch_size = features.shape[0]
        mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            1)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() + 1e-5

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss