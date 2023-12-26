########################################
#        Zhenchun Lei
#  zhenchun.lei@hotmail.com
########################################

import logging
import os

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from asvspoof19.as19_experiment import AS19Experiment
from asvspoof19.as19_util import get_num_classes
from model.gmm import GMM
from util.util import show_model


class AS19GMMMPathExperiment(AS19Experiment):
    def __init__(self, model_type, feature_type, access_type, parm={}):
        super(AS19GMMMPathExperiment, self).__init__(model_type, feature_type, access_type, parm=parm)
        self._load_gmm()

    def _load_gmm(self):
        self.gmm_file = os.path.join(self.exp_path, '{}_{}'.format(self.parm['gmm_type'], self.feature_type),
                                     'ASVspoof2019_gmm_attack_{}_{}_{}_llk_mean_std.h5'.format(
                                         self.feature_type,
                                         self.access_type,
                                         self.parm['gmm_size']))
        logging.info(
            '[AS19GMMLLKMPathExperiment] Loading attack GMMs and mean std from : ' + self.gmm_file)
        self.gmm_attacks = []
        with h5py.File(self.gmm_file, 'r') as f:
            self.gmm_attacks.append(GMM(f['gmm_bonafide_w'][:],
                                        f['gmm_bonafide_mu'][:],
                                        f['gmm_bonafide_sigma'][:],
                                        llk_mean=f['feat_bonafide_mean'][:],
                                        llk_std=f['feat_bonafide_std'][:],
                                        use_gpu=self.use_gpu))

            self.gmm_attacks.append(GMM(f['gmm_spoof_w'][:],
                                        f['gmm_spoof_mu'][:],
                                        f['gmm_spoof_sigma'][:],
                                        llk_mean=f['feat_spoof_mean'][:],
                                        llk_std=f['feat_spoof_std'][:],
                                        use_gpu=self.use_gpu))

            self.attack_type = f['attack_type'][:][0]
            for idx in range(len(self.attack_type)):
                attack_name = self.attack_type[idx].decode()
                gmm = GMM(f['gmm_attack_' + attack_name + '_w'][:],
                          f['gmm_attack_' + attack_name + '_mu'][:],
                          f['gmm_attack_' + attack_name + '_sigma'][:],
                          llk_mean=f['feat_attack_' + attack_name + '_mean'][:],
                          llk_std=f['feat_attack_' + attack_name + '_std'][:],
                          use_gpu=self.use_gpu)
                self.gmm_attacks.append(gmm)


class AS19GMMMPath2StepExperiment(AS19GMMMPathExperiment):
    def __init__(self, model_type, feature_type, access_type, parm={}):
        super(AS19GMMMPath2StepExperiment, self).__init__(model_type, feature_type, access_type, parm=parm)

    def train_model(self, model, train_loader):
        self.train_resnet_blocks(model, train_loader)
        self.train_classifier(model, train_loader)

    def train_resnet_blocks(self, model, train_loader):
        logging.info('=======Training ResNet M Paths ......')
        path_num = len(model.paths)
        path_input_size = self.parm['gmm_size']
        class_num = get_num_classes(self.label_type)

        path_classifiers = []
        optimizers = []
        criterions = []
        schedulers = []
        for idx in range(path_num):
            # path_classifier = model.paths[idx]
            path_classifier = nn.Sequential(model.paths[idx], nn.Linear(model.paths[idx].output_size, class_num))
            if self.use_gpu:
                path_classifier = path_classifier.cuda()

            optimizer = self.get_optimizer(path_classifier)
            optimizers.append(optimizer)
            schedulers.append(self.get_scheduler(optimizer))
            criterion = self.get_criterion()
            if self.use_gpu:
                criterion = criterion.cuda()
            criterions.append(criterion)

            path_classifier.train()
            path_classifiers.append(path_classifier)
        show_model(path_classifiers[0])

        for epoch in range(self.num_epochs):
            correct = torch.zeros(path_num, dtype=torch.int)
            total = torch.zeros(path_num, dtype=torch.int)
            total_classify_loss = torch.zeros(path_num, dtype=torch.float)
            total_regularization_loss = torch.zeros(path_num, dtype=torch.float)

            for batch_idx, (data, target, data_idx) in enumerate(train_loader):
                if self.use_gpu:
                    data, target = data.cuda(), target.cuda()

                # if self.exp_feat_transform:
                #     data = self._feature_transform(data)

                for idx in range(path_num):
                    optimizers[idx].zero_grad()
                    # output = path_classifiers[idx](data[:, path_input_size * idx:path_input_size * (idx + 1), :])
                    output = path_classifiers[idx](data)

                    loss = criterions[idx](output, target)
                    total_classify_loss[idx] += loss.item()

                    # if self.parm['use_regularization_loss']:
                    #     regularization_loss1 = self.get_regularization_loss(self.path1)
                    #     total_regularization_loss1 += regularization_loss1
                    #     loss1 += regularization_loss1

                    loss.backward()
                    optimizers[idx].step()

                    _, predict_label = torch.max(output, 1)
                    correct[idx] += (predict_label.cpu() == target.cpu()).sum().numpy()
                    total[idx] += target.size(0)

            if self.parm['use_scheduler']:
                for idx in range(path_num):
                    schedulers[idx].step(total_classify_loss[idx])

            if self.verbose >= 1:
                train_accuracy = torch.sum(correct) / torch.sum(total)
                if self.parm['use_regularization_loss']:
                    logging.info(
                        "Train Epoch: {}/{} classify loss = {:.6f} Regu loss = {:.6f} Total = {:.6f} accuracy={:.4f}%".format(
                            epoch + 1, self.num_epochs,
                            torch.mean(total_classify_loss), torch.mean(total_regularization_loss),
                            torch.mean(total_classify_loss) + torch.mean(total_regularization_loss),
                            100.0 * train_accuracy))
                else:
                    logging.info(
                        "Train Epoch: {}/{} Total classify loss = {:.6f} accuracy={:.4f}%".format(epoch + 1,
                                                                                                  self.num_epochs,
                                                                                                  torch.mean(
                                                                                                      total_classify_loss),
                                                                                                  100.0 * train_accuracy))

    def train_classifier(self, model, train_loader):
        logging.info('=======Training Classifier ......')
        paths = model.paths
        path_num = len(paths)
        path_input_size = self.parm['gmm_size']

        classifier = model.classifier

        for idx in range(path_num):
            paths[idx].eval()

        dcnn_output = None
        label_id = torch.zeros((len(train_loader.dataset),), dtype=np.long)
        with torch.no_grad():
            for batch_idx, (data, target, data_idx) in enumerate(train_loader):
                if self.use_gpu:
                    data, target = data.cuda(), target.cuda()

                # if self.exp_feat_transform:
                #     data = self._feature_transform(data)

                output = []
                for idx in range(path_num):
                    # subdata = data[:, path_input_size * idx:path_input_size * (idx + 1), :]

                    output.append(paths[idx](data))

                data = torch.cat(output, dim=1)

                if dcnn_output is None:
                    dcnn_output = torch.zeros((len(train_loader.dataset), data.shape[1]))
                dcnn_output[data_idx, :] = data.cpu()
                label_id[data_idx] = target.cpu()

        dcnn_out_dataset = TensorDataset(dcnn_output, label_id)
        train_loader = DataLoader(dcnn_out_dataset, batch_size=self.parm['batch_size'], shuffle=True)

        logging.info(classifier)
        optimizer = self.get_optimizer(classifier)
        scheduler = self.get_scheduler(optimizer)
        criterion = self.get_criterion()
        if self.use_gpu:
            criterion = criterion.cuda()

        classifier.train()
        for epoch in range(self.num_epochs):
            correct = 0
            total = 0
            total_classify_loss = 0.0
            total_regularization_loss = 0.0

            for batch_idx, (data, target) in enumerate(train_loader):
                if self.use_gpu:
                    data, target = data.cuda(), target.cuda()

                optimizer.zero_grad()
                output = classifier(data)

                loss = criterion(output, target)
                total_classify_loss += loss.item()
                # if self.parm['use_regularization_loss']:
                #     regularization_loss = self.get_regularization_loss(self.classifier)
                #     total_regularization_loss += regularization_loss
                #     loss += regularization_loss

                loss.backward()
                optimizer.step()

                _, predict_label = torch.max(output, 1)
                correct += (predict_label.cpu() == target.cpu()).sum().numpy()
                total += target.size(0)

            if self.parm['use_scheduler']:
                scheduler.step(total_classify_loss)

            if self.verbose >= 1:
                train_accuracy = correct / total
                if self.parm['use_regularization_loss']:
                    logging.info(
                        "Train Epoch: {}/{} classify loss = {:.6f} Regu loss = {:.6f} Total = {:.6f} accuracy={:.4f}%".format(
                            epoch + 1, self.num_epochs,
                            total_classify_loss, total_regularization_loss,
                            total_classify_loss + total_regularization_loss,
                            100.0 * train_accuracy))
                else:
                    logging.info(
                        "Train Epoch: {}/{} Total classify loss = {:.6f} accuracy={:.4f}%".format(epoch + 1,
                                                                                                  self.num_epochs,
                                                                                                  total_classify_loss,
                                                                                                  100.0 * train_accuracy))
