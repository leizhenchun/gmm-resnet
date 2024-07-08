########################################
#        Zhenchun Lei
#  zhenchun.lei@hotmail.com
########################################

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from asvspoof21.experiment.as19_experiment import AS19Experiment
from asvspoof21.experiment.as19_util import get_num_classes
from asvspoof21.util.as_util import show_model


class AS192StepExperiment(AS19Experiment):
    def __init__(self, model_type, feature_type, access_type, parm={}):
        super(AS192StepExperiment, self).__init__(model_type, feature_type, access_type, parm=parm)

    def train_model(self, model, train_loader):
        self.train_resnet_blocks(model, train_loader)
        self.train_classifier(model, train_loader)

    def train_resnet_blocks(self, model, train_loader):
        logging.info('=======Training ResNet 2 Paths ......')

        class_num = get_num_classes(self.label_type)
        path_classifier1 = nn.Sequential(model.path1, nn.Linear(model.path1.output_size, class_num))
        path_classifier2 = nn.Sequential(model.path2, nn.Linear(model.path2.output_size, class_num))
        if self.use_gpu:
            path_classifier1 = path_classifier1.cuda()
            path_classifier2 = path_classifier2.cuda()
        show_model(path_classifier1)

        optimizer1 = self.get_optimizer(path_classifier1)
        optimizer2 = self.get_optimizer(path_classifier2)
        scheduler1 = self.get_scheduler(optimizer1)
        scheduler2 = self.get_scheduler(optimizer2)
        criterion1 = self.get_criterion()
        criterion2 = self.get_criterion()
        if self.use_gpu:
            criterion1 = criterion1.cuda()
            criterion2 = criterion2.cuda()

        path_classifier1.train()
        path_classifier2.train()

        for epoch in range(self.num_epochs):
            correct1 = 0
            total1 = 0
            total_classify_loss1 = 0.0
            total_regularization_loss1 = 0.0
            correct2 = 0
            total2 = 0
            total_classify_loss2 = 0.0
            total_regularization_loss2 = 0.0

            for batch_idx, (data, target, data_idx) in enumerate(train_loader):
                if self.use_gpu:
                    data, target = data.cuda(), target.cuda()

                # if self.exp_feat_transform:
                #     data = self._feature_transform(data)

                # data1 = data[:, 0:path_input_size, :]
                # data2 = data[:, path_input_size:, :]

                # train path1
                optimizer1.zero_grad()
                output1 = path_classifier1(data)

                loss1 = criterion1(output1, target)
                total_classify_loss1 += loss1.item()
                # if self.parm['use_regularization_loss']:
                #     regularization_loss1 = self.get_regularization_loss(self.path1)
                #     total_regularization_loss1 += regularization_loss1
                #     loss1 += regularization_loss1

                loss1.backward()
                optimizer1.step()

                _, predict_label1 = torch.max(output1, 1)
                correct1 += (predict_label1.cpu() == target.cpu()).sum().numpy()
                total1 += target.size(0)

                if self.verbose >= 2 and batch_idx % 100 == 0:
                    logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss1.item()))

                # train path2
                optimizer2.zero_grad()
                output2 = path_classifier2(data)

                loss2 = criterion2(output2, target)
                total_classify_loss2 += loss2.item()
                # if self.parm['use_regularization_loss']:
                #     regularization_loss2 = self.get_regularization_loss(self.path2)
                #     total_regularization_loss2 += regularization_loss2
                #     loss2 += regularization_loss2

                loss2.backward()
                optimizer2.step()

                _, predict_label2 = torch.max(output2, 1)
                correct2 += (predict_label2.cpu() == target.cpu()).sum().numpy()
                total2 += target.size(0)

            if self.parm['use_scheduler']:
                scheduler1.step(total_classify_loss1)
                scheduler2.step(total_classify_loss2)

            if self.verbose >= 1:
                train_accuracy1 = correct1 / total1
                train_accuracy2 = correct2 / total2
                if self.parm['use_regularization_loss']:
                    logging.info(
                        "Train Epoch: {}/{} classify loss = {:.6f} Regu loss = {:.6f} Total = {:.6f} accuracy={:.4f}%".format(
                            epoch + 1, self.num_epochs,
                            total_classify_loss1 + total_classify_loss2,
                            total_regularization_loss1 + total_regularization_loss2,
                            total_classify_loss1 + total_regularization_loss1 + total_classify_loss2 + total_regularization_loss2,
                            100.0 * (train_accuracy1 + train_accuracy2) / 2))
                    # logging.info(
                    #     "Train Epoch: {}/{} classify loss = {:.6f} Regu loss = {:.6f} Total = {:.6f} accuracy={:.4f}%".format(
                    #         epoch + 1, self.parm['self.num_epochs'],
                    #         total_classify_loss2, total_regularization_loss2,
                    #         total_classify_loss2 + total_regularization_loss2,
                    #         100.0 * train_accuracy2))
                else:
                    logging.info(
                        "Train Epoch: {}/{} Total classify loss 1 = {:.6f} accuracy={:.4f}%".format(epoch + 1,
                                                                                                    self.num_epochs,
                                                                                                    total_classify_loss1,
                                                                                                    100.0 * train_accuracy1))
                    logging.info(
                        "Train Epoch: {}/{} Total classify loss 2 = {:.6f} accuracy={:.4f}%".format(epoch + 1,
                                                                                                    self.num_epochs,
                                                                                                    total_classify_loss2,
                                                                                                    100.0 * train_accuracy2))

    def train_classifier(self, model, train_loader):
        logging.info('=======Training Classifier ......')
        # path_input_size = self.parm['gmm_size']

        # model.path1.classifier = nn.Identity()
        # model.path2.classifier = nn.Identity()
        model.path1.eval()
        model.path2.eval()

        dcnn_output = None
        label_id = torch.zeros((len(train_loader.dataset),), dtype=torch.long)

        with torch.no_grad():
            for batch_idx, (data, target, data_idx) in enumerate(train_loader):
                if self.use_gpu:
                    data, target = data.cuda(), target.cuda()

                # if self.exp_feat_transform:
                #     data = self._feature_transform(data)

                # data1 = data[:, :path_input_size, :]
                # data2 = data[:, path_input_size:, :]

                data1 = model.path1(data)
                data2 = model.path2(data)
                data = torch.cat((data1, data2), dim=1)

                if dcnn_output is None:
                    dcnn_output = torch.zeros((len(train_loader.dataset), data.shape[1]))
                dcnn_output[data_idx, :] = data.cpu()
                label_id[data_idx] = target.cpu()

        dcnn_out_dataset = TensorDataset(dcnn_output, label_id)
        train_loader = DataLoader(dcnn_out_dataset, batch_size=self.parm['batch_size'], shuffle=True)

        logging.info(model.classifier)
        optimizer = self.get_optimizer(model.classifier)
        scheduler = self.get_scheduler(optimizer)
        criterion = self.get_criterion()
        if self.use_gpu:
            criterion = criterion.cuda()

        model.classifier.train()
        for epoch in range(self.num_epochs):
            correct = 0
            total = 0
            total_classify_loss = 0.0
            total_regularization_loss = 0.0

            for batch_idx, (data, target) in enumerate(train_loader):
                if self.use_gpu:
                    data, target = data.cuda(), target.cuda()

                optimizer.zero_grad()
                output = model.classifier(data)

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

    # deprecated
    # def train_model_old(self, model, train_loader):
    #     if self.use_gpu:
    #         self.path1 = self.path1.cuda()
    #         self.path2 = self.path2.cuda()
    #         self.classifier = self.classifier.cuda()
    #     show_model(self.path1)
    #     show_model(self.path2)
    #
    #     optimizer1 = self.get_optimizer(self.path1)
    #     optimizer2 = self.get_optimizer(self.path2)
    #     criterion1 = self.get_criterion()
    #     criterion2 = self.get_criterion()
    #     if self.use_gpu:
    #         criterion1 = criterion1.cuda()
    #         criterion2 = criterion2.cuda()
    #
    #     if self.nn_para['use_scheduler']:
    #         scheduler1 = self.get_scheduler(optimizer1)
    #         scheduler2 = self.get_scheduler(optimizer2)
    #
    #     self.path1.train()
    #     self.path2.train()
    #     for epoch in range(self.nn_para['num_epochs']):
    #         correct1 = 0
    #         total1 = 0
    #         total_classify_loss1 = 0.0
    #         total_regularization_loss1 = 0.0
    #         correct2 = 0
    #         total2 = 0
    #         total_classify_loss2 = 0.0
    #         total_regularization_loss2 = 0.0
    #
    #         for batch_idx, (data, target, data_idx) in enumerate(train_loader):
    #             data1 = data[:, 0:self.num_channels // 2, :]
    #             data2 = data[:, self.num_channels // 2:, :]
    #
    #             if self.use_gpu:
    #                 data1, data2, target = data1.cuda(), data2.cuda(), target.cuda()
    #
    #             # train dcnn1
    #             optimizer1.zero_grad()
    #             output1 = self.path1(data1)
    #
    #             loss1 = criterion1(output1, target)
    #             total_classify_loss1 += loss1.item()
    #             # if self.nn_para['use_regularization_loss']:
    #             #     regularization_loss1 = self.get_regularization_loss(self.path1)
    #             #     total_regularization_loss1 += regularization_loss1
    #             #     loss1 += regularization_loss1
    #
    #             loss1.backward()
    #             optimizer1.step()
    #
    #             _, predict_label1 = torch.max(output1, 1)
    #             correct1 += (predict_label1.cpu() == target.cpu()).sum().numpy()
    #             total1 += target.size(0)
    #
    #             if self.verbose >= 2 and batch_idx % 100 == 0:
    #                 logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #                     epoch, batch_idx * len(data), len(train_loader.dataset),
    #                            100. * batch_idx / len(train_loader), loss1.item()))
    #
    #             # train dcnn2
    #             optimizer2.zero_grad()
    #             output2 = self.path2(data2)
    #
    #             loss2 = criterion2(output2, target)
    #             total_classify_loss2 += loss2.item()
    #             # if self.nn_para['use_regularization_loss']:
    #             #     regularization_loss2 = self.get_regularization_loss(self.path2)
    #             #     total_regularization_loss2 += regularization_loss2
    #             #     loss2 += regularization_loss2
    #
    #             loss2.backward()
    #             optimizer2.step()
    #
    #             _, predict_label2 = torch.max(output2, 1)
    #             correct2 += (predict_label2.cpu() == target.cpu()).sum().numpy()
    #             total2 += target.size(0)
    #
    #             if self.verbose >= 2 and batch_idx % 100 == 0:
    #                 logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #                     epoch, batch_idx * len(data), len(train_loader.dataset),
    #                            100. * batch_idx / len(train_loader), loss2.item()))
    #
    #         train_accuracy1 = correct1 / total1
    #         train_accuracy2 = correct2 / total2
    #         if self.verbose >= 1:
    #             if self.nn_para['use_regularization_loss']:
    #                 logging.info(
    #                     "Train Epoch: {}/{} classify loss = {:.6f} Regu loss = {:.6f} Total = {:.6f} accuracy={:.4f}%".format(
    #                         epoch + 1, self.nn_para['num_epochs'],
    #                         total_classify_loss1, total_regularization_loss1,
    #                         total_classify_loss1 + total_regularization_loss1,
    #                         100.0 * train_accuracy1))
    #                 logging.info(
    #                     "Train Epoch: {}/{} classify loss = {:.6f} Regu loss = {:.6f} Total = {:.6f} accuracy={:.4f}%".format(
    #                         epoch + 1, self.nn_para['num_epochs'],
    #                         total_classify_loss2, total_regularization_loss2,
    #                         total_classify_loss2 + total_regularization_loss2,
    #                         100.0 * train_accuracy2))
    #             else:
    #                 logging.info(
    #                     "Train Epoch: {}/{} Total classify loss = {:.6f} accuracy={:.4f}%".format(epoch + 1,
    #                                                                                               self.nn_para[
    #                                                                                                   'num_epochs'],
    #                                                                                               total_classify_loss1,
    #                                                                                               100.0 * train_accuracy1))
    #                 logging.info(
    #                     "Train Epoch: {}/{} Total classify loss = {:.6f} accuracy={:.4f}%".format(epoch + 1,
    #                                                                                               self.nn_para[
    #                                                                                                   'num_epochs'],
    #                                                                                               total_classify_loss2,
    #                                                                                               100.0 * train_accuracy2))
    #
    #         if self.nn_para['use_scheduler']:
    #             scheduler1.step(total_classify_loss1)
    #             scheduler2.step(total_classify_loss2)
    #
    #     self.path1.classifier = nn.Identity()
    #     self.path2.classifier = nn.Identity()
    #     self.path1.eval()
    #     self.path2.eval()
    #
    #     dcnn_output = None
    #     label_id = torch.zeros((len(train_loader.dataset),), dtype=np.long)
    #     with torch.no_grad():
    #         for batch_idx, (data, target, data_idx) in enumerate(train_loader):
    #             data1 = data[:, :self.num_channels // 2, :]
    #             data2 = data[:, self.num_channels // 2:, :]
    #             if self.use_gpu:
    #                 data1, data2, target = data1.cuda(), data2.cuda(), target.cuda()
    #
    #             data1 = self.path1(data1)
    #             data2 = self.path2(data2)
    #             data = torch.cat((data1, data2), dim=1)
    #
    #             if dcnn_output is None:
    #                 dcnn_output = torch.zeros((len(train_loader.dataset), data.shape[1]))
    #             dcnn_output[data_idx, :] = data.cpu()
    #             label_id[data_idx] = target.cpu()
    #
    #     dcnn_out_dataset = TensorDataset(dcnn_output, label_id)
    #     train_loader = DataLoader(dcnn_out_dataset, batch_size=self.nn_para['batch_size'], shuffle=True)
    #
    #     logging.info(self.classifier)
    #     optimizer = self.get_optimizer(self.classifier)
    #     criterion = self.get_criterion()
    #     if self.use_gpu:
    #         criterion = criterion.cuda()
    #     if self.nn_para['use_scheduler']:
    #         scheduler = self.get_scheduler(optimizer)
    #
    #     self.classifier.train()
    #     for epoch in range(self.nn_para['num_epochs']):
    #         correct = 0
    #         total = 0
    #         total_classify_loss = 0.0
    #         total_regularization_loss = 0.0
    #
    #         for batch_idx, (data, target) in enumerate(train_loader):
    #             if self.use_gpu:
    #                 data, target = data.cuda(), target.cuda()
    #
    #             optimizer.zero_grad()
    #             output = self.classifier(data)
    #
    #             loss = criterion(output, target)
    #             total_classify_loss += loss.item()
    #             # if self.nn_para['use_regularization_loss']:
    #             #     regularization_loss = self.get_regularization_loss(self.classifier)
    #             #     total_regularization_loss += regularization_loss
    #             #     loss += regularization_loss
    #
    #             loss.backward()
    #             optimizer.step()
    #
    #             _, predict_label = torch.max(output, 1)
    #             correct += (predict_label.cpu() == target.cpu()).sum().numpy()
    #             total += target.size(0)
    #
    #             if self.verbose >= 2 and batch_idx % 100 == 0:
    #                 logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #                     epoch, batch_idx * len(data), len(train_loader.dataset),
    #                            100. * batch_idx / len(train_loader), loss.item()))
    #
    #         train_accuracy = correct / total
    #         if self.verbose >= 1:
    #             if self.nn_para['use_regularization_loss']:
    #                 logging.info(
    #                     "Train Epoch: {}/{} classify loss = {:.6f} Regu loss = {:.6f} Total = {:.6f} accuracy={:.4f}%".format(
    #                         epoch + 1, self.nn_para['num_epochs'],
    #                         total_classify_loss, total_regularization_loss,
    #                         total_classify_loss + total_regularization_loss,
    #                         100.0 * train_accuracy))
    #             else:
    #                 logging.info(
    #                     "Train Epoch: {}/{} Total classify loss = {:.6f} accuracy={:.4f}%".format(epoch + 1,
    #                                                                                               self.nn_para[
    #                                                                                                   'num_epochs'],
    #                                                                                               total_classify_loss,
    #                                                                                               100.0 * train_accuracy))
    #         if self.nn_para['use_scheduler']:
    #             scheduler.step(total_classify_loss)
