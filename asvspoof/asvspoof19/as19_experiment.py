########################################
#        Zhenchun Lei
#  zhenchun.lei@hotmail.com
########################################

import logging
import os
import sys
import time
from collections import OrderedDict
from shutil import copy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from asvspoof19 import as19_util
from asvspoof19.as19_dataset import get_augment_dataset, ASVSpoofDataset
from asvspoof19.as19_eval_metrics import compute_tDCF_EER
from asvspoof19.as19_util import read_protocol, save_scores, get_label_id_two, get_num_classes
from util.logging_util import logger_init_basic, MyLogger
from util.util import make_dir, is_debug, show_model, save_array_h5


def get_parameter():
    params = {}

    params['asvspoof_root_path'] = '/home/lzc/lzc/ASVspoof/'
    params['asvspoof_exp_path'] = '/home/labuser/lzc/ASVspoof2021exp/'
    params['dtype'] = np.float32

    params['batch_size'] = 32
    params['batch_size_test'] = 32

    params['feature_num'] = 400
    params['feature_num_test'] = 400
    params['feature_transpose'] = False
    params['feature_file_extension'] = '.h5'

    params['feature_keep_in_memory'] = False
    params['feature_keep_in_memory_debug'] = False

    params['dataloader_num_workers'] = 0
    params['dataloader_prefetch_factor'] = None
    params['dataloader_persistent_workers'] = False

    params['data_augmentation'] = []  # ['Original']
    params['groups'] = 1

    params['lr'] = 0.001
    params['min_lr'] = 1e-7
    params['weight_decay'] = 0.0
    params['use_regularization_loss'] = False
    params['use_scheduler'] = True
    params['num_epochs'] = 100
    params['validate_dev_epoch'] = False

    params['train_model'] = True
    params['test_train2019'] = False
    params['test_dev2019'] = True
    params['test_eval2019'] = True
    params['evaluate_asvspoof2021'] = True
    params['evaluate_asvspoof2021_df'] = True

    params['sample_ratio'] = 1.0

    params['train_data_one_file'] = False
    params['train_data_ufm'] = False

    params['test_data_basic'] = True
    params['test_data_ufm'] = False
    params['test_data_adaptive'] = False

    params['save_model'] = False
    params['save_score'] = True
    params['save_nn_output'] = False
    params['verbose'] = 1

    return params


def parameter_2_str(parameter):
    p = OrderedDict(parameter)
    result = 'nn_parameter:\n'
    for key in p:
        result += '{} : {}\n'.format(key, p[key])
    return result


class AS19Experiment:
    def __init__(self, model_type, feature_type, access_type, parm=get_parameter()):
        label_type = 'KEY'

        self.model_type = model_type
        self.feature_type = feature_type
        self.access_type = access_type
        self.label_type = label_type
        self.verbose = parm['verbose']
        self.parm = parm

        self.num_epochs = parm['num_epochs']
        self.use_scheduler = parm['use_scheduler']
        self.validate_dev_epoch = parm['validate_dev_epoch']

        self.use_gpu = torch.cuda.is_available()
        self.is_debug = is_debug()

        self.dataset_cls_train = ASVSpoofDataset
        self.dataset_cls_test = ASVSpoofDataset

        self.feat_transformer_fn = None
        self.dtype = parm['dtype']
        if self.dtype == np.float64:
            torch.set_default_dtype(torch.float64)

        self.exp_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))

        self._path_init()

        make_dir(self.exp_path)
        copy(sys.argv[0], self.exp_path)

        if self.is_debug:
            logger_init_basic()
        else:
            sys.stdout = MyLogger(self.exp_path, '{}_{}_{}_log.txt'.format(model_type, feature_type, access_type))
        logging.info('\n\n======== {} {} {} {} ========'.format(self.model_type, self.feature_type, self.access_type,
                                                                self.label_type))
        logging.info(parameter_2_str(self.parm))

    def _path_init(self):
        self.root_path = self.parm['asvspoof_root_path']

        self.feat_path19 = self.root_path + 'ASVspoof2019feat/' + self.feature_type + '/{}_{}_{}'
        self.ds19_path = os.path.join(self.root_path, 'DS_10283_3336')

        mf = self.model_type + '_' + self.feature_type
        mfa = mf + '_' + self.access_type
        mfal = mfa + '_' + self.label_type

        if self.parm['asvspoof_exp_path']:
            self.exp_path = self.parm['asvspoof_exp_path']
        else:
            self.exp_path = os.path.join(self.root_path, 'ASVspoof2019exp')
        self.exp_path = os.path.join(self.exp_path, mf, self.exp_time + '_' + mfa)

        self.model_file = self.exp_path + '/AS19_' + mfal + '_model.pkl'
        self.score_file19 = self.exp_path + '/AS19_' + mfal + '_{}_score.txt'
        self.output_file19 = self.exp_path + '/AS19_' + mfal + '_{}_nn_output.h5'

    def run(self):
        result = {}
        if self.parm['train_model']:
            logging.info(
                '========== ASVspoof 2019 Train Model: {} {} {} {} ......'.format(self.model_type, self.feature_type,
                                                                                  self.access_type,
                                                                                  self.label_type))
            self.model = self.train()
            logging.info("========== Done!")

        if self.parm['test_train2019']:
            logging.info(
                '========== ASVspoof 2019 Test TRAIN: {} {} {} {} ......'.format(self.model_type, self.feature_type,
                                                                                 self.access_type,
                                                                                 self.label_type))
            train_score = self.test(self.model, 'train')
            result['train_score'] = train_score
            logging.info("========== Done!")

        if self.parm['test_dev2019']:
            logging.info(
                '========== ASVspoof 2019 Test DEV: {} {} {} {} ......'.format(self.model_type, self.feature_type,
                                                                               self.access_type,
                                                                               self.label_type))
            dev_eer, dev_min_tDCF = self.test(self.model, 'dev')
            result['dev_eer'] = dev_eer
            result['dev_min_tDCF'] = dev_min_tDCF
            logging.info("========== Done!")

        if self.parm['test_eval2019']:
            logging.info(
                '========== ASVspoof 2019 Test EVAL: {} {} {} {} ......'.format(self.model_type, self.feature_type,
                                                                                self.access_type,
                                                                                self.label_type))
            eval_eer, eval_min_tDCF = self.test(self.model, 'eval')
            result['eval_eer'] = eval_eer
            result['eval_min_tDCF'] = eval_min_tDCF
            logging.info("========== Done!")

        return result

    def get_label_id(self, label_type, env, attack, key):
        return get_label_id_two(label_type, env, attack, key)

    def train(self):
        train_protocol_file = os.path.join(self.root_path, 'DS_10283_3336', self.access_type,
                                           'ASVspoof2019_{}_cm_protocols'.format(self.access_type),
                                           'ASVspoof2019.{}.cm.train.trn.txt'.format(self.access_type))
        train_spk, train_utterance, train_env, train_attack, train_key = read_protocol(train_protocol_file)
        train_label_id = self.get_label_id(self.label_type, train_env, train_attack, train_key)
        train_feat_path = os.path.join(self.feat_path19.format(self.feature_type, self.access_type, 'train'))

        logging.info("Modeling .......")
        model = self.get_net(len(np.unique(train_label_id)))
        if self.use_gpu:
            model = model.cuda()
        model.train()
        show_model(model)

        if self.num_epochs <= 0:
            return model

        # logging.info('Loading training feature from dir: {}/{}'.format(train_feat_path, self.parm['data_augmentation']))
        in_memory = self.parm['feature_keep_in_memory'] if not self.is_debug else self.parm[
            'feature_keep_in_memory_debug']
        train_dataset = get_augment_dataset(dataset_cls=self.dataset_cls_train,
                                            augment_methods=self.parm['data_augmentation'],
                                            feature_file_path=train_feat_path,
                                            feature_file_list=train_utterance,
                                            feature_file_extension=self.parm['feature_file_extension'],
                                            label_id=train_label_id,
                                            feat_length=self.parm['feature_num'],
                                            num_channels=0,
                                            transpose=self.parm['feature_transpose'],
                                            keep_in_memory=in_memory,
                                            sample_ratio=1.0,
                                            feat_transformer_fn=self.feat_transformer_fn,
                                            dtype=self.dtype,
                                            augment=True)
        # train_dataset = self.dataset_cls_train(feature_file_path=train_feat_path,
        #                                        feature_file_list=train_utterance,
        #                                        feature_file_extension=self.parm['feature_file_extension'],
        #                                        label_id=train_label_id,
        #                                        feat_length=self.parm['feature_num'],
        #                                        num_channels=0,
        #                                        transpose=self.parm['feature_transpose'],
        #                                        keep_in_memory=False,
        #                                        sample_ratio=1.0,
        #                                        feat_transformer_fn=self.feat_transformer_fn,
        #                                        dtype=self.dtype,
        #                                        )
        train_loader = DataLoader(train_dataset, batch_size=self.parm['batch_size'], shuffle=True,
                                  num_workers=self.parm['dataloader_num_workers'],
                                  prefetch_factor=self.parm['dataloader_prefetch_factor'],
                                  persistent_workers=self.parm['dataloader_persistent_workers'], )
        logging.info('Dataset Length : {}   DataLoader Length : {}'.format(len(train_dataset), len(train_loader)))

        self.train_loader_len = len(train_loader)

        self.train_model(model, train_loader)

        if 'save_model' in self.parm and self.parm['save_model']:
            make_dir(self.exp_path)
            logging.info("Saving Model to: " + self.model_file)
            torch.save(model, self.model_file)

        return model

    def train_model(self, model, train_loader):
        self.optimizer = optimizer = self.get_optimizer(model)
        self.scheduler = scheduler = self.get_scheduler(optimizer)
        self.criterion = criterion = self.get_criterion()
        if self.use_gpu:
            criterion = criterion.cuda()
        logging.info('Optimizer : ' + str(optimizer))
        logging.info('Scheduler : ' + str(scheduler))
        logging.info('Criterion : ' + str(criterion))

        model.train()

        if self.validate_dev_epoch:
            dev_protocol_file = os.path.join(self.root_path, 'DS_10283_3336', self.access_type,
                                             'ASVspoof2019_{}_cm_protocols'.format(self.access_type),
                                             'ASVspoof2019.{}.cm.{}.trl.txt'.format(self.access_type, 'dev'))
            dev_feat_path = os.path.join(self.feat_path19.format(self.feature_type, self.access_type, 'dev'))
            dev_spk, dev_utterance, dev_env, dev_attack, dev_key = read_protocol(dev_protocol_file)
            dev_label_id = self.get_label_id(self.label_type, dev_env, dev_attack, dev_key)

            # if self.parm['data_augmentation']: dev_feat_path = os.path.join(dev_feat_path, 'Original')
            logging.info('Loading development feature from dir : ' + dev_feat_path)

            dev_dataset = get_augment_dataset(dataset_cls=self.dataset_cls_train,
                                              augment_methods=['Original', ],
                                              feature_file_path=dev_feat_path,
                                              feature_file_list=dev_utterance,
                                              feature_file_extension=self.parm['feature_file_extension'],
                                              label_id=None,
                                              feat_length=self.parm['feature_num_test'],
                                              num_channels=0,
                                              transpose=self.parm['feature_transpose'],
                                              keep_in_memory=True,
                                              sample_ratio=1.0,
                                              feat_transformer_fn=self.feat_transformer_fn,
                                              dtype=self.dtype, )
            dev_loader = DataLoader(dev_dataset, batch_size=self.parm['batch_size'], shuffle=False)
            logging.info('Dataset Length : {}   DataLoader Length : {}'.format(len(dev_dataset), len(dev_loader)))

            best_dev_eer = 1.
            best_dev_tdcf = 0.05
            # n_swa_update = 0  # number of snapshots of model to use in SWA

        logging.info("Training ......")
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self.train_one_epoch(model, optimizer, criterion, scheduler, train_loader, epoch)

            if self.validate_dev_epoch:
                dev_output, dev_scores = self.test_data(model, dev_loader, get_num_classes(self.label_type),
                                                        dev_label_id)

                dev_eer, dev_min_tDCF = compute_tDCF_EER(dev_scores, self.access_type, 'dev', self.ds19_path)
                logging.info(" dev_EER : {:.5f}, dev_min_tDCF : {:.6f}".format(dev_eer, dev_min_tDCF))

                best_dev_tdcf = min(dev_min_tDCF, best_dev_tdcf)
                if best_dev_eer >= dev_eer:
                    logging.info('best model find at epoch {}'.format(epoch))
                    best_dev_eer = dev_eer
                # torch.save(model.state_dict(),
                #            model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))

                # logging.info("Saving epoch {} for swa".format(epoch))
                # optimizer_swa.update_swa()
                # n_swa_update += 1

    def train_one_epoch(self, model, optimizer, criterion, scheduler, train_loader, epoch):
        clr = [param['lr'] for param in optimizer.param_groups]

        correct = 0
        total = 0
        total_classify_loss = 0.0
        total_regularization_loss = 0.0
        for batch_idx, (data, target, data_idx) in enumerate(train_loader):
            if self.use_gpu:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)

            # if isinstance(output, list):
            #     loss_list = [criterion(o, target) / len(output) for o in output]
            #     loss = sum(loss_list)
            # else:
            #     loss = criterion(output, target)

            loss = self.compute_loss(output, target, criterion)

            total_classify_loss += loss.item()
            # if self.parm['use_regularization_loss']:
            #     regularization_loss = self.get_regularization_loss(model)
            #     total_regularization_loss += regularization_loss
            #     loss += regularization_loss
            #
            loss.backward()
            optimizer.step()

            if isinstance(output, list):
                output = sum(output) / len(output)

            if self.use_gpu:
                output, target = output.cpu(), target.cpu()
            _, predict_label = torch.max(output, 1)
            correct += (predict_label == target).sum().numpy()
            total += target.size(0)

        if self.parm['use_scheduler']:
            self.scheduler_step(scheduler, total_classify_loss)
            # scheduler.step()
            # scheduler.step(total_classify_loss)

        if self.verbose >= 1:
            train_accuracy = correct / total
            if self.parm['use_regularization_loss']:
                logging.info(
                    "Train Epoch: {}/{}  LR={:.8f}  ACC={:.4f}%  LOSS={:.6f} Regu loss={:.6f}".format(
                        epoch + 1, self.num_epochs,
                        max(clr),
                        100.0 * train_accuracy,
                        total_classify_loss, total_regularization_loss,
                        total_classify_loss + total_regularization_loss,
                    ))
            else:
                logging.info(
                    "Train Epoch : {}/{}  LR = {:.8f}  ACC = {:.4f}%  LOSS = {:.6f}".format(epoch + 1, self.num_epochs,
                                                                                            max(clr),
                                                                                            100.0 * train_accuracy,
                                                                                            total_classify_loss,
                                                                                            ))

    def test(self, model, test_type):
        if test_type == 'train':
            test_protocol_file = os.path.join(self.root_path, 'DS_10283_3336', self.access_type,
                                              'ASVspoof2019_{}_cm_protocols'.format(self.access_type),
                                              'ASVspoof2019.{}.cm.{}.trn.txt'.format(self.access_type, test_type))
        else:
            test_protocol_file = os.path.join(self.root_path, 'DS_10283_3336', self.access_type,
                                              'ASVspoof2019_{}_cm_protocols'.format(self.access_type),
                                              'ASVspoof2019.{}.cm.{}.trl.txt'.format(self.access_type, test_type))
        test_feat_path = self.feat_path19.format(self.feature_type, self.access_type, test_type)
        # test_feat_path = os.path.join(self.feat_path19.format(self.feature_type, self.access_type, test_type), 'Original')

        test_spk, test_utterance, test_env, test_attack, test_key = read_protocol(test_protocol_file)
        test_label_id = self.get_label_id(self.label_type, test_env, test_attack, test_key)

        # if self.parm['data_augmentation']: test_feat_path = os.path.join(test_feat_path, 'Original')
        # logging.info('Loading test feature from dir : ' + test_feat_path)

        if model is None:
            model = self.get_net(get_num_classes(self.label_type))
            logging.info('Load model from:' + self.model_file)
            model_info = torch.load(self.model_file)
            model.load_state_dict(model_info)
            logging.info(model)
        model.eval()

        if 'test_data_basic' in self.parm and self.parm['test_data_basic']:
            logging.info('Testing ......')
            test_dataset = self.dataset_cls_test(feature_file_path=test_feat_path,
                                                 feature_file_list=test_utterance,
                                                 feature_file_extension=self.parm['feature_file_extension'],
                                                 label_id=None,
                                                 feat_length=self.parm['feature_num_test'],
                                                 num_channels=0,
                                                 transpose=self.parm['feature_transpose'],
                                                 keep_in_memory=False,
                                                 sample_ratio=1.0,
                                                 feat_transformer_fn=self.feat_transformer_fn,
                                                 dtype=self.dtype,
                                                 )

            test_loader = DataLoader(test_dataset, batch_size=self.parm['batch_size_test'], shuffle=False)
            logging.info('Dataset Length : {}   DataLoader Length : {}'.format(len(test_dataset), len(test_loader)))

            output, scores = self.test_data(model, test_loader, get_num_classes(self.label_type), test_label_id)

            if 'save_score' in self.parm and self.parm['save_score']:
                test_score_file = self.score_file19.format(test_type)
                logging.info('Saving Scores to:' + test_score_file)
                save_scores(test_score_file, test_utterance, scores)

            if 'save_nn_output' in self.parm and self.parm['save_nn_output']:
                test_out_file = self.output_file19.format(test_type)
                logging.info('Saving NN Output to:' + test_out_file)
                save_array_h5(test_out_file, output)

            if test_type != 'train':
                eer, min_tDCF = compute_tDCF_EER(scores, self.access_type, test_type, self.ds19_path)

        if 'test_data_ufm' in self.parm and self.parm['test_data_ufm']:
            logging.info('Testing UFM ......')
            test_dataset = self.dataset_cls_test(feature_file_path=test_feat_path,
                                                 feature_file_list=test_utterance,
                                                 feature_file_extension=self.parm['feature_file_extension'],
                                                 label_id=None,
                                                 feat_length=0,
                                                 num_channels=0,
                                                 transpose=self.parm['feature_transpose'],
                                                 keep_in_memory=False,
                                                 sample_ratio=1.0,
                                                 feat_transformer_fn=self.feat_transformer_fn,
                                                 dtype=self.dtype,
                                                 )
            test_loader = DataLoader(test_dataset, batch_size=self.parm['batch_size_test'], shuffle=False,
                                     collate_fn=as19_util.ufm_collate_fn)
            logging.info('Dataset Length : {}   DataLoader Length : {}'.format(len(test_dataset), len(test_loader)))

            output_ufm, scores_ufm = self.test_data_ufm(model, test_loader, get_num_classes(self.label_type),
                                                        test_label_id,
                                                        self.parm['feature_ufm_length'],
                                                        self.parm['feature_ufm_hop'])

            if 'save_score' in self.parm and self.parm['save_score']:
                test_score_file_ufm = self.score_file19.format(test_type + '_ufm')
                logging.info('Saving Scores to:' + test_score_file_ufm)
                save_scores(test_score_file_ufm, test_utterance, scores_ufm)

            if 'save_nn_output' in self.parm and self.parm['save_nn_output']:
                output_file = self.output_file19.format(test_type + '_ufm')
                logging.info('Saving NN Output to:' + output_file)
                save_array_h5(output_file, output_ufm)

            if test_type != 'train':
                eer, min_tDCF = compute_tDCF_EER(scores_ufm, self.access_type, test_type, self.ds19_path)

        if 'test_data_adaptive' in self.parm and self.parm['test_data_adaptive']:
            logging.info('Testing with adaptive length ......')
            test_dataset = self.dataset_cls_train(feature_file_path=test_feat_path,
                                                  feature_file_list=test_utterance,
                                                  feature_file_extension=self.parm['feature_file_extension'],
                                                  label_id=None,
                                                  feat_length=0,
                                                  num_channels=0,
                                                  transpose=self.parm['feature_transpose'],
                                                  keep_in_memory=False,
                                                  sample_ratio=1.0,
                                                  feat_transformer_fn=self.feat_transformer_fn,
                                                  dtype=self.dtype,
                                                  )
            test_loader = DataLoader(test_dataset, batch_size=self.parm['batch_size_test'], shuffle=False,
                                     collate_fn=as19_util.ufm_collate_fn)
            logging.info('Dataset Length : {}   DataLoader Length : {}'.format(len(test_dataset), len(test_loader)))

            output_adaptive, scores_adaptive = self.test_data(model, test_loader,
                                                              get_num_classes(self.label_type),
                                                              test_label_id)

            if 'save_score' in self.parm and self.parm['save_score']:
                test_score_file_adaptive = self.score_file19.format(test_type + '_adaptive')
                logging.info('Saving Scores to:' + test_score_file_adaptive)
                save_scores(test_score_file_adaptive, test_utterance, scores_adaptive)

            if 'save_nn_output' in self.parm and self.parm['save_nn_output']:
                output_file = self.output_file19.format(test_type + '_adaptive')
                logging.info('Saving NN Output to:' + output_file)
                save_array_h5(output_file, output_adaptive)

            if test_type != 'train':
                eer, min_tDCF = compute_tDCF_EER(scores_adaptive, self.access_type, test_type, self.ds19_path)

        if test_type != 'train':
            return eer, min_tDCF

        return scores

    def test_data(self, model, data_loader, num_classes, label_id):
        nn_output, scores = self.evaluate_data(model, data_loader, num_classes)

        predict_label = np.argmax(nn_output, 1)
        correct = sum(predict_label == label_id)
        logging.info('Test: Accuracy: {}/{} ({:.2f}%)'.format(correct, len(data_loader.dataset),
                                                              100. * correct / len(data_loader.dataset)))
        return nn_output, scores

    def evaluate_data(self, model, data_loader, num_classes):
        test_count = len(data_loader.dataset)
        scores = np.zeros((test_count,))
        nn_output = np.zeros((test_count, num_classes))

        model.eval()
        with torch.no_grad():
            for data, test_idx in tqdm(data_loader):
                if self.use_gpu:
                    data = data.cuda()

                output = model(data)

                if isinstance(output, list):
                    output = sum(output) / len(output)

                if self.use_gpu:
                    output = output.cpu()

                nn_output[test_idx, :] = output.numpy()
                output_arr = F.softmax(output, dim=1).data.numpy()
                scores[test_idx] = output_arr[:, 0]

        return nn_output, scores

    def test_data_ufm(self, model, data_loader, num_classes, label_id, frame_length=400, hop_length=200):
        test_count = len(data_loader.dataset)
        scores = np.zeros((test_count,))
        nn_output = np.zeros((test_count, num_classes))

        model.eval()
        with torch.no_grad():
            for batch_data, batch_test_idx in tqdm(data_loader):
                batch_segments = []
                batch_segment_idxes = []

                for idx in range(len(batch_test_idx)):
                    data = batch_data[idx]
                    test_idx = batch_test_idx[idx]
                    segment = as19_util.segment(data, frame_length, hop_length)
                    batch_segments.append(segment)
                    for j in range(segment.shape[0]): batch_segment_idxes.append(test_idx)
                batch_segments = np.concatenate(batch_segments, axis=0)
                batch_segments = torch.from_numpy(batch_segments)
                batch_segment_idxes = np.array(batch_segment_idxes)

                if self.use_gpu:
                    batch_segments = batch_segments.cuda()

                output = model(batch_segments)

                if isinstance(output, list):
                    output = sum(output) / len(output)

                if self.use_gpu:
                    output = output.cpu()

                output = F.softmax(output, dim=1)
                output = output.numpy()

                for idx in range(len(batch_test_idx)):
                    test_idx = batch_test_idx[idx]
                    nn_output[test_idx, :] = np.mean(output[batch_segment_idxes == test_idx, :], axis=0)
                    scores[test_idx] = nn_output[test_idx, 0]

        predict_label = np.argmax(nn_output, 1)
        correct = sum(predict_label == label_id)
        logging.info('Test: Accuracy: {}/{} ({:.2f}%)'.format(correct, len(data_loader.dataset),
                                                              100. * correct / len(data_loader.dataset)))
        return nn_output, scores

    def get_net(self, num_classes=2):
        return None
        # raise NotImplementedError('The {} model is not implemented!'.format(self.model_type))

    def get_criterion(self):
        return nn.CrossEntropyLoss()
        # return nn.CrossEntropyLoss(weight=torch.FloatTensor([0.9, 0.1]))
        # return nn.BCELoss()

    def get_optimizer(self, model):
        return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.parm['lr'],
                          weight_decay=self.parm['weight_decay'])  # , amsgrad=True
        # return optim.Adam(model.parameters(), lr=self.parm['lr'], weight_decay=self.parm['weight_decay'])

    def get_scheduler(self, optimizer):
        return ReduceLROnPlateau(optimizer, patience=5, factor=0.1, min_lr=self.parm['min_lr'], threshold=1e-6,
                                 verbose=True)

    def scheduler_step(self, scheduler, loss):
        scheduler.step(loss)

    def get_regularization_loss(self, model):
        return 0
        # regularization_loss = 0
        # for param in filter(lambda p: p.requires_grad, model.parameters()):
        #     regularization_loss += torch.sum(abs(param))
        # return 0.001 * regularization_loss

    def compute_loss(self, output, target, criterion):
        if isinstance(output, list):
            loss_list = [criterion(o, target) for o in output]
            loss = sum(loss_list) / len(output)
        else:
            loss = criterion(output, target)

        return loss
