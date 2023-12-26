########################################
#        Zhenchun Lei
#  zhenchun.lei@hotmail.com
########################################

import logging
import os

from torch.utils.data import DataLoader

from asvspoof19.as19_experiment import AS19Experiment
from asvspoof19.as19_experiment_2step import AS192StepExperiment
from asvspoof19.as19_gmm_experiment_mpath import AS19GMMMPathExperiment, AS19GMMMPath2StepExperiment
from asvspoof19.as19_util import read_protocol, save_scores, get_num_classes
from asvspoof21.as21_eval_metrics import compute_tDCF_EER21
from model.gmm import GMM
from util.util import save_array_h5


class AS21Evaluate:
    def __init__(self, model_type, feature_type, access_type, parm={}):
        super(AS21Evaluate, self).__init__(model_type, feature_type, access_type, parm=parm)

    def _path_init(self):
        super()._path_init()

        self.asv_score_file_pattern = self.root_path + 'DS_10283_3336/{}/ASVspoof2019_{}_asv_scores/ASVspoof2019.{}.asv.{}.gi.trl.scores.txt'
        self.feat_path19 = self.root_path + 'ASVspoof2019feat/' + self.feature_type + '/{}_{}_{}'
        self.feat_path21 = self.root_path + 'ASVspoof2021feat/' + self.feature_type + '/{}_{}_{}'
        self.key_path21 = self.root_path + 'ASVspoof2021data/{}-keys-stage-1/keys'

        mf = self.model_type + '_' + self.feature_type
        mfa = mf + '_' + self.access_type
        mfal = mfa + '_' + self.label_type

        if self.parm['asvspoof_exp_path']:
            self.exp_path = self.parm['asvspoof_exp_path']
        else:
            self.exp_path = os.path.join(self.root_path, 'ASVspoof2021exp')
        self.exp_path = os.path.join(self.exp_path, mf, self.exp_time + '_' + mfa)

        self.model_file = self.exp_path + '/AS19_' + mfal + '_model.pkl'
        self.score_file19 = self.exp_path + '/AS19_' + mfal + '_{}_score.txt'
        self.output_file19 = self.exp_path + '/AS19_' + mfal + '_{}_nn_output.h5'

        self.score_file21 = self.exp_path + '/AS21_' + mf + '_{}_' + self.label_type + '_score.txt'
        self.output_file21 = self.exp_path + '/AS21_' + mf + '_{}_' + self.label_type + '_nn_output.h5'
        self.score_ufm_file21 = self.exp_path + '/AS21_' + mf + '_{}_' + self.label_type + '_ufm_score.txt'
        self.output_ufm_file21 = self.exp_path + '/AS21_' + mf + '_{}_' + self.label_type + '_ufm_nn_output.h5'
        # self.score_df_ufm_file21 = self.exp_path + '/AS21_' + mf + '_DF_' + self.label_type + '_ufm_score.txt'
        # self.output_df_ufm_file21 = self.exp_path + '/AS21_' + mf + '_ufm_nn_output.h5'

        # self.asv_key_file = os.path.join(self.key_path21, 'ASV/trial_metadata.txt')
        # self.asv_scr_file = os.path.join(self.key_path21, 'ASV/ASVTorch_Kaldi/score.txt')
        # self.cm_key_file = os.path.join(self.key_path21, 'CM/trial_metadata.txt')

    def run(self):
        result = super().run()

        if self.parm['evaluate_asvspoof2021']:
            logging.info('========== ASVspoof 2021 : {} {} {} {} ......'.format(self.model_type, self.feature_type,
                                                                                self.access_type, self.label_type))
            self.evaluate_asvspoof2021(self.model)
            logging.info("========== ASVspoof 2021 Done!")

        if self.parm['evaluate_asvspoof2021_df'] and self.access_type == 'LA':
            self.access_type = 'DF'
            logging.info('========== ASVspoof 2021 DF : {} {} {} {} ......'.format(self.model_type, self.feature_type,
                                                                                   self.access_type, self.label_type))
            self.evaluate_asvspoof2021(self.model)
            self.access_type = 'LA'
            logging.info("========== ASVspoof 2021 DF Done!")

        return result

    def evaluate_asvspoof2021(self, model):
        test_protocol_file = os.path.join(self.root_path,
                                          'ASVspoof2021data/ASVspoof2021_{}_eval'.format(self.access_type),
                                          'ASVspoof2021.{}.cm.eval.trl.txt'.format(self.access_type))
        test_feat_path = self.feat_path21.format(self.feature_type, self.access_type, 'eval')
        # test_feat_path = os.path.join(self.feat_path21.format(self.feature_type, self.access_type, 'eval'), 'Original')
        test_utterance, test_env, test_attack, test_key, _ = read_protocol(test_protocol_file)

        # logging.info('Loading test feature from dir : ' + test_feat_path)

        model.eval()

        if 'test_data_basic' in self.parm and self.parm['test_data_basic']:
            logging.info('Evaluating ......')
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

            output, scores = self.evaluate_data(model, test_loader, get_num_classes(self.label_type))

            if 'save_score' in self.parm and self.parm['save_score']:
                score_fliename = self.score_file21.format(self.access_type)
                logging.info('Saving Scores to:' + score_fliename)
                save_scores(score_fliename, test_utterance, scores)

                compute_tDCF_EER21(score_fliename, self.access_type, self.key_path21.format(self.access_type))

            if 'save_nn_output' in self.parm and self.parm['save_nn_output']:
                output_fliename = self.output_file21.format(self.access_type)
                logging.info('Saving NN Output to:' + output_fliename)
                save_array_h5(output_fliename, output)

        if 'test_data_ufm' in self.parm and self.parm['test_data_ufm']:
            logging.info('Evaluating UFM ......')
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

            test_loader = DataLoader(test_dataset, batch_size=self.parm['batch_size_test'], shuffle=False)
            logging.info('Dataset Length : {}   DataLoader Length : {}'.format(len(test_dataset), len(test_loader)))

            output_ufm, scores_ufm = self.test_data_ufm(model, test_loader,
                                                        get_num_classes(self.label_type),
                                                        self.parm['feature_ufm_length'],
                                                        self.parm['feature_ufm_hop'])

            if 'save_score' in self.parm and self.parm['save_score']:
                score_fliename = self.score_ufm_file21.format(self.access_type)
                logging.info('Saving Scores to:' + score_fliename)
                save_scores(score_fliename, test_utterance, scores_ufm)

                compute_tDCF_EER21(score_fliename, self.access_type, self.key_path21.format(self.access_type))

            if 'save_nn_output' in self.parm and self.parm['save_nn_output']:
                output_filename = self.output_ufm_file21.format(self.access_type)
                logging.info('Saving NN Output to:' + output_filename)
                save_array_h5(output_filename, output_ufm)

        return


class AS21Experiment(AS21Evaluate, AS19Experiment):
    def __init__(self, model_type, feature_type, access_type, parm={}):
        super(AS21Experiment, self).__init__(model_type, feature_type, access_type, parm=parm)


class AS21GMMExperiment(AS21Evaluate, AS19Experiment):
    def __init__(self, model_type, feature_type, access_type, parm={}):
        super(AS21GMMExperiment, self).__init__(model_type, feature_type, access_type, parm=parm)
        self.gmm_ubm = GMM.load_gmm(self.parm['gmm_file'])


class AS21GMM2PathExperiment(AS21Evaluate, AS19Experiment):
    def __init__(self, model_type, feature_type, access_type, parm={}):
        super(AS21GMM2PathExperiment, self).__init__(model_type, feature_type, access_type, parm=parm)
        self.gmm_spoof = GMM.load_gmm(self.parm['gmm_spoof_file'])
        self.gmm_bonafide = GMM.load_gmm(self.parm['gmm_bonafide_file'])


class AS21GMM2Path2StepExperiment(AS21Evaluate, AS192StepExperiment):
    def __init__(self, model_type, feature_type, access_type, parm={}):
        super(AS21GMM2Path2StepExperiment, self).__init__(model_type, feature_type, access_type, parm=parm)
        self.gmm_spoof = GMM.load_gmm(self.parm['gmm_spoof_file'])
        self.gmm_bonafide = GMM.load_gmm(self.parm['gmm_bonafide_file'])


class AS21GMMMPathExperiment(AS21Evaluate, AS19GMMMPathExperiment):
    def __init__(self, model_type, feature_type, access_type, parm={}):
        super(AS21GMMMPathExperiment, self).__init__(model_type, feature_type, access_type, parm=parm)
        self.gmm_ubm = GMM.load_gmm(self.parm['gmm_file'])


class AS21GMMMPath2StepExperiment(AS21Evaluate, AS19GMMMPath2StepExperiment):
    def __init__(self, model_type, feature_type, access_type, parm={}):
        super(AS21GMMMPath2StepExperiment, self).__init__(model_type, feature_type, access_type, parm=parm)
        self.gmm_ubm = GMM.load_gmm(self.parm['gmm_file'])
