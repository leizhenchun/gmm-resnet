########################################
#        Zhenchun Lei
#  zhenchun.lei@hotmail.com
########################################

import logging
import os

from asvspoof21.experiment.as19_experiment import AS19Experiment
from asvspoof21.experiment.as19_experiment_2step import AS192StepExperiment
from asvspoof21.experiment.as21_experiment import AS21Evaluate
from asvspoof21.model.as_gmm import GMM


class AS21MultiGMMExperiment(AS21Evaluate, AS19Experiment):
    def __init__(self, model_type, feature_type, access_type, parm={}):
        super(AS21MultiGMMExperiment, self).__init__(model_type, feature_type, access_type, parm=parm)

        self.gmm_file_dir = self.parm['gmm_file_dir']
        logging.info('[AS21GMMFPNExperiment] Loading All GMMs from : ' + self.gmm_file_dir)

        index_trans = parm['gmm_index_trans']
        regroup_num = parm['gmm_regroup_num']
        shuffle = parm['gmm_shuffle']

        self.gmm_1024 = GMM.load_gmm(os.path.join(self.gmm_file_dir, 'ASVspoof2019_GMM_{}_{}_1024.h5'.format(
            self.feature_type, self.access_type)), regroup_num=regroup_num, index_trans=index_trans, shuffle=shuffle)

        self.gmm_512 = GMM.load_gmm(os.path.join(self.gmm_file_dir, 'ASVspoof2019_GMM_{}_{}_512.h5'.format(
            self.feature_type, self.access_type)), regroup_num=regroup_num, index_trans=index_trans, shuffle=shuffle)

        self.gmm_256 = GMM.load_gmm(os.path.join(self.gmm_file_dir, 'ASVspoof2019_GMM_{}_{}_256.h5'.format(
            self.feature_type, self.access_type)), regroup_num=regroup_num, index_trans=index_trans, shuffle=shuffle)

        self.gmm_128 = GMM.load_gmm(os.path.join(self.gmm_file_dir, 'ASVspoof2019_GMM_{}_{}_128.h5'.format(
            self.feature_type, self.access_type)), regroup_num=regroup_num, index_trans=index_trans, shuffle=shuffle)

        self.gmm_64 = GMM.load_gmm(os.path.join(self.gmm_file_dir, 'ASVspoof2019_GMM_{}_{}_64.h5'.format(
            self.feature_type, self.access_type)), regroup_num=regroup_num, index_trans=index_trans, shuffle=shuffle)


class AS21MultiGMM2Path():
    def __init__(self, model_type, feature_type, access_type, parm={}):
        super(AS21MultiGMM2Path, self).__init__(model_type, feature_type, access_type, parm=parm)

        self.gmm_file_dir = self.parm['gmm_file_dir']
        logging.info('[AS21GMMFPNExperiment] Loading All GMMs from : ' + self.gmm_file_dir)

        index_trans = parm['gmm_index_trans']
        regroup_num = parm['gmm_regroup_num']
        shuffle = parm['gmm_shuffle']

        self.gmm_bonafide_1024 = GMM.load_gmm(
            os.path.join(self.gmm_file_dir, 'ASVspoof2019_GMM_{}_{}_1024_bonafide.h5'.format(
                self.feature_type, self.access_type)), regroup_num=regroup_num, index_trans=index_trans,
            shuffle=shuffle)

        self.gmm_bonafide_512 = GMM.load_gmm(
            os.path.join(self.gmm_file_dir, 'ASVspoof2019_GMM_{}_{}_512_bonafide.h5'.format(
                self.feature_type, self.access_type)), regroup_num=regroup_num, index_trans=index_trans,
            shuffle=shuffle)

        self.gmm_bonafide_256 = GMM.load_gmm(
            os.path.join(self.gmm_file_dir, 'ASVspoof2019_GMM_{}_{}_256_bonafide.h5'.format(
                self.feature_type, self.access_type)), regroup_num=regroup_num, index_trans=index_trans,
            shuffle=shuffle)

        self.gmm_bonafide_128 = GMM.load_gmm(
            os.path.join(self.gmm_file_dir, 'ASVspoof2019_GMM_{}_{}_128_bonafide.h5'.format(
                self.feature_type, self.access_type)), regroup_num=regroup_num, index_trans=index_trans,
            shuffle=shuffle)

        self.gmm_bonafide_64 = GMM.load_gmm(
            os.path.join(self.gmm_file_dir, 'ASVspoof2019_GMM_{}_{}_64_bonafide.h5'.format(
                self.feature_type, self.access_type)), regroup_num=regroup_num, index_trans=index_trans,
            shuffle=shuffle)

        self.gmm_spoof_1024 = GMM.load_gmm(
            os.path.join(self.gmm_file_dir, 'ASVspoof2019_GMM_{}_{}_1024_spoof.h5'.format(
                self.feature_type, self.access_type)), regroup_num=regroup_num, index_trans=index_trans,
            shuffle=shuffle)

        self.gmm_spoof_512 = GMM.load_gmm(
            os.path.join(self.gmm_file_dir, 'ASVspoof2019_GMM_{}_{}_512_spoof.h5'.format(
                self.feature_type, self.access_type)), regroup_num=regroup_num, index_trans=index_trans,
            shuffle=shuffle)

        self.gmm_spoof_256 = GMM.load_gmm(
            os.path.join(self.gmm_file_dir, 'ASVspoof2019_GMM_{}_{}_256_spoof.h5'.format(
                self.feature_type, self.access_type)), regroup_num=regroup_num, index_trans=index_trans,
            shuffle=shuffle)

        self.gmm_spoof_128 = GMM.load_gmm(
            os.path.join(self.gmm_file_dir, 'ASVspoof2019_GMM_{}_{}_128_spoof.h5'.format(
                self.feature_type, self.access_type)), regroup_num=regroup_num, index_trans=index_trans,
            shuffle=shuffle)

        self.gmm_spoof_64 = GMM.load_gmm(
            os.path.join(self.gmm_file_dir, 'ASVspoof2019_GMM_{}_{}_64_spoof.h5'.format(
                self.feature_type, self.access_type)), regroup_num=regroup_num, index_trans=index_trans,
            shuffle=shuffle)


class AS21MultiGMM2PathExperiment(AS21Evaluate, AS21MultiGMM2Path, AS19Experiment):
    def __init__(self, model_type, feature_type, access_type, parm={}):
        super(AS21MultiGMM2PathExperiment, self).__init__(model_type, feature_type, access_type, parm=parm)


class AS21MultiGMM2Path2StepExperiment(AS21Evaluate, AS21MultiGMM2Path, AS192StepExperiment):
    def __init__(self, model_type, feature_type, access_type, parm={}):
        super(AS21MultiGMM2Path2StepExperiment, self).__init__(model_type, feature_type, access_type, parm=parm)
