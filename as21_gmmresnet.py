########################################
#        Zhenchun Lei
#  zhenchun.lei@hotmail.com
########################################

import torch
import torchinfo
import torchsummary
from torch import nn, Tensor

from asvspoof19.as19_experiment import get_parameter
from asvspoof21.as21_experiment import AS21GMMExperiment, AS21GMM2Path2StepExperiment, AS21GMM2PathExperiment
from model.gmm import GMMLayer


def exp_parameters():
    exp_param = get_parameter()
    exp_param['asvspoof_root_path'] = '/home/labuser/ssd/lzc/ASVspoof/'
    exp_param['asvspoof_exp_path'] = '/home/labuser/lzc/ASVspoof2021exp/'

    exp_param['batch_size'] = 32
    exp_param['batch_size_test'] = 128

    exp_param['feature_size'] = 60
    exp_param['feature_num'] = 400
    exp_param['feature_num_test'] = 400
    exp_param['feature_file_extension'] = '.h5'

    exp_param['feature_keep_in_memory'] = True
    exp_param['feature_keep_in_memory_debug'] = False

    exp_param['gmm_index_trans'] = False
    exp_param['gmm_regroup_num'] = 1
    exp_param['gmm_shuffle'] = False

    exp_param['gmmlayer_index_trans'] = False
    exp_param['gmmlayer_regroup_num'] = 1
    exp_param['gmmlayer_shuffle'] = False

    exp_param['weight_decay'] = 0.0

    exp_param['num_epochs'] = 100

    exp_param['lr'] = 0.0001
    exp_param['min_lr'] = 1e-8
    exp_param['use_regularization_loss'] = False
    exp_param['use_scheduler'] = True

    exp_param['test_train2019'] = False
    exp_param['test_dev2019'] = True
    exp_param['test_eval2019'] = True
    exp_param['evaluate_asvspoof2021'] = True
    exp_param['evaluate_asvspoof2021_df'] = True

    exp_param['test_data_basic'] = True
    exp_param['test_data_ufm'] = False
    exp_param['test_data_adaptive'] = False

    return exp_param


class ResNetBlock(nn.Module):
    def __init__(self, in_channels=512, kernel_size=3) -> None:
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2, groups=1, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2, groups=1, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm1d(in_channels)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class GMMResNetPath(nn.Module):
    def __init__(self, gmm) -> None:
        super(GMMResNetPath, self).__init__()

        self.gmm_size = gmm.size()

        self.gmm_layer = GMMLayer(gmm, requires_grad=False)

        channels = gmm.size()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels=self.gmm_size, out_channels=channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

        self.block1 = ResNetBlock(in_channels=channels)
        self.block2 = ResNetBlock(in_channels=channels)
        self.block3 = ResNetBlock(in_channels=channels)
        self.block4 = ResNetBlock(in_channels=channels)
        self.block5 = ResNetBlock(in_channels=channels)
        self.block6 = ResNetBlock(in_channels=channels)

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.output_size = channels

    def forward(self, x: Tensor) -> Tensor:
        x = self.gmm_layer(x)

        x = self.stem(x)

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.block5(x)

        x = self.block6(x)

        result = self.pool(x).squeeze(2)

        return result


class GMMResNet(nn.Module):
    def __init__(self, gmm) -> None:
        super(GMMResNet, self).__init__()

        self.path = GMMResNetPath(gmm)

        self.classifier = nn.Linear(512, 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.path(x)

        return self.classifier(x)


class AS21GMMResNetExperiment(AS21GMMExperiment):
    def __init__(self, model_type, feature_type, access_type, parm):
        super(AS21GMMResNetExperiment, self).__init__(model_type, feature_type, access_type, parm=parm)

    def get_net(self, num_classes=2):
        if self.model_type == 'GMMResNet':
            model = GMMResNet(gmm=self.gmm_ubm)

        model = model.cuda()

        torchinfo.summary(model, (2, self.parm['feature_size'], self.parm['feature_num']), depth=5)
        torchsummary.summary(model, (self.parm['feature_size'], self.parm['feature_num']))

        return model


class GMMResNet2P(nn.Module):
    def __init__(self, gmm_bonafide, gmm_spoof) -> None:
        super(GMMResNet2P, self).__init__()

        self.path1 = GMMResNetPath(gmm_bonafide)

        self.path2 = GMMResNetPath(gmm_spoof)

        self.classifier = nn.Linear(1024, 2)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.path1(x)
        x2 = self.path2(x)

        result = self.classifier(torch.concatenate((x1, x2), dim=1))

        return result


class AS21GMMResNet2PExperiment(AS21GMM2PathExperiment):
    def __init__(self, model_type, feature_type, access_type, parm):
        super(AS21GMMResNet2PExperiment, self).__init__(model_type, feature_type, access_type, parm=parm)

    def get_net(self, num_classes=2):
        if self.model_type == 'GMMResNet_2P':
            model = GMMResNet2P(self.gmm_bonafide, self.gmm_spoof)

        model = model.cuda()
        torchinfo.summary(model, (2, self.parm['feature_size'], self.parm['feature_num']), depth=5)
        torchsummary.summary(model, (self.parm['feature_size'], self.parm['feature_num']))

        return model


class AS21GMMResNet2P2SExperiment(AS21GMM2Path2StepExperiment):
    def __init__(self, model_type, feature_type, access_type, parm):
        super(AS21GMMResNet2P2SExperiment, self).__init__(model_type, feature_type, access_type, parm=parm)

    def get_net(self, num_classes=2):
        if self.model_type == 'GMMResNet_2P2S':
            model = GMMResNet2P(self.gmm_bonafide, self.gmm_spoof)

        model = model.cuda()
        torchinfo.summary(model, (2, self.parm['feature_size'], self.parm['feature_num']), depth=5)
        torchsummary.summary(model, (self.parm['feature_size'], self.parm['feature_num']))

        return model


if __name__ == '__main__':
    exp_param = exp_parameters()

    access_type = 'LA'
    feature_type = 'LFCC21NN'

    exp_param['lr'] = 0.0001
    exp_param['min_lr'] = 0.0
    exp_param['num_epochs'] = 100

    exp_param['data_augmentation'] = ["Original", "RB4"]
    exp_param[
        'gmm_file'] = '/home/labuser/ssd/lzc/ASVspoof/ASVspoof2021exp/GMM_aug2_rb4_{}/ASVspoof2019_GMM_{}_{}_512.h5'.format(
        feature_type, feature_type, access_type)
    exp_param[
        'gmm_spoof_file'] = '/home/labuser/ssd/lzc/ASVspoof/ASVspoof2021exp/GMM_aug2_rb4_{}/ASVspoof2019_GMM_{}_{}_512_spoof.h5'.format(
        feature_type, feature_type, access_type)
    exp_param[
        'gmm_bonafide_file'] = '/home/labuser/ssd/lzc/ASVspoof/ASVspoof2021exp/GMM_aug2_rb4_{}/ASVspoof2019_GMM_{}_{}_512_bonafide.h5'.format(
        feature_type, feature_type, access_type)

    model_type = 'GMMResNet'
    for _ in range(5):
        AS21GMMResNetExperiment(model_type, feature_type, access_type, parm=exp_param).run()

    model_type = 'GMMResNet_2P'
    for _ in range(5):
        AS21GMMResNet2PExperiment(model_type, feature_type, access_type, parm=exp_param).run()

    model_type = 'GMMResNet_2P2S'
    for _ in range(5):
        AS21GMMResNet2P2SExperiment(model_type, feature_type, access_type, parm=exp_param).run()
