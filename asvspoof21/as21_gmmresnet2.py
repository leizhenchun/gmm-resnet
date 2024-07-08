########################################
#        Zhenchun Lei
#  zhenchun.lei@hotmail.com
########################################
import logging

import torch
import torchinfo
import torchsummary
from thop import profile
from torch import nn, Tensor
from torch.nn import BatchNorm1d

from asvspoof21.experiment.as19_experiment import get_parameter
from asvspoof21.experiment.as21_msgmm_experiment import AS21MultiGMMExperiment
from asvspoof21.model.as_gmm import GMMLayer, GMM, DiagGMMLayer


def exp_parameters():
    exp_param = get_parameter()
    exp_param['asvspoof_root_path'] = '/home/labuser/ssd/lzc/ASVspoof/'
    exp_param['asvspoof_exp_path'] = '/home/labuser/ssd/lzc/ASVspoof2021exp/'

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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, first=False, use_max_pool=False):
        super(ResidualBlock, self).__init__()
        self.first = first

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               groups=groups,
                               bias=False)

        self.bn2 = nn.BatchNorm1d(num_features=out_channels)

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               groups=groups,
                               bias=False)

        if in_channels != out_channels:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             padding=0,
                                             kernel_size=1,
                                             stride=1,
                                             groups=groups,
                                             bias=False)

        else:
            self.downsample = False

        self.use_max_pool = use_max_pool
        if self.use_max_pool:
            self.mp = nn.MaxPool1d(2)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity

        if self.use_max_pool:
            out = self.mp(out)

        return out


class GMMResNet2(nn.Module):
    def __init__(self, gmm, groups=1, group_width=512,
                 gmmlayer_regroup_num=1, gmmlayer_index_trans=False, gmmlayer_shuffle=False) -> None:
        super(GMMResNet2, self).__init__()

        self.groups = groups
        self.group_width = group_width

        self.relu = nn.ReLU()

        self.gmm = gmm
        self.gmm_layer = DiagGMMLayer(self.gmm, requires_grad=False, regroup_num=gmmlayer_regroup_num,
                                  index_trans=gmmlayer_index_trans, shuffle=gmmlayer_shuffle)

        channels = groups * group_width
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels=self.gmm.size(), out_channels=channels, kernel_size=1, stride=1,
                      padding=0, dilation=1, groups=groups, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

        self.block1 = ResidualBlock(in_channels=channels, out_channels=channels, groups=groups, first=True)
        self.block2 = ResidualBlock(in_channels=channels, out_channels=channels, groups=groups)
        self.block3 = ResidualBlock(in_channels=channels, out_channels=channels, groups=groups)
        self.block4 = ResidualBlock(in_channels=channels, out_channels=channels, groups=groups)
        self.block5 = ResidualBlock(in_channels=channels, out_channels=channels, groups=groups)
        self.block6 = ResidualBlock(in_channels=channels, out_channels=channels, groups=groups)

        self.bn1 = BatchNorm1d(channels)
        self.bn2 = BatchNorm1d(channels)
        self.bn3 = BatchNorm1d(channels)
        self.bn4 = BatchNorm1d(channels)
        self.bn5 = BatchNorm1d(channels)
        self.bn6 = BatchNorm1d(channels)

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.output_size = channels * 6

        self.sub_classifiers = nn.Conv1d(self.output_size, self.groups * 2, kernel_size=1, groups=self.groups)

    def forward(self, x: Tensor) -> Tensor:

        x = self.gmm_layer(x)

        x = self.stem(x)

        y = []

        x = self.block1(x)
        y.append(self.pool(self.relu(self.bn1(x))))

        x = self.block2(x)
        y.append(self.pool(self.relu(self.bn2(x))))

        x = self.block3(x)
        y.append(self.pool(self.relu(self.bn3(x))))

        x = self.block4(x)
        y.append(self.pool(self.relu(self.bn4(x))))

        x = self.block5(x)
        y.append(self.pool(self.relu(self.bn5(x))))

        x = self.block6(x)
        y.append(self.pool(self.relu(self.bn6(x))))

        z = []
        for g in range(self.groups):
            for y_i in y:
                z.append(y_i[:, g * self.group_width:(g + 1) * self.group_width])
        z = torch.cat(z, dim=1)

        z = self.sub_classifiers(z)
        z = z.squeeze(2)
        z = torch.split(z, 2, dim=1)
        result = list(z)

        z_assembly = sum(z) / len(z)
        result.append(z_assembly)

        if self.training:
            return result
        else:
            return z_assembly


class AS21GMMResNet2Experiment(AS21MultiGMMExperiment):
    def __init__(self, model_type, feature_type, access_type, parm):
        super(AS21GMMResNet2Experiment, self).__init__(model_type, feature_type, access_type, parm=parm)

        self.gmm = GMM.concatenate_gmms((self.gmm_1024, self.gmm_512, self.gmm_256, self.gmm_128, self.gmm_64),
                                        groups=self.parm['groups'])

    def get_net(self, num_classes=2):
        if self.model_type == 'GMMResNet2':
            model = GMMResNet2(self.gmm,
                               groups=self.parm['groups'],
                               group_width=self.parm['group_width'],
                               gmmlayer_regroup_num=self.parm['gmmlayer_regroup_num'],
                               gmmlayer_index_trans=self.parm['gmmlayer_index_trans'],
                               gmmlayer_shuffle=self.parm['gmmlayer_shuffle'],
                               )

        input = torch.randn(1, self.parm['feature_size'], self.parm['feature_num'])
        macs, params = profile(model, inputs=(input,))
        logging.info('MACs={} G,    #Params={} M'.format(macs / 1e9, params / 1e6))
        model = model.cuda()

        torchinfo.summary(model, (2, self.parm['feature_size'], self.parm['feature_num']), depth=5)
        torchsummary.summary(model, (self.parm['feature_size'], self.parm['feature_num']))

        # model = torch.compile(model, mode='reduce-overhead')

        return model

    def compute_loss(self, output, target, criterion):

        if isinstance(output, list):
            output_path = output[:-1]
            output_assembly = output[-1]

            # return criterion(output_assembly, target)

            loss_list = [criterion(o, target) for o in output]
            loss = sum(loss_list) / len(loss_list)

        else:
            loss = criterion(output, target)

        return loss


if __name__ == '__main__':
    exp_param = exp_parameters()

    access_type = 'LA'
    feature_type = 'LFCC21NN'

    exp_param['lr'] = 0.0001
    exp_param['min_lr'] = 0.0
    exp_param['weight_decay'] = 0.0

    model_type = 'GMMResNet2'

    exp_param['num_epochs'] = 100

    exp_param['groups'] = 8
    exp_param['group_width'] = 256

    exp_param['gmm_index_trans'] = False
    exp_param['gmm_regroup_num'] = exp_param['groups']
    exp_param['gmm_shuffle'] = False

    exp_param['data_augmentation'] = ['Original', 'RB4']  # 'Original', 'RB4'
    exp_param['gmm_file_dir'] = r'/home/labuser/ssd/lzc/ASVspoof2021exp/GMM_D_aug2_rb4_LFCC21NN_20240630_204210'

    # torch.set_float32_matmul_precision('high')

    for _ in range(10):
        AS21GMMResNet2Experiment(model_type, feature_type, access_type, parm=exp_param).run()
