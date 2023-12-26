########################################
#        Zhenchun Lei
#  zhenchun.lei@hotmail.com
########################################

# -*- encoding: utf-8 -*-
import logging

import h5py
import numpy
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter


def gmm_index_trans(number):
    data = []
    for i in range(number):
        data.append([i, ])

    # combine
    while len(data) > 1:
        data_len = len(data)
        for i in range(data_len // 2):
            data[i].extend(data[i + data_len // 2])
        data[data_len // 2:] = []

    return data[0]


class GMM:
    def __init__(self, w, mu, sigma, llk_mean=None, llk_std=None,
                 regroup_num=1, index_trans=False, shuffle=False, use_gpu=False, dtype=numpy.float32):
        self.w = w.astype(dtype)
        self.mu = mu.astype(dtype)
        self.sigma = sigma.astype(dtype)
        self.llk_mean = llk_mean.astype(dtype)
        self.llk_std = llk_std.astype(dtype)
        self.index_trans = index_trans
        self.regroup_num = regroup_num
        self.shuffle = shuffle
        self.use_gpu = use_gpu

        if regroup_num > 1:
            logging.info('[GMM] Regroup (group={}) GMM index ......'.format(regroup_num))
            gmm_size = len(w)
            gmm_idx = []
            for g in range(regroup_num):
                group = list(range(g, gmm_size, regroup_num))
                gmm_idx.extend(group)

            self.w = self.w[gmm_idx]
            self.mu = self.mu[gmm_idx, :]
            self.sigma = self.sigma[gmm_idx, :]
            self.llk_mean = self.llk_mean[gmm_idx, :]
            self.llk_std = self.llk_std[gmm_idx, :]

        if index_trans:
            logging.info('[GMM] GMM index trans  ......')

            gmm_size = len(self.w)
            g_idx = gmm_index_trans(gmm_size)
            self.w = self.w[g_idx]
            self.mu = self.mu[g_idx, :]
            self.sigma = self.sigma[g_idx, :]
            self.llk_mean = self.llk_mean[g_idx, :]
            self.llk_std = self.llk_std[g_idx, :]

        if shuffle:
            logging.info('[GMM] Random permutation GMM index ......')
            gmm_idx = numpy.random.permutation(numpy.arange(len(w)))

            self.w = self.w[g_idx]
            self.mu = self.mu[gmm_idx, :]
            self.sigma = self.sigma[gmm_idx, :]
            self.llk_mean = self.llk_mean[gmm_idx, :]
            self.llk_std = self.llk_std[gmm_idx, :]

        self.inv_sigma = 1.0 / self.sigma
        self.two_mu_inv_sigma = 2.0 * self.mu / self.sigma

        if self.llk_std is not None:
            self.llk_inv_std = 1.0 / self.llk_std

        if self.use_gpu:
            self.inv_sigma_tensor = torch.from_numpy(self.inv_sigma).cuda()
            self.two_mu_inv_sigma_tensor = torch.from_numpy(self.two_mu_inv_sigma).cuda()

            if self.llk_mean is not None:
                self.llk_mean_tensor = torch.from_numpy(self.llk_mean).cuda()

            if self.llk_std is not None:
                self.llk_inv_std_tensor = torch.from_numpy(self.llk_inv_std).cuda()

    def size(self):
        return len(self.w)

    def compute_llk_np(self, data):
        # logprob = (1. / sigma)' * (data .* data) - 2 * (mu./sigma)' * data;
        feat = numpy.dot(self.inv_sigma, data * data) - numpy.dot(self.two_mu_inv_sigma, data)
        feat = (feat - self.llk_mean) * self.llk_inv_std
        return feat

    def compute_llk(self, data):
        # logprob = (1. / sigma)' * (data .* data) - 2 * (mu./sigma)' * data;
        feat = torch.matmul(self.inv_sigma_tensor, data * data) - torch.matmul(self.two_mu_inv_sigma_tensor, data)
        feat = (feat - self.llk_mean_tensor) * self.llk_inv_std_tensor
        return feat

    @staticmethod
    def compute_llk_2path_np(data, gmm_spoof, gmm_bonafide, axis=0):
        feat1 = gmm_spoof.compute_llk_np(data)
        feat2 = gmm_bonafide.compute_llk_np(data)
        return numpy.concatenate((feat1, feat2), axis=axis)

    @staticmethod
    def compute_llk_2path(data, gmm_spoof, gmm_bonafide, dim=1):
        feat1 = gmm_spoof.compute_llk(data)
        feat2 = gmm_bonafide.compute_llk(data)
        return torch.cat((feat1, feat2), dim=dim)

    @staticmethod
    def compute_llk_mpath_np(data, gmms, axis=0):
        gmmllks = []
        for idx in range(len(gmms)):
            gmmllks.append(gmms[idx].compute_llk_np(data))

        return numpy.concatenate(gmmllks, axis=axis)

    @staticmethod
    def compute_llk_mpath(data, gmms, dim=1):
        gmmllks = []
        for idx in range(len(gmms)):
            gmmllks.append(gmms[idx].compute_llk(data))

        return torch.cat(gmmllks, dim=dim)

    @staticmethod
    def load_gmm(gmm_file, regroup_num=1, index_trans=False, shuffle=False, use_gpu=False, dtype=numpy.float32):
        logging.info('[AS19GMMLLKExperiment] Loading GMM and mean std from : ' + gmm_file)
        with h5py.File(gmm_file, 'r') as f:
            gmm = GMM(f['w'][:],
                      f['mu'][:],
                      f['sigma'][:],
                      llk_mean=f['feat_mean'][:],
                      llk_std=f['feat_std'][:],
                      regroup_num=regroup_num,
                      index_trans=index_trans,
                      shuffle=shuffle,
                      use_gpu=use_gpu,
                      dtype=dtype)

        return gmm

    @staticmethod
    def combine_gmms(gmm1, gmm2):
        gmm = GMM(numpy.concatenate((gmm1.w, gmm2.w), axis=0) / 2.0,
                  numpy.concatenate((gmm1.mu, gmm2.mu), axis=0),
                  numpy.concatenate((gmm1.sigma, gmm2.sigma), axis=0),
                  numpy.concatenate((gmm1.llk_mean, gmm2.llk_mean), axis=0),
                  numpy.concatenate((gmm1.llk_std, gmm2.llk_std), axis=0))
        return gmm

    @staticmethod
    def concatenate_gmms(gmms, groups=1):
        gmm_num = len(gmms)

        w = []
        mu = []
        sigma = []
        llk_mean = []
        llk_std = []

        for i in range(groups):
            for j in range(gmm_num):
                gmm = gmms[j]
                group_size = len(gmm.w) // groups
                w.append(gmm.w[i * group_size:(i + 1) * group_size])
                mu.append(gmm.mu[i * group_size:(i + 1) * group_size])
                sigma.append(gmm.sigma[i * group_size:(i + 1) * group_size])
                llk_mean.append(gmm.llk_mean[i * group_size:(i + 1) * group_size])
                llk_std.append(gmm.llk_std[i * group_size:(i + 1) * group_size])

        gmm = GMM(numpy.concatenate(w, axis=0) / gmm_num,
                  numpy.concatenate(mu, axis=0),
                  numpy.concatenate(sigma, axis=0),
                  numpy.concatenate(llk_mean, axis=0),
                  numpy.concatenate(llk_std, axis=0))
        return gmm


class GMMLayer(nn.Module):
    def __init__(self, gmm, requires_grad=False, regroup_num=0, index_trans=False, shuffle=False):
        super().__init__()
        torch._C._log_api_usage_once("model.gmmlayer")

        self.channels = len(gmm.w)
        self.feature_size = gmm.mu.shape[1]
        self.requires_grad = requires_grad
        self.regroup_num = regroup_num
        self.index_trans = index_trans
        self.shuffle = shuffle

        self.mu = torch.from_numpy(gmm.mu)
        self.sigma = torch.from_numpy(gmm.sigma)
        self.llk_mean = torch.from_numpy(gmm.llk_mean)
        self.llk_std = torch.from_numpy(gmm.llk_std)

        if regroup_num > 1:
            logging.info('[GMMLayer] Regroup (group={}) GMM index ......'.format(regroup_num))
            gmm_size = self.mu.shape[0]
            gmm_idx = []
            for g in range(regroup_num):
                group = list(range(g, gmm_size, regroup_num))
                gmm_idx.extend(group)

            self.mu = self.mu[gmm_idx, :]
            self.sigma = self.sigma[gmm_idx, :]
            self.llk_mean = self.llk_mean[gmm_idx, :]
            self.llk_std = self.llk_std[gmm_idx, :]

        if index_trans:
            logging.info('[GMMLayer] GMM index trans  ......')

            gmm_idx = gmm_index_trans(self.mu.shape[0])

            self.mu = self.mu[gmm_idx, :]
            self.sigma = self.sigma[gmm_idx, :]
            self.llk_mean = self.llk_mean[gmm_idx, :]
            self.llk_std = self.llk_std[gmm_idx, :]

        if shuffle:
            logging.info('[GMMLayer] Random permutation GMM index ......')
            gmm_idx = numpy.random.permutation(numpy.arange(gmm.size()))

            self.mu = self.mu[gmm_idx, :]
            self.sigma = self.sigma[gmm_idx, :]
            self.llk_mean = self.llk_mean[gmm_idx, :]
            self.llk_std = self.llk_std[gmm_idx, :]

        self.llk_mean = Parameter(self.llk_mean, requires_grad=requires_grad)
        self.llk_inv_std = Parameter(1.0 / self.llk_std, requires_grad=requires_grad)
        self.inv_sigma = Parameter(1.0 / self.sigma, requires_grad=requires_grad)
        self.two_mu_inv_sigma = Parameter(2.0 * self.mu / self.sigma, requires_grad=requires_grad)

    def extra_repr(self):
        s = ('{channels}, {feature_size}, requires_grad={requires_grad}'
             ', shuffle={shuffle}')
        if self.regroup_num > 1:
            s += ', regroup_num={regroup_num}'
        if self.index_trans:
            s += ', index_trans={index_trans}'
        return s.format(**self.__dict__)

    def forward(self, x):
        # x = torch.matmul(self.inv_sigma, x * x) - torch.matmul(self.two_mu_inv_sigma, x)
        x1 = torch.matmul(self.inv_sigma, x * x)
        x2 = torch.matmul(self.two_mu_inv_sigma, x)
        x = x1 - x2

        x = (x - self.llk_mean) * self.llk_inv_std
        return x


class GMMLayerNoNorm(nn.Module):
    def __init__(self, gmm, requires_grad=False, regroup_num=0, shuffle=False):
        super().__init__()
        torch._C._log_api_usage_once("model.gmmlayer")

        self.mu = torch.from_numpy(gmm.mu)
        self.sigma = torch.from_numpy(gmm.sigma)

        if shuffle:
            logging.info('random permutation GMM index ......')
            gmm_idx = numpy.random.permutation(numpy.arange(gmm.size()))

            self.mu = self.mu[gmm_idx, :]
            self.sigma = self.sigma[gmm_idx, :]

        if regroup_num > 1:
            logging.info('Regroup (group={}) GMM index ......'.format(regroup_num))
            gmm_size = self.mu.shape[0]
            gmm_idx = []
            for g in range(regroup_num):
                group = list(range(g, gmm_size, regroup_num))
                gmm_idx.extend(group)

            self.mu = self.mu[gmm_idx, :]
            self.sigma = self.sigma[gmm_idx, :]

        self.inv_sigma = Parameter(1.0 / self.sigma, requires_grad=requires_grad)
        self.two_mu_inv_sigma = Parameter(2.0 * self.mu / self.sigma, requires_grad=requires_grad)

    def forward(self, x):
        # x = torch.matmul(self.inv_sigma, x * x) - torch.matmul(self.two_mu_inv_sigma, x)
        x1 = torch.matmul(self.inv_sigma, x * x)
        x2 = torch.matmul(self.two_mu_inv_sigma, x)

        return x1 - x2


class GaussianIndexMasking(nn.Module):
    def __init__(self, gaussian_mask_param) -> None:
        super(GaussianIndexMasking, self).__init__()
        self.gaussian_mask_param = gaussian_mask_param

    def forward(self, x: Tensor, mask_value: float = 0.0) -> Tensor:
        gaussian_num = x.shape[1]
        selected_num = torch.randint(self.gaussian_mask_param,
                                     size=(1,))  # numpy.random.randint(0, self.gaussian_mask_param)  #
        selected = torch.randperm(gaussian_num)[0:selected_num[0]].cuda()
        x[:, selected] = mask_value

        return x


if __name__ == '__main__':
    data = gmm_index_trans(16)
    data = gmm_index_trans(16)
    a = 1
