########################################
#        Zhenchun Lei
#  zhenchun.lei@hotmail.com
########################################

import logging

import h5py
import numpy
import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
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
    def __init__(self, gmmtype, w, mu, sigma, llk_mean=None, llk_std=None,
                 regroup_num=1, index_trans=False, shuffle=False, use_gpu=False, dtype=numpy.float32):
        self.gmmtype = gmmtype
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

        if self.gmmtype == 1:
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
            try:
                gmmtype = f['type'][:]
                gmmtype = gmmtype[0][0]
            except KeyError:
                gmmtype = 1
            # if not gmmtype:
            #     gmmtype = 1

            gmm = GMM(gmmtype,
                      f['w'][:],
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

    def save_gmm(self, gmm_file):
        logging.info('[AS19GMMLLKExperiment] Saving GMM to : ' + gmm_file)
        with h5py.File(gmm_file, 'w') as f:
            f['type'] = self.gmmtype
            f['w'] = self.w
            f['mu'] = self.mu
            f['sigma'] = self.sigma
            f['feat_mean'] = self.llk_mean
            f['feat_std'] = self.llk_std

    @staticmethod
    def combine_gmms(gmm1, gmm2):
        gmm = GMM(gmm1.gmmtype,
                  numpy.concatenate((gmm1.w, gmm2.w), axis=0) / 2.0,
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

        gmm = GMM(gmms[0].gmmtype,
                  numpy.concatenate(w, axis=0) / gmm_num,
                  numpy.concatenate(mu, axis=0),
                  numpy.concatenate(sigma, axis=0),
                  numpy.concatenate(llk_mean, axis=0),
                  numpy.concatenate(llk_std, axis=0))
        return gmm


class GMMLayer(nn.Module):
    def __init__(self, gmm, requires_grad=False, regroup_num=0, index_trans=False, shuffle=False, mv_norm=True):
        super().__init__()
        torch._C._log_api_usage_once("model.gmmlayer")

        self.gmmtype = gmm.gmmtype

        self.n_components = len(gmm.w)
        self.feature_size = gmm.mu.shape[1]
        self.requires_grad = requires_grad
        self.regroup_num = regroup_num
        self.index_trans = index_trans
        self.shuffle = shuffle
        self.mv_norm = mv_norm

        self.mu = torch.from_numpy(gmm.mu)
        self.sigma = torch.from_numpy(gmm.sigma)

        if self.mv_norm:
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

            if self.mv_norm:
                self.llk_mean = self.llk_mean[gmm_idx, :]
                self.llk_std = self.llk_std[gmm_idx, :]

        if index_trans:
            logging.info('[GMMLayer] GMM index trans  ......')

            gmm_idx = gmm_index_trans(self.mu.shape[0])

            self.mu = self.mu[gmm_idx, :]
            self.sigma = self.sigma[gmm_idx, :]

            if self.mv_norm:
                self.llk_mean = self.llk_mean[gmm_idx, :]
                self.llk_std = self.llk_std[gmm_idx, :]

        if shuffle:
            logging.info('[GMMLayer] Random permutation GMM index ......')
            gmm_idx = numpy.random.permutation(numpy.arange(gmm.size()))

            self.mu = self.mu[gmm_idx, :]
            self.sigma = self.sigma[gmm_idx, :]

            if self.mv_norm:
                self.llk_mean = self.llk_mean[gmm_idx, :]
                self.llk_std = self.llk_std[gmm_idx, :]

    def extra_repr(self):
        s = ('{feature_size}, {n_components}, requires_grad={requires_grad}'
             ', shuffle={shuffle}')
        if self.regroup_num > 1:
            s += ', regroup_num={regroup_num}'
        if self.index_trans:
            s += ', index_trans={index_trans}'
        return s.format(**self.__dict__)


class DiagGMMLayer(GMMLayer):
    def __init__(self, gmm, requires_grad=False, regroup_num=0, index_trans=False, shuffle=False):
        self.gmmtype = gmm.gmmtype

        super(DiagGMMLayer, self).__init__(gmm, requires_grad=requires_grad, regroup_num=regroup_num,
                                           index_trans=index_trans, shuffle=shuffle)

        self.inv_sigma = Parameter(1.0 / self.sigma, requires_grad=requires_grad)
        self.two_mu_inv_sigma = Parameter(2.0 * self.mu / self.sigma, requires_grad=requires_grad)

        if self.mv_norm:
            self.llk_mean = Parameter(self.llk_mean, requires_grad=requires_grad)
            self.llk_inv_std = Parameter(1.0 / self.llk_std, requires_grad=requires_grad)

    def forward(self, x):
        # x = torch.matmul(self.inv_sigma, x * x) - torch.matmul(self.two_mu_inv_sigma, x)
        x1 = torch.matmul(self.inv_sigma, x * x)
        x2 = torch.matmul(self.two_mu_inv_sigma, x)
        x = x1 - x2

        if self.mv_norm:
            x = (x - self.llk_mean) * self.llk_inv_std

        return x


class FullGMMLayer(GMMLayer):
    def __init__(self, gmm, requires_grad=False, regroup_num=0, index_trans=False, shuffle=False):
        self.gmmtype = gmm.gmmtype

        super(FullGMMLayer, self).__init__(gmm, requires_grad=requires_grad, regroup_num=regroup_num,
                                           index_trans=index_trans, shuffle=shuffle)

        self.mu = Parameter(self.mu, requires_grad=requires_grad)
        self.precisions_chol = self._compute_precision_cholesky(self.sigma)
        self.precisions_chol = Parameter(self.precisions_chol, requires_grad=requires_grad)

        if self.mv_norm:
            self.llk_mean = Parameter(self.llk_mean, requires_grad=requires_grad)
            self.llk_inv_std = Parameter(1.0 / self.llk_std, requires_grad=requires_grad)

    def forward(self, x):
        # x = x.transpose(1, 2)
        # x = torch.unsqueeze(x, 2)
        # x = x - self.mu
        # x = torch.unsqueeze(x, -2)
        # x = torch.matmul(x, self.precisions_chol)
        # x = x * x
        # x = torch.sum(x, -1)
        # x = x.squeeze(-1)
        # x = x.transpose(1, 2)

        n_batch, n_features, n_samples = x.shape
        x = x.transpose(1, 2)
        log_prob = torch.zeros((n_batch, n_samples, self.n_components), dtype=torch.float32, device=self.mu.device)
        for idx in range(self.n_components):
            y = torch.matmul(x - self.mu[idx], self.precisions_chol[idx])
            log_prob[:, :, idx] = torch.sum(torch.square(y), dim=2)

        x = log_prob.transpose(1, 2)

        if self.mv_norm:
            x = (x - self.llk_mean) * self.llk_inv_std

        return x

    def _compute_precision_cholesky(self, covariances):
        estimate_precision_error_message = (
            "Fitting the mixture model failed because some components have "
            "ill-defined empirical covariance (for instance caused by singleton "
            "or collapsed samples). Try to decrease the number of components, "
            "or increase reg_covar."
        )

        n_components, n_features, _ = covariances.shape
        precisions_chol = np.zeros((n_components, n_features, n_features), dtype=np.float32)
        for k in range(n_components):
            try:
                cov_chol = linalg.cholesky(covariances[k], lower=True)
            except linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol[k] = linalg.solve_triangular(cov_chol, np.eye(n_features), lower=True).T

        return torch.from_numpy(precisions_chol)


# class FullGMMLayer(GMMLayer):
#     def __init__(self, gmm, requires_grad=False, regroup_num=0, index_trans=False, shuffle=False):
#         self.gmmtype = gmm.gmmtype
#
#         super(FullGMMLayer, self).__init__(gmm, requires_grad=requires_grad, regroup_num=regroup_num,
#                                            index_trans=index_trans, shuffle=shuffle)
#         inv_sigma = torch.zeros((self.n_components, self.feature_size, self.feature_size))
#         two_mu_inv_sigma = torch.zeros((self.n_components, self.feature_size, 1))
#
#         for idx in range(self.n_components):
#             inv_sigma[idx] = torch.linalg.inv(self.sigma[idx])
#             two_mu_inv_sigma[idx] = 2.0 * torch.matmul(inv_sigma[idx], self.mu[idx].unsqueeze(-1))
#
#         self.inv_sigma = Parameter(inv_sigma, requires_grad=requires_grad)
#         self.two_mu_inv_sigma = Parameter(two_mu_inv_sigma, requires_grad=requires_grad)
#
#         if self.mv_norm:
#             self.llk_mean = Parameter(self.llk_mean, requires_grad=requires_grad)
#             self.llk_inv_std = Parameter(1.0 / self.llk_std, requires_grad=requires_grad)
#
#     def forward(self, x):
#         x = x.transpose(1, 2).unsqueeze(2).unsqueeze(3)
#         x1 = torch.matmul(x, self.inv_sigma)
#         x1 = torch.matmul(x1, x.transpose(-1, -2))
#         x2 = torch.matmul(x, self.two_mu_inv_sigma)
#         x = x1 - x2
#         x = x.squeeze(-1).squeeze(-1)
#         x = x.transpose(1, 2)
#
#         if self.mv_norm:
#             x = (x - self.llk_mean) * self.llk_inv_std
#
#         return x


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
