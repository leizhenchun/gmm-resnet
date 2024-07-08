# import logging
#
# import numpy
# import numpy as np
#
# from experiment.model.as_gmm import GMM
# from experiment.util.as_util import load_array_h5
#
#
# def asvspoof_gmm_diag_em(dataList, nmix, final_niter, ds_factor=1, nworkers=10, verbose=1):
#     gmm_all = []
#
#     ispow2 = np.log2(nmix)
#     if not ispow2 == 0.5:
#         raise ValueError('oh dear! nmix should be a power of two!')
#
#     dataList = load_data(dataList)
#
#     nfiles = len(dataList)
#
#     logging.info('Initializing the GMM hyperparameters ...')
#
#     [gm, gv] = comp_gm_gv(dataList)
#     gmm = gmm_init(gm, gv)
#
#     gmm_all.append(gmm)
#
#     # gradually increase the number of iterations per binary split
#     # mix = [1 2 4 8 16 32 64 128 256 512 1024]
#     # niter = [1 2 4 4  4  4  6  6   10  10  15]
#     niter = [1, 2, 4, 4, 4, 4, 6, 6, 10, 10, 15]
#
#     # niter(log2(nmix) + 1) = final_niter
#
#     gmm = gmm_mixup(gmm)
#
#     mix = 2
#     # while (mix <= nmix)
#     #     if (mix >= nmix / 2), ds_factor = 1 end % not for the last two splits!
#     logging.info('Re-estimating the GMM hyperparameters for {} components ...'.format(mix))
#
#     for iter in range(20):  # 1: 20 % niter(log2(mix) + 1)
#         logging.info('EM iter#: %d \t'.format(iter))
#
#         N = np.zeros((nmix,), dtype=np.float64)
#         F = np.zeros((nmix,), dtype=np.float64)
#         S = np.zeros((nmix,), dtype=np.float64)
#
#         L = 0
#         nframes = 0
#         # tim = tic
#
#         for ix in range(len(nfiles)):
#             [n, f, s, l] = expectation(dataList[ix][:, 1: ds_factor:], gmm)
#             N = N + n
#             F = F + f
#             S = S + s
#
#             L = L + np.sum(l)
#             nframes = nframes + len(l)
#
#         # tim = toc(tim)
#         # fprintf('[llk = %.2f] \t [elaps = %.2f s]\n', L / nframes, tim)
#         logging.info('[llk = ')
#         gmm = maximization(N, F, S)
#
#     gmm.type = 1
#     gmm_all.append(gmm)
#
#     if mix < nmix:
#         gmm = gmm_mixup(gmm)
#
#     mix = mix * 2
#
#
# def comp_gm_gv(data):
#     # computes the global mean and variance of data
#
#     nframes = 0
#     gm = np.zeros((data[0].shape[0],), dtype=np.float64)
#     gv = np.zeros((data[0].shape[0],), dtype=np.float64)
#     for x in data:
#         nframes += x.shape[0]
#         gm += np.sum(x, 1)
#         gv += np.sum(x * x, 1)
#
#     gm = gm / nframes
#     gv = numpy.sqrt(gv / nframes - gm * gm)
#
#     return gm, gv
#
#
# def gmm_init(glob_mu, glob_sigma):
#     # initialize the GMM hyperparameters (Mu, Sigma, and W)
#     gmm = GMM()
#     gmm.mu = glob_mu
#     gmm.sigma = glob_sigma
#     gmm.w = [1, ]
#
#     return gmm
#
#
# def expectation(data, gmm):
#     # compute the sufficient statistics
#     [post, llk] = postprob(data, gmm)
#     N = np.sum(post, 2)
#     F = data * post.T
#     S = data * data * post.T
#     return N, F, S, llk
#
#
# def postprob(data, gmm):
#     # compute the posterior probability of mixtures for each frame
#     post = lgmmprob(data, gmm)
#     llk = logsumexp(post, 1)
#     post = np.exp(post - llk)
#     return post, llk
#
#
# def lgmmprob(data, gmm):
#     # compute the log probability of observations given the GMM
#     ndim = data.shape[0]
#     w = gmm.w
#     mu = gmm.mu
#     sigma = gmm.sigma
#
#     C = sum(mu * mu / sigma) + sum(np.log(sigma))
#     D1 = (1 / sigma).T * (data * data)
#     D2 = 2 * (mu / sigma).T * data
#     D = (1 / sigma).T * (data * data) - 2 * (mu / sigma).T * data + ndim * np.log(2 * np.pi)
#     logprob = -0.5 * (C.T + D)
#     logprob = logprob + np.log(w)
#     return logprob
#
#
# def logsumexp(x, dim):
#     # compute log(sum(exp(x),dim)) while avoiding numerical underflow
#     xmax = np.max(x, dim)
#     y = xmax + np.log(sum(np.exp(x - xmax)), dim)
#     # ind  = find(~isfinite(xmax))
#     # if ~isempty(ind):
#     #     y(ind) = xmax(ind)
#
#     return y
#
#
# def maximization(N, F, S):
#     # ML re-estimation of GMM hyperparameters which are updated from accumulators
#     w = N / sum(N)
#     mu = F / N
#     sigma = S / N - (mu * mu)
#     sigma = apply_var_floors(w, sigma, 0.1)
#
#     gmm = GMM()
#     gmm.w = w
#     gmm.mu = mu
#     gmm.sigma = sigma
#
#     return gmm
#
#
# def apply_var_floors(w, sigma, floor_const):
#     # set a floor on covariances based on a weighted average of component variances
#     vFloor = sigma * w.T * floor_const
#     sigma = np.max(sigma, vFloor)
#     # sigma = bsxfun(@plus, sigma, 1e-6 * ones(size(sigma, 1), 1))
#     return sigma
#
#
# def gmm_mixup(gmm):
#     # perform a binary split of the GMM hyperparameters
#     mu = gmm.mu
#     sigma = gmm.sigma
#     w = gmm.w
#     [ndim, nmix] = sigma.shape[0]
#     [sig_max, arg_max] = np.max(sigma)
#     eps = np.sparse(0 * mu)
#     #    eps(np.sub2ind([ndim, nmix], arg_max, 1 : nmix)) = np.sqrt(sig_max)
#     # only perturb means associated with the max std along each dim
#     mu = [mu - eps, mu + eps]
#     # mu = [mu - 0.2 * eps, mu + 0.2 * eps] % HTK style
#     sigma = [sigma, sigma]
#     w = [w, w] * 0.5
#     gmm.w = w
#     gmm.mu = mu
#     gmm.sigma = sigma
#
#     return gmm
#
#
# def load_data(datalist):
#     # load all data into memory
#     data = []
#     for idx in range(len(datalist)):
#         data.append(load_array_h5(datalist[idx]))
#
#     return data
