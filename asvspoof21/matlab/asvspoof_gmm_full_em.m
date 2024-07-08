function gmm_all = asvspoof_gmm_full_em(dataList, nmix, final_niter, ds_factor, nworkers, quiet)
% fits a nmix-component Gaussian mixture model (GMM) to data in dataList
% using niter EM iterations per binary split. The process can be
% parallelized in nworkers batches using parfor.
%
% Inputs:
%   - dataList    : ASCII file containing feature file names (1 file per line) 
%					or a cell array containing features (nDim x nFrames). 
%					Feature files must be in uncompressed HTK format.
%   - nmix        : number of Gaussian components (must be a power of 2)
%   - final_iter  : number of EM iterations in the final split
%   - ds_factor   : feature sub-sampling factor (every ds_factor frame)
%   - nworkers    : number of parallel workers
%   - gmmFilename : output GMM file name (optional)
%
% Outputs:
%   - gmm		  : a structure containing the GMM hyperparameters
%					(gmm.mu: means, gmm.sigma: covariances, gmm.w: weights)
%
%
% Omid Sadjadi <s.omid.sadjadi@gmail.com>
% Microsoft Research, Conversational Systems Research Center

gmm_all = {};

if ( nargin <= 3 )
	ds_factor = 1;
end
if ( nargin <= 4 )
	nworkers = feature('numCores');
end
if ( nargin <= 5 )
	quiet = 0;
end

if ischar(nmix), nmix = str2double(nmix); end
if ischar(final_niter), final_niter = str2double(final_niter); end
if ischar(ds_factor), ds_factor = str2double(ds_factor); end
% if ischar(nworkers), nworkers = str2double(nworkers); end

[ispow2, ~] = log2(nmix);
if ( ispow2 ~= 0.5 )
	error('oh dear! nmix should be a power of two!');
end

if ischar(dataList) || isstring(dataList) || iscellstr(dataList)
	dataList = load_data(dataList);
end
if ~iscell(dataList)
	error('Oops! dataList should be a cell array!');
end

nfiles = length(dataList);

% fprintf('\n\nInitializing the GMM hyperparameters ...\n');
if ~quiet, fprintf('\n\nInitializing the GMM hyperparameters ...\n'); end
[gm, gc] = comp_gm_gc(dataList);
gmm = gmm_init(gm, gc); 

gmm = gmm_mixup(gmm);

% gmm_all{1} = gmm;

% gradually increase the number of iterations per binary split
% mix = [1 2 4 8 16 32 64 128 256 512 1024];
% niter = [1 2 4 4  4  4  6  6   10  10  15];
niter = [1 2 4 4  4  4  6  6   10  10  15];

% niter(log2(nmix) + 1) = final_niter;

mix = 2;
while ( mix <= nmix )
	if ( mix >= nmix/2 ), ds_factor = 1; end % not for the last two splits!
%     fprintf('\nRe-estimating the GMM hyperparameters for %d components ...\n', mix);
%     if ~quiet, fprintf('\nRe-estimating the GMM hyperparameters for %d components ...\n', mix); end
    if ~quiet, show_message(['Re-estimating the GMM hyperparameters for ', num2str(mix),' components ...'] ); end

    for iter = 1 : 20  % niter(log2(mix) + 1)
%         fprintf('EM iter#: %d \t', iter);
        if ~quiet, fprintf([datestr(now, '[yyyy-mm-dd HH:MM:SS]: '), 'EM iter#: %d \t'], iter); end
        N = 0; 
        F = 0; 
        S = 0; 
        
        L = 0; 
        nframes = 0;
        tim = tic;
        parfor (ix = 1 : nfiles, nworkers)
%         for ix = 1 : nfiles
            [n, f, s, l] = expectation(dataList{ix}(:, 1:ds_factor:end), gmm);
            N = N + n; 
            F = F + f; 
            S = S + s; 
            
            L = L + sum(l);
			nframes = nframes + length(l);
        end
        tim = toc(tim);
%         fprintf('[llk = %.2f] \t [elaps = %.2f s]\n', L/nframes, tim);
        if ~quiet, fprintf('[llk = %.2f] \t [elaps = %.2f s]\n', L/nframes, tim); end
        
        
        gmm = maximization(N, F, S);
%         gmm = maximization2(N, F, S, dataList, gmm);
    end
    
    gmm.type = 2;
    gmm_all{end + 1} = gmm;
    
    if ( mix < nmix )
        gmm = gmm_mixup(gmm); 
    end
    mix = mix * 2;
end

% if ( nargin == 6 ),
% % 	fprintf('\nSaving GMM to file %s\n', gmmFilename);
% 	% create the path if it does not exist and save the file
% 	path = fileparts(gmmFilename);
% 	if ( exist(path, 'dir')~=7 && ~isempty(path) ), mkdir(path); end
% 	save(gmmFilename, 'gmm');
% end



end

function data = load_data(datalist)
    % load all data into memory
    if ~iscellstr(datalist)
        fid = fopen(datalist, 'rt');
        filenames = textscan(fid, '%s');
        fclose(fid);
        filenames = filenames{1};
    else
        filenames = datalist;
    end
    nfiles = size(filenames, 1);
    data = cell(nfiles, 1);
    for ix = 1 : nfiles
        data{ix} = htkread(filenames{ix});
    end
end

function [gm, gc] = comp_gm_gc(data)
    % computes the global mean and variance of data
    nframes = cellfun(@(x) size(x, 2), data, 'UniformOutput', false);
    nframes = sum(cell2mat(nframes));
    gm = cellfun(@(x) sum(x, 2), data, 'UniformOutput', false);
    gm = sum(cell2mat(gm'), 2)/nframes;
%     gv = cellfun(@(x) sum(bsxfun(@minus, x, gm).^2, 2), data, 'UniformOutput', false);
%     gv = sum(cell2mat(gv'), 2)/( nframes - 1 );
%     
%     gc = diag(gv);
    
    
    gc = zeros(length(gm), length(gm));
    for idx = 1 : length(data)
        x = data{idx};
        
        gc = gc + x * x';
    end
    gc = gc / ( nframes - 1 ) - gm * gm';

end

function gmm = gmm_init(glob_mu, glob_sigma)
    % initialize the GMM hyperparameters (Mu, Sigma, and W)
    gmm.mu    = glob_mu;
    gmm.w     = 1.0;
    
%     sigma_eps = eye(length(glob_mu)) * 1e-6;
    gmm.sigma(:, :, 1) = glob_sigma;
    gmm.prec_chol(:, :, 1) = chol(inv(glob_sigma), 'lower');

end

function [N, F, S, llk] = expectation(data, gmm)
    % compute the sufficient statistics
    [post, llk] = postprob(data, gmm);
    N = sum(post, 2)';
    F = data * post';
%     S = (data .* data) * post';

    nmix = length(gmm.w);
    ndim = size(data, 1);

    S = zeros(ndim, ndim, nmix);
    for idx = 1 : nmix
        data_i = data .* post(idx, :);
        S(:, :, idx) = S(:, :, idx) + data_i * data';
%         for j = 1:size(data, 2)
%             data_j = data(:, j);
%             S(:, :, idx) = S(:, :, idx) + data_j * data_j' * post(idx, j);
%         end
    end
end

function [post, llk] = postprob(data, gmm)
    % compute the posterior probability of mixtures for each frame
    post = lgmmprob(data, gmm);
    llk  = logsumexp(post, 1);
    post = exp(bsxfun(@minus, post, llk));
end

function log_prob = lgmmprob(data, gmm)
    mu = gmm.mu;
    sigma = gmm.sigma;
    w = gmm.w;
    prec_chol = gmm.prec_chol;
    
    [ndim, n_samples] = size(data);
    n_components = length(w);
    log_prob = zeros(n_components, n_samples);
        
    for idx = 1 : n_components
        mu_i = mu(:, idx);
        sigma_i = sigma(:, :, idx);
        prec_chol_i = prec_chol(:, :, idx);

        y = (data - mu_i)' * prec_chol_i;
        
        log_prob(idx, :) = -0.5 * (ndim * log(2 * pi) + sum(y.*y, 2) + log(det(sigma_i)));
    end
            
    log_prob = bsxfun(@plus, log_prob, log(w(:)));
    

end

function y = logsumexp(x, dim)
    % compute log(sum(exp(x),dim)) while avoiding numerical underflow
    xmax = max(x, [], dim);
    y    = xmax + log(sum(exp(bsxfun(@minus, x, xmax)), dim));
    ind  = find(~isfinite(xmax));
    if ~isempty(ind)
        y(ind) = xmax(ind);
    end
end

function gmm = maximization(N, F, S)
    % ML re-estimation of GMM hyperparameters which are updated from accumulators
    w  = N / sum(N);
    mu = bsxfun(@rdivide, F, N);
%     sigma = bsxfun(@rdivide, S, N) - (mu .* mu);
    
    nmix = length(w);
    ndim = size(mu, 1);
    
%     sigma_eps = eye(ndim) * 1e-6;
    sigma = zeros(ndim, ndim, nmix);
    for idx = 1 : nmix
        mu_i = mu(:, idx);
        sigma(:, :, idx) = S(:,:,idx) / N(idx) - mu_i * mu_i'; % + sigma_eps;
    end
    
    sigma = apply_var_floors(w, sigma, 0.1);

    prec_chol = zeros(ndim, ndim, nmix);
    for idx = 1 : nmix
        prec_chol(:, :, idx) = chol(inv(sigma(:, :, idx)), 'lower');
    end
    
    gmm.w = w;
    gmm.mu= mu;
    gmm.sigma = sigma;
    gmm.prec_chol = prec_chol;
end

% function gmm = maximization2(N, F, S, datalist, gmm)
%     % ML re-estimation of GMM hyperparameters which are updated from accumulators
%     w  = N / sum(N);
%     mu = bsxfun(@rdivide, F, N);
% %     sigma = bsxfun(@rdivide, S, N) - (mu .* mu);
%     
%     nmix = length(w);
%     ndim = size(mu, 1);
%     
%     covariances = zeros(ndim, ndim, nmix);
%     
%     nfiles = length(datalist);
%     for ix = 1 : nfiles
%         data = datalist{ix};
%         [post, llk] = postprob(data, gmm);
% 
%         for im = 1 : nmix
%             diff = data - mu(:, im);
%             covariances(:, :, im) = covariances(:, :, im) + post(im, :) .* diff * diff';
%         end
%     end
%     
%     sigma_eps = eye(ndim) * 1e-6;
%     sigma = zeros(ndim, ndim, nmix);
%     prec_chol = zeros(ndim, ndim, nmix);
%     for idx = 1 : nmix
%         sigma(:, :, idx) = S(:,:,idx) / N(idx);
%         sigma(:, :, idx) = apply_var_floors(w(idx), sigma(:, :, idx), 0.1);
%         sigma(:, :, idx) = sigma(:, :, idx) + sigma_eps;
%         prec_chol(:, :, idx) = chol(inv(sigma(:, :, idx)));
%     end
%     
%     gmm.w = w;
%     gmm.mu= mu;
%     gmm.sigma = sigma;
%     gmm.prec_chol = prec_chol;
% end


function sigma = apply_var_floors(w, sigma, floor_const)
    % set a floor on covariances based on a weighted average of component
    % variances
    
    ndim = size(sigma, 1);
    nmix = size(sigma, 3);
    sigma_floor = zeros(ndim);
    for idx = 1 : nmix
        sigma_floor = sigma_floor + diag(sigma(:, :, idx)) * w(idx) * floor_const;
    end
    
    for idx = 1 : nmix
        for j = 1 : ndim
            if sigma(j, j, idx) < sigma_floor(j)
                sigma(j, j, idx) = sigma_floor(j);
            end
        end
    end
    
%     vFloor = sigma * w' * floor_const;
%     sigma  = bsxfun(@max, sigma, vFloor);
%     % sigma = bsxfun(@plus, sigma, 1e-6 * ones(size(sigma, 1), 1));
    
end


function gmm = gmm_mixup(gmm)
    % perform a binary split of the GMM hyperparameters
    mu = gmm.mu; 
    sigma = gmm.sigma; 
    w = gmm.w;
    nmix = length(w);
    ndim = size(mu, 1);
    
%     [sig_max, arg_max] = max(sigma);
%     eps = sparse(0 * mu);
%     eps(sub2ind([ndim, nmix], arg_max, 1 : nmix)) = sqrt(sig_max);
    % only perturb means associated with the max std along each dim 
    
    eps = zeros(size(mu));
    for idx = 1 : nmix
        aa = diag(squeeze(sigma(:, :, idx)));
        [sig_max, arg_max] = max(aa);
        eps(arg_max, idx) = sqrt(sig_max);
    end
    mu = [mu - eps, mu + eps];
    % mu = [mu - 0.2 * eps, mu + 0.2 * eps]; % HTK style
%     sigma = [sigma; sigma];
    sigma = cat(3, sigma, sigma);
    prec_chol = cat(3, gmm.prec_chol, gmm.prec_chol);
    w = [w, w] * 0.5;
    gmm.w  = w;
    gmm.mu = mu;
    gmm.sigma = sigma;
    gmm.prec_chol = prec_chol;
end

