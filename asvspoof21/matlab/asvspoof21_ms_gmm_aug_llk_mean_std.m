function asvspoof21_ms_gmm_aug_llk_mean_std()

% aug_methods = ["Original", "ALAW", "ULAW"];
% aug_methods = ["Original", "RIR", "NOISE", "MUSIC", "SPEECH"];


feature_type = 'LFCC';

% aug_methods = ["Original"];
% gmm_type = 'gmm';
% asvspoof_gmm_llk_mean_std_run_21(feature_type, 'LA', 512, aug_methods, gmm_type);

for i = 4 : 4
%     aug_methods = "RB" + i;
%     gmm_type = ['gmm_rb', num2str(i)];

    aug_methods = ["Original", "RB" + i];
    gmm_type = ['gmm_aug2_rb', num2str(i)];

%     asvspoof_gmm_llk_mean_std_run_21(feature_type, 'LA', 128, aug_methods, gmm_type);
%     asvspoof_gmm_llk_mean_std_run_21(feature_type, 'LA', 256, aug_methods, gmm_type);
    asvspoof_gmm_llk_mean_std_run_21(feature_type, 'LA', 512, aug_methods, gmm_type);
end


% aug_methods = ["Original", "ALAW", "ULAW", "RIR", "NOISE", "MUSIC", "SPEECH"];
% for i = 1:length(aug_methods)
%     aug_method = aug_methods(i);
%     gmm_type = "gmm_" + lower(aug_method);
%     asvspoof_gmm_llk_mean_std_run_21(feature_type, 'LA', 128, aug_method, gmm_type);
% end


% aug_methods = ["Original", "RIR", "NOISE", "MUSIC", "SPEECH"];
% gmm_type = 'gmm_aug5_rirnoise';
% asvspoof_gmm_llk_mean_std_run_21(feature_type, 'LA', 256, aug_methods, gmm_type);

% 
% aug_methods = ["Original", "ALAW", "ULAW"];
% gmm_type = 'gmm_aug3_law';
% asvspoof_gmm_llk_mean_std_run_21(feature_type, 'LA', 512, aug_methods, gmm_type);


% aug_methods = ["Original", "ALAW", "ULAW", "RIR", "NOISE", "MUSIC", "SPEECH", "RB1", "RB2", "RB3", "RB4", "RB5", "RB6", "RB7", "RB8"];
% gmm_type = 'gmm_aug_all';
% asvspoof_gmm_llk_mean_std_run_21(feature_type, 'LA', 512, aug_methods, gmm_type);

% aug_methods = ["Original", "RB1", "RB2", "RB3", "RB4"];
% gmm_type = 'gmm_aug5_rb1234';
% asvspoof_gmm_llk_mean_std_run_21(feature_type, 'LA', 512, aug_methods, gmm_type);


end

function asvspoof_gmm_llk_mean_std_run_19(feature_type, access_type, nmix, aug_methods, gmm_type)

root_path = '/home/lzc/lzc/ASVspoof';
exp_path  = fullfile(root_path, 'ASVspoof2019exp/', [gmm_type, '_', feature_type]);

asvspoof_gmm_llk_mean_std_run(feature_type, access_type, nmix, root_path, exp_path, aug_methods);

end

function asvspoof_gmm_llk_mean_std_run_21(feature_type, access_type, nmix, aug_methods, gmm_type)

root_path = '/home/lzc/lzc/ASVspoof';
exp_path  = fullfile(root_path, 'ASVspoof2021exp/', [gmm_type, '_', feature_type]);

asvspoof_gmm_llk_mean_std_run(feature_type, access_type, nmix, root_path, exp_path, aug_methods);

end

function asvspoof_gmm_llk_mean_std_run(feature_type, access_type, nmix, root_path, exp_path, aug_methods)
show_message(['asvspoof_gmm_llk_mean_std ', feature_type, ' ', access_type]);

feat_acc  = [feature_type, '_', access_type];

feat_path     = fullfile(root_path, 'ASVspoof2019feat/', feature_type);
protocol_path = fullfile(root_path, 'DS_10283_3336/', access_type, ['ASVspoof2019_', access_type, '_cm_protocols']);

gmmModelFile      = fullfile(exp_path, ['ASVspoof2019_gmm_', feat_acc, '_', num2str(nmix)]);
gmmllkMeanStdFile = fullfile(exp_path, ['ASVspoof2019_gmm_', feat_acc, '_', num2str(nmix), '_llk_mean_std.h5']);

train_feat_path   = fullfile(feat_path, [feat_acc, '_train']);
trainProtocolFile = fullfile(protocol_path, ['ASVspoof2019.', access_type, '.cm.train.trn.txt']);

[train_speakerIds, train_fileIds, train_environmentIds, train_attackIds, train_key] = ASVspoof2019ReadProtocolFile( trainProtocolFile );


%% load gmm
show_message(['loading gmm : ', gmmModelFile]);
load(gmmModelFile, 'ubm', 'gmm_bonafide', 'gmm_spoof');

show_message('Load training data ...');

bonafide_idx    = strcmp(train_key, 'bonafide');
spoof_idx       = strcmp(train_key, 'spoof');

train_feature = {};
train_bonafide_feature = {};
train_spoof_feature = {};
for i = 1:length(aug_methods)
    train_feature_tmp   = asvspoof19_load_feature(fullfile(train_feat_path, aug_methods(i)), train_fileIds);
    train_bonafide_feature_tmp = train_feature_tmp(bonafide_idx);
    train_spoof_feature_tmp    = train_feature_tmp(spoof_idx);
    
    train_feature = [train_feature; train_feature_tmp];
    train_bonafide_feature = [train_bonafide_feature; train_bonafide_feature_tmp];
    train_spoof_feature = [train_spoof_feature; train_spoof_feature_tmp];
end
show_message(['Training dataset size : ', num2str(length(train_feature))]);

[gm, gv] = comp_gm_gv(train_feature);

show_message( 'compute gmm llk : ubm');
[ubm_mean, ubm_std] = compute_mean_std(ubm, train_feature);

show_message( 'compute gmm llk : bonafide');
[bonafide_mean, bonaffide_std] = compute_mean_std(gmm_bonafide, train_feature);

show_message( 'compute gmm llk : spoof');
[spoof_mean, spoof_std] = compute_mean_std(gmm_spoof, train_feature);


%% Saving
show_message(['saving mean, std to : ', gmmllkMeanStdFile]);

h5create(gmmllkMeanStdFile, '/gm', size(gm));
h5write( gmmllkMeanStdFile, '/gm', gm);

h5create(gmmllkMeanStdFile, '/gv', size(gv));
h5write( gmmllkMeanStdFile, '/gv', gv);

h5create(gmmllkMeanStdFile, '/gstd', size(gv));
h5write( gmmllkMeanStdFile, '/gstd', sqrt(gv));


h5create(gmmllkMeanStdFile, '/gmm_ubm_w', size(ubm.w));
h5write( gmmllkMeanStdFile, '/gmm_ubm_w', ubm.w);

h5create(gmmllkMeanStdFile, '/gmm_ubm_mu', size(ubm.mu));
h5write( gmmllkMeanStdFile, '/gmm_ubm_mu', ubm.mu);

h5create(gmmllkMeanStdFile, '/gmm_ubm_sigma', size(ubm.sigma));
h5write( gmmllkMeanStdFile, '/gmm_ubm_sigma', ubm.sigma);

h5create(gmmllkMeanStdFile, '/feat_ubm_mean', size(ubm_mean));
h5write( gmmllkMeanStdFile, '/feat_ubm_mean', ubm_mean);

h5create(gmmllkMeanStdFile, '/feat_ubm_std', size(ubm_std));
h5write( gmmllkMeanStdFile, '/feat_ubm_std', ubm_std);


h5create(gmmllkMeanStdFile, '/gmm_bonafide_w', size(gmm_bonafide.w));
h5write( gmmllkMeanStdFile, '/gmm_bonafide_w', gmm_bonafide.w);

h5create(gmmllkMeanStdFile, '/gmm_bonafide_mu', size(gmm_bonafide.mu));
h5write( gmmllkMeanStdFile, '/gmm_bonafide_mu', gmm_bonafide.mu);

h5create(gmmllkMeanStdFile, '/gmm_bonafide_sigma', size(gmm_bonafide.sigma));
h5write( gmmllkMeanStdFile, '/gmm_bonafide_sigma', gmm_bonafide.sigma);

h5create(gmmllkMeanStdFile, '/feat_bonafide_mean', size(bonafide_mean));
h5write( gmmllkMeanStdFile, '/feat_bonafide_mean', bonafide_mean);

h5create(gmmllkMeanStdFile, '/feat_bonafide_std', size(bonaffide_std));
h5write( gmmllkMeanStdFile, '/feat_bonafide_std', bonaffide_std);


h5create(gmmllkMeanStdFile, '/gmm_spoof_w', size(gmm_spoof.w));
h5write( gmmllkMeanStdFile, '/gmm_spoof_w', gmm_spoof.w);

h5create(gmmllkMeanStdFile, '/gmm_spoof_mu', size(gmm_spoof.mu));
h5write( gmmllkMeanStdFile, '/gmm_spoof_mu', gmm_spoof.mu);

h5create(gmmllkMeanStdFile, '/gmm_spoof_sigma', size(gmm_spoof.sigma));
h5write( gmmllkMeanStdFile, '/gmm_spoof_sigma', gmm_spoof.sigma);

h5create(gmmllkMeanStdFile, '/feat_spoof_mean', size(spoof_mean));
h5write( gmmllkMeanStdFile, '/feat_spoof_mean', spoof_mean);

h5create(gmmllkMeanStdFile, '/feat_spoof_std', size(spoof_std));
h5write( gmmllkMeanStdFile, '/feat_spoof_std', spoof_std);

end

function logprob = lgmmprob2(data, mu, sigma, w)
    logprob = (1./sigma)' * (data .* data) - 2 * (mu./sigma)' * data;
end

function [feat_mean, feat_std] = compute_mean_std(gmm, train_feature)

    feat_x1 = zeros(1, size(gmm.mu, 2));
    feat_xx1 = zeros(1, size(gmm.mu, 2));
    feat_count = 0;
    
    fileCount = length(train_feature);
    for i=1:fileCount
        feature = train_feature{i};
        feat_count = feat_count + size(feature, 2);

        post1 = lgmmprob2(feature, gmm.mu, gmm.sigma, gmm.w');
        feat_x1 = feat_x1 + sum(post1, 2)';
        feat_xx1 = feat_xx1 + sum(post1 .* post1, 2)';
        
    end
    
    feat_mean = feat_x1 / feat_count;
    feat_std = sqrt(feat_xx1 / feat_count - feat_mean .* feat_mean);
    
end



function [gm, gv] = comp_gm_gv(data)
    % computes the global mean and variance of data
    nframes = cellfun(@(x) size(x, 2), data, 'UniformOutput', false);
    nframes = sum(cell2mat(nframes));
    gm = cellfun(@(x) sum(x, 2), data, 'UniformOutput', false);
    gm = sum(cell2mat(gm'), 2)/nframes;
    gv = cellfun(@(x) sum(bsxfun(@minus, x, gm).^2, 2), data, 'UniformOutput', false);
    gv = sum(cell2mat(gv'), 2)/( nframes - 1 );
end
