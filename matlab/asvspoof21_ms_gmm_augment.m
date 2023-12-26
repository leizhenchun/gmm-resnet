function asvspoof21_ms_gmm_augment()

% aug_methods = ["Original", "ALAW", "ULAW"];
% aug_methods = ["Original", "RIR", "NOISE", "MUSIC", "SPEECH"];

% feature_type = 'LFCC21NN';


% aug_methods = ["Original"];
% gmm_type = 'gmm';
% asvspoof_gmm_run_21(feature_type, 'PA', 512, aug_methods, gmm_type);
% asvspoof_gmm_run_21(feature_type, 'PA', 256, aug_methods, gmm_type);
% asvspoof_gmm_run_21(feature_type, 'PA', 128, aug_methods, gmm_type);

feature_type = 'LFCC21NN';


aug_methods = ["Original"];
gmm_type = 'GMM';

% asvspoof_gmm_run_21(feature_type, 'LA', 1024, aug_methods, gmm_type);
% asvspoof_gmm_run_21(feature_type, 'PA', 1024, aug_methods, gmm_type);

for i = 4 : 4
%     aug_methods = "RB" + i;
%     gmm_type = ['gmm_rb', num2str(i)];

    aug_methods = ["Original", "RB" + i];
    gmm_type = ['GMM302_aug2_rb', num2str(i)];
    
    asvspoof_gmm_run_21(feature_type, 'LA', 1024, aug_methods, gmm_type);
%     asvspoof_gmm_run_21(feature_type, 'PA', 1024, aug_methods, gmm_type);
end


% aug_methods = ["Original", "ALAW", "ULAW", "RIR", "NOISE", "MUSIC", "SPEECH"];
% for i = 1:length(aug_methods)
%     aug_method = aug_methods(i);
%     gmm_type = "gmm_" + lower(aug_method);
%     asvspoof_gmm_run_21(feature_type, 'LA', 512, aug_method, gmm_type);
% end

% aug_methods = ["Original", "RIR", "NOISE", "MUSIC", "SPEECH"];
% gmm_type = 'gmm_aug5_rirnoise';
% asvspoof_gmm_run_21(feature_type, 'LA', 256, aug_methods, gmm_type);


% aug_methods = ["Original", "ALAW", "ULAW"];
% gmm_type = 'gmm_aug3_law';
% asvspoof_gmm_run_21(feature_type, 'LA', 512, aug_methods, gmm_type);

% aug_methods = ["Original", "ALAW", "ULAW", "RIR", "NOISE", "MUSIC", "SPEECH", "RB1", "RB2", "RB3", "RB4", "RB5", "RB6", "RB7", "RB8"];
% gmm_type = 'gmm_aug_all';
% asvspoof_gmm_run_21(feature_type, 'LA', 512, aug_methods, gmm_type);

% aug_methods = ["Original", "RB1", "RB2", "RB3", "RB4"];
% gmm_type = 'gmm_aug5_rb1234';
% asvspoof_gmm_run_21(feature_type, 'LA', 512, aug_methods, gmm_type);


end

function asvspoof_gmm_run_19(feature_type, access_type, nmix, aug_methods, gmm_type)

root_path = '/home/labuser/ssd/lzc/ASVspoof';
exp_path  = fullfile(root_path, 'ASVspoof2019exp', [gmm_type, '_', feature_type]);
make_path(exp_path);

asvspoof_gmm_run(feature_type, access_type, nmix, root_path, exp_path, aug_methods);

end

function asvspoof_gmm_run_21(feature_type, access_type, nmix, aug_methods, gmm_type)

root_path = '/home/labuser/ssd/lzc/ASVspoof';
exp_path  = fullfile(root_path, 'ASVspoof2021exp', [gmm_type, '_', feature_type]);
make_path(exp_path);

asvspoof_gmm_run(feature_type, access_type, nmix, root_path, exp_path, aug_methods);

asvspoof_gmm_llk_mean_std_run(feature_type, access_type, nmix, root_path, exp_path, aug_methods);

end

function asvspoof_gmm_run(feature_type, access_type, nmix, root_path, exp_path, aug_methods)
show_message(['Model:ASVspoof_gmm: ', feature_type, ' ', access_type]);

gmm_niter = 30;
feat_acc = [feature_type, '_', access_type];

feat_path19           = fullfile(root_path, 'ASVspoof2019feat', feature_type);

protocol_path       = fullfile(root_path, 'DS_10283_3336/', access_type, ['ASVspoof2019_', access_type, '_cm_protocols']);
trainProtocolFile   = fullfile(protocol_path, ['ASVspoof2019.', access_type, '.cm.train.trn.txt']);
devProtocolFile     = fullfile(protocol_path, ['ASVspoof2019.', access_type, '.cm.dev.trl.txt']);
evalProtocolFile    = fullfile(protocol_path, ['ASVspoof2019.', access_type, '.cm.eval.trl.txt']);

[train_speakerIds, train_fileIds, train_environmentIds, train_attackIds, train_key] = ASVspoof2019ReadProtocolFile( trainProtocolFile );
[dev_speakerIds, dev_fileIds, dev_environmentIds, dev_attackIds, dev_key] = ASVspoof2019ReadProtocolFile( devProtocolFile );
[eval_speakerIds, eval_fileIds, eval_environmentIds, eval_attackIds, eval_key] = ASVspoof2019ReadProtocolFile( evalProtocolFile );

gmmModelFile        = fullfile(exp_path, ['ASVspoof2019_GMM_', feat_acc, '_', num2str(nmix)]);
devScoreFile        = fullfile(exp_path, ['ASVspoof2019_GMM_', feat_acc, '_', num2str(nmix), '_dev_score.txt']);
evalScoreFile       = fullfile(exp_path, ['ASVspoof2019_GMM_', feat_acc, '_', num2str(nmix), '_eval_score.txt']);

feat_path21         = fullfile(root_path, 'ASVspoof2021feat/', feature_type, [feat_acc, '_eval']);
evalProtocolFile21  = fullfile(root_path, 'ASVspoof2021data/', ['ASVspoof2021_', access_type, '_eval'], ['ASVspoof2021.', access_type, '.cm.eval.trl.txt']);
[eval_fileIds21, ~] = ASVspoof2019ReadProtocolFile( evalProtocolFile21 );
evalScoreFile21     = fullfile(exp_path, ['ASVspoof2021_GMM_', feat_acc, '_', num2str(nmix), '_eval_score.txt']);

feat_DF_path21       = fullfile(root_path, 'ASVspoof2021feat/', feature_type, [feature_type, '_DF_eval']);
evalDFProtocolFile21 = fullfile(root_path, 'ASVspoof2021data/', ['ASVspoof2021_DF_eval'], ['ASVspoof2021.DF.cm.eval.trl.txt']);
evalDFScoreFile21    = fullfile(exp_path, ['ASVspoof2021_GMM_', feature_type, '_DF_', num2str(nmix), '_eval_score.txt']);


diary(fullfile(exp_path, ['ASVspoof2019_GMM_', feat_acc, '_', num2str(nmix), '_log.txt']));
diary on;


%% GMM training
show_message('Load training data ...');
train_feat_path = fullfile(feat_path19, [feat_acc, '_train']);
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

show_message('UBM training...');
ubm_all = gmm_em_quiet2(train_feature, nmix, gmm_niter);

% show_message('GMM training bonafide...');
% gmm_bonafide_all = gmm_em_quiet2(train_bonafide_feature, nmix, gmm_niter);
% 
% show_message('GMM training spoof...');
% gmm_spoof_all   = gmm_em_quiet2(train_spoof_feature,   nmix, gmm_niter);


% save(gmmModelFile, 'ubm_all', 'gmm_bonafide_all', 'gmm_spoof_all');
save(gmmModelFile, 'ubm_all');

clear train_feature train_bonafide_feature  train_spoof_feature;
show_message('Done!');
return;

ubm = ubm_all{end};
gmm_bonafide = gmm_bonafide_all{end};
gmm_spoof = gmm_spoof_all{end};


show_message(['Loading GMM from: ', gmmModelFile]);
load(gmmModelFile, 'ubm', 'gmm_bonafide', 'gmm_spoof');

%% scoring of ASVspoof 2019 development data
show_message(['Computing scores for ASVspoof 2019 DEV trials : ', feature_type, '  ', access_type]);

dev_scores = compute_scores(fullfile(feat_path19, [feat_acc, '_dev']), dev_fileIds, gmm_bonafide, gmm_spoof);  % , 'Original'
asvspoof19_evaluate_tDCF_score(dev_scores, access_type, 'dev');

asvspoof19_save_scores(devScoreFile, dev_fileIds, dev_scores);
show_message(['Saving scores to: ', devScoreFile]);


%% Scoring of ASVspoof 2019 eval data
show_message(['Computing scores for ASVspoof 2019 EVAL trials : ', feature_type, '  ', access_type]);

eval_scores = compute_scores(fullfile(feat_path19, [feat_acc, '_eval']), eval_fileIds, gmm_bonafide, gmm_spoof);
asvspoof19_evaluate_tDCF_score(eval_scores, access_type, 'eval');

asvspoof19_save_scores(evalScoreFile, eval_fileIds, eval_scores);
show_message(['Saving scores to: ', evalScoreFile]);


%% Scoring of asvspoof2021 eval data
show_message(['Computing scores for asvspoof2021 EVAL : ', feature_type, '  ', access_type]);

eval_scores21 = compute_scores(fullfile(feat_path21), eval_fileIds21, gmm_bonafide, gmm_spoof);
asvspoof21_evaluate_tDCF_score(eval_scores21, access_type);

asvspoof_save_scores(evalScoreFile21, eval_fileIds21, eval_scores21);
show_message(['Saving scores to: ', evalScoreFile21]);

% return;

%% Scoring of asvspoof2021 DF eval data

if access_type == 'PA'
    show_message();
    show_message('==========  All Done!  ===========');
    return;
end

show_message(['Computing scores for asvspoof2021 DF EVAL : ', feature_type, '  ', access_type]);
[eval_DF_fileIds21, ~] = ASVspoof2019ReadProtocolFile( evalDFProtocolFile21 );

eval_DF_scores21 = compute_scores(fullfile(feat_DF_path21), eval_DF_fileIds21, gmm_bonafide, gmm_spoof);
asvspoof21_evaluate_tDCF_score(eval_DF_scores21, 'DF');

asvspoof_save_scores(evalDFScoreFile21, eval_DF_fileIds21, eval_DF_scores21);
show_message(['Saving scores to: ', evalDFScoreFile21]);

show_message('==========  All Done!  ===========');
diary off;

end


function scores = compute_scores(feat_path, fileIds, gmm_bonafide, gmm_spoof)
    fileCount = length(fileIds);
    scores = zeros(fileCount, 1);
    parfor i = 1 : fileCount
        h5filename = fullfile(feat_path, [fileIds{i}, '.h5']);
        x_feature = h5read(h5filename, '/data')';

        llk_bonafide = mean(gmm_compute_llk(x_feature, gmm_bonafide.mu, gmm_bonafide.sigma, gmm_bonafide.w'));
        llk_spoof = mean(gmm_compute_llk(x_feature, gmm_spoof.mu, gmm_spoof.sigma, gmm_spoof.w'));
        scores(i) = llk_bonafide - llk_spoof;
        
        if mod(i, 10000) == 0
            show_message( ['evaluating : ', num2str(100 * i / fileCount), '%   ']);
        end
    end
    
end


function asvspoof_save_scores(score_file, filelist, scores)
    fid = fopen(score_file, 'w');
    for i=1:length(scores)
        fprintf(fid,'%s %.6f\n', filelist{i}, scores(i));
    end
    fclose(fid);
end


function asvspoof_gmm_llk_mean_std_run(feature_type, access_type, nmix, root_path, exp_path, aug_methods)
show_message(['asvspoof_gmm_llk_mean_std ', feature_type, ' ', access_type]);

feat_acc  = [feature_type, '_', access_type];

feat_path     = fullfile(root_path, 'ASVspoof2019feat/', feature_type);
protocol_path = fullfile(root_path, 'DS_10283_3336/', access_type, ['ASVspoof2019_', access_type, '_cm_protocols']);

gmmModelFile      = fullfile(exp_path, ['ASVspoof2019_GMM_', feat_acc, '_', num2str(nmix)]);
gmmllkMeanStdFile = fullfile(exp_path, ['ASVspoof2019_GMM_', feat_acc, '_', num2str(nmix), '_llk_mean_std.h5']);

train_feat_path   = fullfile(feat_path, [feat_acc, '_train']);
trainProtocolFile = fullfile(protocol_path, ['ASVspoof2019.', access_type, '.cm.train.trn.txt']);

[train_speakerIds, train_fileIds, train_environmentIds, train_attackIds, train_key] = ASVspoof2019ReadProtocolFile( trainProtocolFile );


%% load gmm
show_message(['loading GMM : ', gmmModelFile]);
% load(gmmModelFile, 'ubm_all', 'gmm_bonafide_all', 'gmm_spoof_all');
load(gmmModelFile, 'ubm_all');

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



for gmm_idx = 1 : length(ubm_all)
    
    ubm = ubm_all{gmm_idx};
%     gmm_bonafide = gmm_bonafide_all{gmm_idx};
%     gmm_spoof = gmm_spoof_all{gmm_idx};
    
    nmix = length(ubm.w);
    
    
    show_message(['compute GMM llk : nmix ', num2str(nmix)]);
    
    [ubm_mean, ubm_std] = compute_mean_std(ubm, train_feature);
    gmm_filename = fullfile(exp_path, ['ASVspoof2019_GMM_', feat_acc, '_', num2str(nmix), '.h5']);
    save_gmm(gmm_filename, ubm.w, ubm.mu, ubm.sigma, ubm_mean, ubm_std);

%     [bonafide_mean, bonaffide_std] = compute_mean_std(gmm_bonafide, train_feature);
%     gmm_filename = fullfile(exp_path, ['ASVspoof2019_GMM_', feat_acc, '_', num2str(nmix), '_bonafide.h5']);
%     save_gmm(gmm_filename, gmm_bonafide.w, gmm_bonafide.mu, gmm_bonafide.sigma, bonafide_mean, bonaffide_std);
% 
%     [spoof_mean, spoof_std] = compute_mean_std(gmm_spoof, train_feature);
%     gmm_filename = fullfile(exp_path, ['ASVspoof2019_GMM_', feat_acc, '_', num2str(nmix), '_spoof.h5']);
%     save_gmm(gmm_filename, gmm_spoof.w, gmm_spoof.mu, gmm_spoof.sigma, spoof_mean, spoof_std);

end


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


function save_gmm(gmm_filename, w, mu, sigma, feat_mean, feat_std)

h5create(gmm_filename, '/w', size(w));
h5write( gmm_filename, '/w', w);

h5create(gmm_filename, '/mu', size(mu));
h5write( gmm_filename, '/mu', mu);

h5create(gmm_filename, '/sigma', size(sigma));
h5write( gmm_filename, '/sigma', sigma);

h5create(gmm_filename, '/feat_mean', size(feat_mean));
h5write( gmm_filename, '/feat_mean', feat_mean);

h5create(gmm_filename, '/feat_std', size(feat_std));
h5write( gmm_filename, '/feat_std', feat_std);


end


