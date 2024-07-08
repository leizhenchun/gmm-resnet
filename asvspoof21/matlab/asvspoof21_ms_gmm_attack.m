function asvspoof21_ms_gmm_attack()


asvspoof21_ms_gmm_attack_run('LFCC', 'LA', 128);
% asvspoof21_ms_gmm_tech_run('LFCC', 'PA', 128);


end

function asvspoof21_ms_gmm_attack_run(feature_type, access_type, nmix)
show_message(['evaluating asvspoof2021_ms_gmm_attack: ', feature_type, ' ', access_type]);

feat_acc = [feature_type, '_', access_type];

root_path = '/home/lzc/lzc/ASVspoof2021/';

exp_path21         = fullfile(root_path, 'ASVspoof2021exp/', ['ms_gmm_attack_', feature_type]);
feat_path21        = fullfile(root_path, 'ASVspoof2021feat/', feature_type, [feat_acc, '_eval']);
evalProtocolFile21 = fullfile(root_path, 'ASVspoof2021data/', ['ASVspoof2021_', access_type, '_eval'], ['ASVspoof2021.', access_type, '.cm.eval.trl.txt']);

make_path(exp_path21);
rng('shuffle');

[eval_fileIds21, ~] = ASVspoof2019ReadProtocolFile( evalProtocolFile21 );

gmmModelFile        = fullfile(exp_path21, ['ASVspoof2019_ms_gmm_attack_', feat_acc, '_gmm', num2str(nmix)]);
evalScoreFile21     = fullfile(exp_path21, ['AS21_ms_gmm_attack_', feat_acc, '_gmm', num2str(nmix), '_eval_score.txt']);

feat_DF_path21        = fullfile(root_path, 'ASVspoof2021feat/', feature_type, [feature_type, '_DF_eval']);
evalDFProtocolFile21  = fullfile(root_path, 'ASVspoof2021data/', ['ASVspoof2021_DF_eval'], ['ASVspoof2021.DF.cm.eval.trl.txt']);
evalDFScoreFile21     = fullfile(exp_path21, ['AS21_ms_gmm_attack_', feature_type, '_DF_gmm', num2str(nmix), '_eval_score.txt']);


diary(fullfile(exp_path21, ['ASVspoof2019_ms_gmm_attack_', feat_acc, '_gmm', num2str(nmix), '_log.txt']));
diary on;

show_message(['Model:ASVspoof2019_ms_gmm_attack: ', feature_type, ' ', access_type]);
show_message(['Loading GMM from: ', gmmModelFile]);
load(gmmModelFile, 'ubm', 'gmm_bonafide', 'gmm_spoof', 'gmm_attack', 'attack_type');


%% Scoring of asvspoof2021 eval data
show_message(['Computing scores for asvspoof2021 EVAL : ', feature_type, '  ', access_type]);

eval_scores21 = compute_scores(feat_path21, eval_fileIds21, gmm_bonafide, gmm_attack, attack_type);
asvspoof21_evaluate_tDCF_score(eval_scores21, access_type);

asvspoof21_save_scores(evalScoreFile21, eval_fileIds21, eval_scores21);
show_message(['Saving scores to: ', evalScoreFile21]);


%% Scoring of asvspoof2021 DF eval data

if access_type == 'PA'
    show_message();
    show_message('==========  All Done!  ===========');
    return;
end

show_message(['Computing scores for asvspoof2021 DF EVAL : ', feature_type, '  ', access_type]);

[eval_DF_fileIds21, ~] = ASVspoof2019ReadProtocolFile( evalDFProtocolFile21 );

eval_DF_scores21 = compute_scores(feat_DF_path21, eval_DF_fileIds21, gmm_bonafide, gmm_attack, attack_type);
asvspoof21_evaluate_tDCF_score(eval_DF_scores21, 'DF');

asvspoof21_save_scores(evalDFScoreFile21, eval_DF_fileIds21, eval_DF_scores21);
show_message(['Saving scores to: ', evalDFScoreFile21]);



show_message('==========  All Done!  ===========');
diary off;
end


function scores = compute_scores(feat_path, fileIds, gmm_bonafide, gmm_attack, attack_type)
    fileCount = length(fileIds);
    scores = zeros(fileCount, 1);
    parfor i = 1 : fileCount
        h5filename = fullfile(feat_path, [fileIds{i}, '.h5']);
        x_feature = h5read(h5filename, '/data')';

        llk_bonafide = mean(compute_llk(x_feature, gmm_bonafide.mu, gmm_bonafide.sigma, gmm_bonafide.w'));

        attack_scores = zeros(length(attack_type), 1);
        for attack = 1 : length(attack_type)
            attack_scores(attack) = mean(compute_llk(x_feature, gmm_attack{attack}.mu, gmm_attack{attack}.sigma, gmm_attack{attack}.w'));
        end
        scores(i) = llk_bonafide - max(attack_scores);

        if mod(i, 10000) == 0
            show_message( ['evaluating : ', num2str(100 * i / fileCount), '%   ']);
        end

    end

end


function asvspoof21_save_scores(score_file, filelist, scores)
    fid = fopen(score_file, 'w');
    for i=1:length(scores)
        fprintf(fid,'%s %.6f\n', filelist{i}, scores(i));
    end
    fclose(fid);
end
