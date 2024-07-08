

% asvspoof21_evaluate_tDCF_score('/home/lzc/lzc/ASVspoof2021/ASVspoof2021exp/GMM_ResNet_2P2S_LFCC/20220131_234133_GMM_ResNet_2P2S_LFCC_PA/AS21_GMM_ResNet_2P2S_LFCC_PA_KEY_ufm_score.txt', 'PA');

% asvspoof21_evaluate_tDCF_score('/home/lzc/lzc/ASVspoof2021/ASVspoof2021exp/GMM_ResNet_2P2S_LFCC/20220130_084009_GMM_ResNet_2P2S_LFCC_LA/AS21_GMM_ResNet_2P2S_LFCC_LA_KEY_ufm_score.txt', 'LA');

% asvspoof21_evaluate_tDCF_score('/home/lzc/lzc/ASVspoof2021/ASVspoof2021exp/GMM_ResNet_2P2S_LFCC/20220130_084009_GMM_ResNet_2P2S_LFCC_LA/AS21_GMM_ResNet_2P2S_LFCC_DF_KEY_ufm_score.txt', 'DF');


% asvspoof19_evaluate_tDCF_score('/home/lzc/lzc/ASVspoof2019/ASVspoof2019exp/ms_gmm_LFCC/ASVspoof2019_ms_gmm_LFCC_LA_gmm512_dev_score.txt', 'LA', 'dev');
% 
% asvspoof19_evaluate_tDCF_score('/home/lzc/lzc/ASVspoof2019/ASVspoof2019exp/ms_gmm_LFCC/ASVspoof2019_ms_gmm_LFCC_LA_gmm512_eval_score.txt', 'LA', 'eval');


% asvspoof19_evaluate_tDCF_attack('/home/lzc/lzc/ASVspoof2019/ASVspoof2019exp/ms_gmm_LFCC/ASVspoof2019_ms_gmm_LFCC_LA_gmm512_eval_score.txt', 'LA', 'eval');

% asvspoof19_evaluate_tDCF_attack('/home/lzc/lzc/ASVspoof2019/ASVspoof2019exp/GMM_ResNet_2Path_LFCC/20220114_203836_GMM_ResNet_2Path_LFCC_LA/AS19_GMM_ResNet_2Path_LFCC_LA_KEY_eval_ufm_score.txt', 'LA', 'eval');



% clear;/home/labuser/lzc/ASVspoof/ASVspoof2019feat/LFCC21NN/LFCC21NN_LA_train/Original
% 
aa = double(h5read('/home/labuser/lzc/ASVspoof/ASVspoof2019feat/LFCC21NN/LFCC21NN_LA_train/Original/LA_T_1000137.h5', '/data')');
bb = double(h5read('/home/labuser/ssd/lzc/ASVspoof/ASVspoof2019feat/LFCC21NN/LFCC21NN_LA_train/Original/LA_T_1000137.h5', '/data')');
% cc = double(h5read('/home/lzc/lzc/ASVspoof2021/ASVspoof2019feat/LFCCE21NN/LFCCE21NN_LA_train/Original/LA_T_1000137.h5', '/data')');
% dd = double(h5read('/home/lzc/lzc/ASVspoof2021/ASVspoof2019feat/LFCC/LFCC_LA_train/Original/LA_T_1000137.h5', '/data')');

sum(sum(aa-bb))

a=1
% filename = fullfile('/home/lzc/lzc/ASVspoof2021/DS_10283_3336/LA/ASVspoof2019_LA_train/flac/LA_T_2361751.flac');
% featureaa = asvspoof_extract_feature(filename, 'LA', 'LFCCBP')';
% 
% aa = 1;
