function asvspoof21_feat()


% asvspoof19_extract_feat('LA', 'LFCCBP');
% asvspoof21_extract_feat('LA', 'LFCCMatlab');
% asvspoof21_extract_feat('DF', 'LFCCMatlab');

asvspoof19_extract_feat('PA', 'LFCC19');
asvspoof21_extract_feat('PA', 'LFCC19');

asvspoof19_extract_feat('PA', 'LFCC30');
asvspoof21_extract_feat('PA', 'LFCC30');

% asvspoof_ms_gmm();
% 
% asvspoof2021_ms_gmm();

asvspoof21_ms_gmm_augment();

end

function asvspoof21_extract_feat(access_type, feature_type)

feat_acc = [feature_type, '_', access_type];

asvspoofRootPath = '/home/lzc/lzc/ASVspoof2021/';

dataPath21     = fullfile(asvspoofRootPath, 'ASVspoof2021data', ['ASVspoof2021_', access_type, '_eval']);
featPath21     = fullfile(asvspoofRootPath, 'ASVspoof2021feat/', feature_type);


evalProtocolFile21 = fullfile(dataPath21, ['ASVspoof2021.', access_type, '.cm.eval.trl.txt']);
evalWavePath21     = fullfile(dataPath21, 'flac');
evalFeatPath21     = fullfile(featPath21, [feat_acc, '_eval'], 'Original');
make_path(evalFeatPath21);

%%  eval protocol
show_message(['Extracting features for ASVspoof2021 EVAL data : ', feat_acc]);
[eval_fileIds, ~ ] = ASVspoof2019ReadProtocolFile( evalProtocolFile21 );
asvspoof_extract_feature_list(evalWavePath21, evalFeatPath21, eval_fileIds, access_type, feature_type);

show_message('Done!');

end



function asvspoof19_extract_feat(access_type, feature_type)

feat_acc = [feature_type, '_', access_type];

asvspoofRootPath = '/home/lzc/lzc/ASVspoof2021/';

dsPath19       = fullfile(asvspoofRootPath, 'DS_10283_3336/');
featPath19     = fullfile(asvspoofRootPath, 'ASVspoof2019feat', feature_type);
protocolPath19 = fullfile(dsPath19, access_type, ['ASVspoof2019_', access_type, '_cm_protocols']);
wavePath19     = fullfile(dsPath19, access_type);

% clear_path(featPath19);

trainProtocolFile19 = fullfile(protocolPath19, ['ASVspoof2019.', access_type, '.cm.train.trn.txt']);
devProtocolFile19   = fullfile(protocolPath19, ['ASVspoof2019.', access_type, '.cm.dev.trl.txt']);
evalProtocolFile19  = fullfile(protocolPath19, ['ASVspoof2019.', access_type, '.cm.eval.trl.txt']);

trainFeatPath19 = fullfile(featPath19, [feat_acc, '_train'], 'Original');
devFeatPath19   = fullfile(featPath19, [feat_acc, '_dev'], 'Original');
evalFeatPath19  = fullfile(featPath19, [feat_acc, '_eval'], 'Original');

%% Feature extraction for training data
show_message(['Extracting features for ASVspoof2019 TRAIN data : ', feat_acc]);
% make_path(trainFeatPath19);
trainWavePath = fullfile(wavePath19, ['ASVspoof2019_', access_type, '_train'], 'flac');
[train_speakerIds, train_fileIds, ~ ] = ASVspoof2019ReadProtocolFile( trainProtocolFile19 );
asvspoof_extract_feature_list(trainWavePath, trainFeatPath19, train_fileIds, access_type, feature_type);


%%  development protocol
show_message(['Extracting features for ASVspoof2019 DEV data : ', feat_acc]);
make_path(devFeatPath19);
devWavePath = fullfile(wavePath19, ['ASVspoof2019_', access_type, '_dev'], 'flac');
[dev_speakerIds, dev_fileIds, ~ ] = ASVspoof2019ReadProtocolFile( devProtocolFile19 );
asvspoof_extract_feature_list(devWavePath, devFeatPath19, dev_fileIds, access_type, feature_type);


%%  eval protocol
show_message(['Extracting features for ASVspoof2019 EVAL data : ', feat_acc]);
make_path(evalFeatPath19);
evalWavePath = fullfile(wavePath19, ['ASVspoof2019_', access_type, '_eval'], 'flac');
[eval_speakerIds, eval_fileIds, ~ ] = ASVspoof2019ReadProtocolFile( evalProtocolFile19 );
asvspoof_extract_feature_list(evalWavePath, evalFeatPath19, eval_fileIds, access_type, feature_type);


show_message('Done!');

end




function asvspoof_extract_feature_list(wavePath, featPath, fileIds, access_type, feature_type)
    if ~exist(featPath, 'dir')
        mkdir(featPath);
    end
    
    fileCount = length(fileIds);
    parfor i=1:fileCount
        filename = fullfile(wavePath, [fileIds{i}, '.flac']);
        feature = asvspoof_extract_feature(filename, access_type, feature_type);

        h5create(fullfile(featPath, [fileIds{i}, '.h5']), '/data', size(feature), 'Datatype', 'single');
        h5write(fullfile(featPath, [fileIds{i}, '.h5']), '/data', single(feature));
        
        if mod(i, 10000) == 0
            show_message( ['extract ', feature_type, ': ', num2str(100 * i / fileCount), '%   ']);
        end
    end
end


% 
% function feature = asvspoof21_extract_feature(wave_file_name, access_type, feature_type, useMvn)
% 
%     if ~exist('useMvn','var') 
%         useMvn = false;
%     end
% 
%     [x, fs] = audioread(wave_file_name);
%         
%     if strcmp(feature_type,'CQCC')
%         % ASVspoof 2021 default settings
%         B = 12;             % number of bins per octave
%         fmin = 62.50;       % lowest frequency to be analyzed
%         fmax = 4000;        % highest frequency to be analyzed
%         d = 16;             % number of uniform samples in the first octave
%         cf = 19;            % number of cepstral coefficients excluding 0'th coefficient
%         feature = cqcc(x, fs, B, fmax, fmin, d, cf, 'ZsdD')';
%         
%         % ASVspoof 2019 default settings
% %         feature = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD')';
% 
%     elseif strcmp(feature_type,'LFCC')
%         % ASVspoof 2021 default settings
%         window_length = 20; % 30 for GMM   and 20 for LCNN   % ms
%         if strcmp(access_type, 'LA') || strcmp(access_type, 'DF')
%             high_freq = 4000;   % 4000 for LA DF    and  80000 for PA  % highest frequency to be analyzed
%         elseif strcmp(access_type, 'PA')
%             high_freq = 8000;
%         else
%             show_message(['access_type = ', access_type, ',  ERROR!!!!!!']);
%             return
%         end
%         NFFT = 1024;        % FFT bins
%         no_Filter = 70;     % no of filters
%         no_coeff = 20;      % no of coefficients including 0'th coefficient
%         low_freq = 0;       % lowest frequency to be analyzed
%         [stat,delta,double_delta] = lfcc_bp(x,fs,window_length,NFFT,no_Filter,no_coeff,low_freq,high_freq);
%         feature = [stat delta double_delta];
%         
%         % ASVspoof 2019 default settings
% %         [stat,delta,double_delta] = extract_lfcc(x,fs,20,512,20);
% %         feature = [stat delta double_delta];
% 
%     elseif strcmp(feature_type,'LFCC30')
%         % ASVspoof 2021 default settings
%         window_length = 30; % 30 for GMM   and 20 for LCNN   % ms
%         if strcmp(access_type, 'LA') || strcmp(access_type, 'DF')
%             high_freq = 4000;   % 4000 for LA DF    and  80000 for PA  % highest frequency to be analyzed
%         elseif strcmp(access_type, 'PA')
%             high_freq = 8000;
%         else
%             show_message(['access_type = ', access_type, ',  ERROR!!!!!!']);
%             return
%         end
%         NFFT = 1024;        % FFT bins
%         no_Filter = 70;     % no of filters
%         no_coeff = 19;      % no of coefficients including 0'th coefficient
%         low_freq = 0;       % lowest frequency to be analyzed
%         [stat,delta,double_delta] = lfcc_bp(x,fs,window_length,NFFT,no_Filter,no_coeff,low_freq,high_freq);
%         feature = [stat delta double_delta];
%         
%     elseif strcmp(feature_type,'MFCC')
%         feature = melcepst(x, fs, '0EdD', 12, 24, 0.02*fs, 0.01*fs, 0.02, 0.5)';
%     elseif strcmp(feature_type,'FBANK')
%         feature = myfbank(x, fs, '0dD', 19, 30, 0.03*fs, 0.015*fs, 0.02, 0.25)';
% 
%     elseif strcmp(feature_type,'LOGCQT')
%         feature = log_cqt(x, fs, 96, fs/2, fs/2^10, 400);
%     elseif strcmp(feature_type,'SPEC')
%         ninc=0.01*fs;           % Frame increment for BW=200 Hz (in samples)
%         nwin=2*ninc;              % Frame length (in samples)
%         win=hamming(nwin);        % Analysis window
%         k=0.5*fs*sum(win.^2);     % Scale factor to convert to power/Hz
%         n = 864 * 2;
%         sf=abs(v_rfft(v_enframe(x,win,ninc), n, 2)).^2/k;           % Calculate spectrum array  
% %         [t,f,b]=v_spgrambw(x, [fs/ninc 0.5*(nwin+1)/fs fs/nwin], 'p');  % Plot spectrum array
%         feature = single(sf);
%         
%     elseif strcmp(feature_type, 'LPOW')
%         feature = logpow(x, fs);
%     elseif strcmp(feature_type, 'LP227')
%         feature = logpow_ext(x, fs);
%     elseif strcmp(feature_type, 'LP768')
%         feature = logpow_ext2(x, fs, 768 * 2, 400);
%     elseif strcmp(feature_type, 'LP864')
%         feature = log_power(x, fs, 864 * 2, 400);
%     elseif strcmp(feature_type, 'logspec256')
%         feature = log_power(x, fs, 512, 400);
%     elseif strcmp(feature_type, 'SPROMFCC')
%         cmdstr = ['sfbcep -F wave -f 16000 -m -D -A -e -i 300 -u 8000 -x 1 ', wavefile, ' ', featfile];
%         show_message(cmdstr);
%         system(cmdstr);
%         feature = [];
%     end
% 
%     
%     if(useMvn)
%         feature = cmvn(feature, true);
%     end
%     
% end


    
