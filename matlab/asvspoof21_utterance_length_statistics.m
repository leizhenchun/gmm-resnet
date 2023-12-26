function asvspoof21_utterance_length_statistics()


% asvspoof19_utterance_length_statistics('LA');
% asvspoof19_utterance_length_statistics('PA');

% asvspoof21_length_statistics('LA');
% asvspoof21_length_statistics('PA');
asvspoof21_length_statistics('DF');

end

function asvspoof19_utterance_length_statistics(access_type)

asvspoofRootPath = '/home/lzc/lzc/ASVspoof/';

dsPath19       = fullfile(asvspoofRootPath, 'DS_10283_3336/');
protocolPath19 = fullfile(dsPath19, access_type, ['ASVspoof2019_', access_type, '_cm_protocols']);
wavePath19     = fullfile(dsPath19, access_type);
% wavePath19     = '/home/lzc/lzc/ASVspoof/ASVspoof2019feat/WAVE_VAD/WAVE_VAD_LA_train/Original/';

trainProtocolFile19 = fullfile(protocolPath19, ['ASVspoof2019.', access_type, '.cm.train.trn.txt']);
devProtocolFile19   = fullfile(protocolPath19, ['ASVspoof2019.', access_type, '.cm.dev.trl.txt']);
evalProtocolFile19  = fullfile(protocolPath19, ['ASVspoof2019.', access_type, '.cm.eval.trl.txt']);

%% Train dataset
show_message(['utterance_length_statistics of ASVspoof2019 Train data : ']);
trainWavePath = fullfile(wavePath19, ['ASVspoof2019_', access_type, '_train'], 'flac');
trainWavePath = '/home/lzc/lzc/ASVspoof/ASVspoof2019feat/WAVE_VAD/WAVE_VAD_LA_train/Original/';
[train_speakerIds, train_fileIds, ~ ] = ASVspoof2019ReadProtocolFile( trainProtocolFile19 );
asvspoof_length_statisticse_list(trainWavePath, train_fileIds, access_type);


%%  development protocol
show_message(['utterance_length_statistics of ASVspoof2019 DEV data : ']);
devWavePath = fullfile(wavePath19, ['ASVspoof2019_', access_type, '_dev'], 'flac');
devWavePath = '/home/lzc/lzc/ASVspoof/ASVspoof2019feat/WAVE_VAD/WAVE_VAD_LA_dev/';
[dev_speakerIds, dev_fileIds, ~ ] = ASVspoof2019ReadProtocolFile( devProtocolFile19 );
asvspoof_length_statisticse_list(devWavePath, dev_fileIds, access_type);


%%  eval protocol
show_message(['utterance_length_statistics of ASVspoof2019 EVAL data : ']);
evalWavePath = fullfile(wavePath19, ['ASVspoof2019_', access_type, '_eval'], 'flac');
evalWavePath = '/home/lzc/lzc/ASVspoof/ASVspoof2019feat/WAVE_VAD/WAVE_VAD_LA_eval/';
[eval_speakerIds, eval_fileIds, ~ ] = ASVspoof2019ReadProtocolFile( evalProtocolFile19 );
asvspoof_length_statisticse_list(evalWavePath, eval_fileIds, access_type);

end


function asvspoof21_length_statistics(access_type)

asvspoofRootPath = '/home/lzc/lzc/ASVspoof/';

dataPath21     = fullfile(asvspoofRootPath, 'ASVspoof2021data', ['ASVspoof2021_', access_type, '_eval']);

evalProtocolFile21 = fullfile(dataPath21, ['ASVspoof2021.', access_type, '.cm.eval.trl.txt']);
evalWavePath21     = fullfile(dataPath21, 'flac');
evalWavePath21 = ['/home/lzc/lzc/ASVspoof/ASVspoof2021feat/WAVE_VAD/WAVE_VAD_', access_type, '_eval'];

%%  eval protocol
show_message(['utterance_length_statistics of ASVspoof2021 EVAL data : ']);
[eval_fileIds, ~ ] = ASVspoof2019ReadProtocolFile( evalProtocolFile21 );
seg_lenghth = asvspoof_length_statisticse_list(evalWavePath21, eval_fileIds, access_type);


truth_dir = fullfile(asvspoofRootPath, 'ASVspoof2021data', [access_type, '-keys-stage-1/keys']);
cm_key_file  = fullfile(truth_dir, 'CM/trial_metadata.txt');
if strcmp(access_type, 'LA') || strcmp(access_type, 'DF')
    [~, cm_seg, ~, ~, ~, cm_key, ~, cm_phase ]= textread(cm_key_file, '%s %s %s %s %s %s %s %s');
end
if strcmp(access_type, 'PA')
    [~, cm_seg, ~, ~, cm_key, ~, cm_phase ]= textread(cm_key_file, '%s %s %s %s %s %s %s');
end


phase = 'progress';
seg_lenghth_phase = seg_lenghth(strcmp(cm_phase, phase));
show_message(['utterance_length_statistics of ASVspoof2021 EVAL data : [', phase, ']']);
disp(['access_type:', access_type, '   count:', num2str(length(seg_lenghth_phase)), '   max:', num2str(max(seg_lenghth_phase)), '    min:', num2str(min(seg_lenghth_phase)), '    mean:', num2str(mean(seg_lenghth_phase)), '    std:', num2str(std(seg_lenghth_phase))]);


phase = 'eval';
seg_lenghth_phase = seg_lenghth(strcmp(cm_phase, phase));
show_message(['utterance_length_statistics of ASVspoof2021 EVAL data : [', phase, ']']);
disp(['access_type:', access_type, '   count:', num2str(length(seg_lenghth_phase)), '   max:', num2str(max(seg_lenghth_phase)), '    min:', num2str(min(seg_lenghth_phase)), '    mean:', num2str(mean(seg_lenghth_phase)), '    std:', num2str(std(seg_lenghth_phase))]);


if strcmp(access_type, 'LA') || strcmp(access_type, 'DF')
    phase = 'hidden_track';
    seg_lenghth_phase = seg_lenghth(strcmp(cm_phase, phase));
    show_message(['utterance_length_statistics of ASVspoof2021 EVAL data : [', phase, ']']);
    disp(['access_type:', access_type, '   count:', num2str(length(seg_lenghth_phase)), '   max:', num2str(max(seg_lenghth_phase)), '    min:', num2str(min(seg_lenghth_phase)), '    mean:', num2str(mean(seg_lenghth_phase)), '    std:', num2str(std(seg_lenghth_phase))]);
end


if strcmp(access_type, 'PA')
    phase = 'hidden_track_1';
    seg_lenghth_phase = seg_lenghth(strcmp(cm_phase, phase));
    show_message(['utterance_length_statistics of ASVspoof2021 EVAL data : [', phase, ']']);
    disp(['access_type:', access_type, '   count:', num2str(length(seg_lenghth_phase)), '   max:', num2str(max(seg_lenghth_phase)), '    min:', num2str(min(seg_lenghth_phase)), '    mean:', num2str(mean(seg_lenghth_phase)), '    std:', num2str(std(seg_lenghth_phase))]);

    
    phase = 'hidden_track_2';
    seg_lenghth_phase = seg_lenghth(strcmp(cm_phase, phase));
    show_message(['utterance_length_statistics of ASVspoof2021 EVAL data : [', phase, ']']);
    disp(['access_type:', access_type, '   count:', num2str(length(seg_lenghth_phase)), '   max:', num2str(max(seg_lenghth_phase)), '    min:', num2str(min(seg_lenghth_phase)), '    mean:', num2str(mean(seg_lenghth_phase)), '    std:', num2str(std(seg_lenghth_phase))]);
end


end


function seg_lenghth = asvspoof_length_statisticse_list(wavePath, fileIds, access_type)
    fileCount = length(fileIds);
    seg_lenghth = zeros(1, fileCount);
    parfor i=1:fileCount
        wave_file_name = fullfile(wavePath, [fileIds{i}, '.flac']);
        [x, fs] = audioread(wave_file_name);
       
        seg_lenghth(i) = length(x) / fs;
        
        if mod(i, 10000) == 0
            show_message( [num2str(100 * i / fileCount), '%   ']);
        end
    end
    
    disp(['access_type:', access_type, '   count:', num2str(length(seg_lenghth)), '   max:', num2str(max(seg_lenghth)), '    min:', num2str(min(seg_lenghth)), '    mean:', num2str(mean(seg_lenghth)), '    std:', num2str(std(seg_lenghth))]);
end

