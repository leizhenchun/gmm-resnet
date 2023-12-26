function asvspoof19_aug_feat()

access_type = 'LA';
feature_type = 'LFCCMatlab';

asvspoof19_extract_feat('LA', 'LFCCMatlab', 'RB6');


% asvspoof_ms_gmm();
% 
% asvspoof2021_ms_gmm();

end


function asvspoof19_extract_feat(access_type, feature_type, aug_method)

feat_acc = [feature_type, '_', access_type];

asvspoofRootPath = '/home/lzc/lzc/ASVspoof2021/';

dsPath19       = fullfile(asvspoofRootPath, 'DS_10283_3336/');
featPath19     = fullfile(asvspoofRootPath, 'ASVspoof2019feat', feature_type);
protocolPath19 = fullfile(dsPath19, access_type, ['ASVspoof2019_', access_type, '_cm_protocols']);
% wavePath19     = fullfile(dsPath19, access_type);

trainProtocolFile19 = fullfile(protocolPath19, ['ASVspoof2019.', access_type, '.cm.train.trn.txt']);

trainFeatPath19 = fullfile(featPath19, [feat_acc, '_train'], aug_method);

%% Feature extraction for training data
show_message(['Extracting features for ASVspoof2019 TRAIN data : ', feat_acc]);
make_path(trainFeatPath19);

trainWavePath = fullfile('/home/lzc/lzc/ASVspoof2021/ASVspoof2019trainwave', access_type, aug_method);
% trainWavePath = fullfile(wavePath19, ['ASVspoof2019_', access_type, '_train'], 'flac');
[train_speakerIds, train_fileIds, ~ ] = ASVspoof2019ReadProtocolFile( trainProtocolFile19 );
asvspoof_extract_feature_list(trainWavePath, trainFeatPath19, train_fileIds, access_type, feature_type);


show_message('Done!');

end


function asvspoof_extract_feature_list(wavePath, featPath, fileIds, access_type, feature_type)
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
    
