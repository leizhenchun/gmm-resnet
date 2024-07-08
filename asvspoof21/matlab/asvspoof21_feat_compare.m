function asvspoof21_feat_compare()


asvspoof19_compare_feat();
% asvspoof21_extract_feat('LA', 'LFCCMatlab');
% asvspoof21_extract_feat('DF', 'LFCCMatlab');

% asvspoof19_extract_feat('PA', 'logspec256');
% asvspoof21_extract_feat('PA', 'logspec256');


end

function asvspoof21_compare_feat(access_type, feature_type)

feat_acc = [feature_type, '_', access_type];

asvspoofRootPath = '/home/lzc/lzc/ASVspoof2021/';

dataPath21     = fullfile(asvspoofRootPath, 'ASVspoof2021data', ['ASVspoof2021_', access_type, '_eval']);
featPath21     = fullfile(asvspoofRootPath, 'ASVspoof2021feat/', feature_type);


evalProtocolFile21 = fullfile(dataPath21, ['ASVspoof2021.', access_type, '.cm.eval.trl.txt']);
evalWavePath21     = fullfile(dataPath21, 'flac');
evalFeatPath21     = fullfile(featPath21, [feat_acc, '_eval'], 'Original');

%%  eval protocol
show_message(['Extracting features for ASVspoof2021 EVAL data : ', feat_acc]);
[eval_fileIds, ~ ] = ASVspoof2019ReadProtocolFile( evalProtocolFile21 );
asvspoof_extract_feature_list(evalWavePath21, evalFeatPath21, eval_fileIds, access_type, feature_type);


show_message('Done!');

end



function asvspoof19_compare_feat()
access_type = 'LA';
aug_method = 'Original';

feat_acc1 = ['LFCCMatlab', '_', access_type];
feat_acc2 = ['LFCCBP', '_', access_type];

asvspoofRootPath = '/home/lzc/lzc/ASVspoof2021/';

featRootPath1     = fullfile(asvspoofRootPath, 'ASVspoof2019feat', 'LFCCMatlab');
featRootPath2     = fullfile(asvspoofRootPath, 'ASVspoof2019feat', 'LFCCBP');

dsPath19       = fullfile(asvspoofRootPath, 'DS_10283_3336/');
protocolPath19 = fullfile(dsPath19, access_type, ['ASVspoof2019_', access_type, '_cm_protocols']);
trainProtocolFile19 = fullfile(protocolPath19, ['ASVspoof2019.', access_type, '.cm.train.trn.txt']);
devProtocolFile19   = fullfile(protocolPath19, ['ASVspoof2019.', access_type, '.cm.dev.trl.txt']);
evalProtocolFile19  = fullfile(protocolPath19, ['ASVspoof2019.', access_type, '.cm.eval.trl.txt']);


%% Feature extraction for training data
show_message(['Compare features for ASVspoof2019 TRAIN data : ', feat_acc1]);

featPath1 = fullfile(featRootPath1, [feat_acc1, '_train'], aug_method);
featPath2 = fullfile(featRootPath2, [feat_acc2, '_train'], aug_method);
[train_speakerIds, train_fileIds, ~ ] = ASVspoof2019ReadProtocolFile( trainProtocolFile19 );
asvspoof_compare_feature_list(featPath1, featPath2, train_fileIds);


%%  development protocol
show_message(['Compare features for ASVspoof2019 DEV data : ', feat_acc1]);

featPath1 = fullfile(featRootPath1, [feat_acc1, '_dev'], aug_method);
featPath2 = fullfile(featRootPath2, [feat_acc2, '_dev'], aug_method);
[dev_speakerIds, dev_fileIds, ~ ] = ASVspoof2019ReadProtocolFile( devProtocolFile19 );
asvspoof_compare_feature_list(featPath1, featPath2, dev_fileIds);


%%  eval protocol
show_message(['Compare features for ASVspoof2019 EVAL data : ', feat_acc1]);

featPath1 = fullfile(featRootPath1, [feat_acc1, '_eval'], aug_method);
featPath2 = fullfile(featRootPath2, [feat_acc2, '_eval'], aug_method);
[eval_speakerIds, eval_fileIds, ~ ] = ASVspoof2019ReadProtocolFile( evalProtocolFile19 );
asvspoof_compare_feature_list(featPath1, featPath2, eval_fileIds);


show_message('Done!');

end




function asvspoof_compare_feature_list(featPath1, featPath2, fileIds)
    fileCount = length(fileIds);
    for i=1:fileCount
        fileId = fileIds{i};
%         disp(fileId);
        
        filename1 = fullfile(featPath1, [fileIds{i}, '.h5']);
        feature1 = h5read(filename1, '/data')';
        
        filename2 = fullfile(featPath2, [fileIds{i}, '.h5']);
        feature2 = h5read(filename2, '/data')';
        
        if (~all(feature1(:) == feature2(:)))
            disp([fileId, ':', num2str(sum(sum(abs(feature1-feature2))))]);
        end
        
        
        
%         feature1 = asvspoof_extract_feature(filename, access_type, feature_type);
% 
%         h5create(fullfile(featPath2, [fileIds{i}, '.h5']), '/data', size(feature), 'Datatype', 'single');
%         h5write(fullfile(featPath2, [fileIds{i}, '.h5']), '/data', single(feature));
%         
%         if mod(i, 10000) == 0
%             show_message( ['extract ', feature_type, ': ', num2str(100 * i / fileCount), '%   ']);
%         end
    end
end
