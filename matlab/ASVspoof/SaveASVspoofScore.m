function SaveASVspoofScore(scorefile, utterance, technique, key, score)
% save scores to disk
    fid = fopen(scorefile, 'w');
    for i=1:length(utterance)
        fprintf(fid,'%s %s %s %.6f\n', utterance{i}, technique{i}, key{i}, score(i));
    end
    fclose(fid);

end
