function [utterance, technique, key, score] = ReadASVspoofScore(scorefile)
fileID = fopen(scorefile);
protocol = textscan(fileID, '%s%s%s%.6f');
fclose(fileID);

% (utterance[i], system_id[i], key[i], score[i])

utterance     = protocol{1};    
technique     = protocol{2};
key           = protocol{3};
score         = protocol{4};

end