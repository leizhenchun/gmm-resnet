function [asv_attacks, asv_key, asv_score] = ASVspoof2019ReadASVScoreFile(filename)

fileID = fopen(filename);
protocol = textscan(fileID, '%s %s %f');
fclose(fileID);

% asv_attacks, asv_key, asv_score
asv_attacks = protocol{1};
asv_key = protocol{2};
asv_score = protocol{3};


end