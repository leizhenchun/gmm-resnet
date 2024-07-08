function [speakerId, audioFile, environmentId, attackId, key] = ASVspoof2019ReadProtocolFile(protocolFile)

fileID = fopen(protocolFile);
protocol = textscan(fileID, '%s%s%s%s%s');
fclose(fileID);

% SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY
% SPEAKER_ID AUDIO_FILE_NAME ENVIRONMENT_ID ATTACK_ID KEY
speakerId = protocol{1};
audioFile = protocol{2};
environmentId = protocol{3};
attackId = protocol{4};
key = protocol{5};

end