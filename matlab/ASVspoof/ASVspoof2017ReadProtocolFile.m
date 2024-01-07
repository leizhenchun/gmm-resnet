function [ fileId, speechType, speakerId, phraseId, environmentId, playbackDeviceId, recordingDeviceId ] = ASVspoof2017ReadProtocolFile( filename )

fileID = fopen(filename);
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);

% 1st column: unique file ID
% 2nd column: speech type identifier: genuine means the trial is original speech; spoof means the file is created with replay attack.
% 3rd column: speaker ID
% 4th column: RedDots common phrase ID
% 5th column: Environment ID ('-' for genuine speech)
% 6th column: Playback device ID ('-' for genuine speech)
% 7th column: Recording device ID ('-' for genuine speech)
        
fileId             = protocol{1};
speechType         = protocol{2};
speakerId          = protocol{3};
phraseId           = protocol{4};
environmentId      = protocol{5};
playbackDeviceId   = protocol{6};
recordingDeviceId  = protocol{7};


for i = 1 : length(fileId)
    fileId{i} = fileId{i}(1:length(fileId{i}) - 4);
%     fileId{i} = strrep(fileId{i}, '.wav', '');
end

end