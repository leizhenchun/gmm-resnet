function [ speakerId, fileId, techniqueIdentifier, spoofingKey ] = ASVspoof2015ReadProtocolFile( filename )

fileID = fopen(filename);
protocol = textscan(fileID, '%s%s%s%s');
fclose(fileID);

% 1st column: target speaker ID
% 2nd column: unique file ID
% 3rd column: spoofing technique identifier: human means the trial is natural speech; 
%             S1-S5 means the trial is generated from one of five speech synthesis/voice conversion systems, 
%             and they are assumed as known attacks.
% 4th column: keys to indicate the trial is human speech or spoofed speech.
    
speakerId           = protocol{1};    
fileId              = protocol{2};
techniqueIdentifier = protocol{3};
spoofingKey         = protocol{4};


end