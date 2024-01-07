function [EER] = computeEERasvspoof2017(scorefile)

[~, techniqueIds, spoofingKeys, scores] = ReadASVspoofScore(scorefile);


humanscores = scores(strcmp(spoofingKeys, 'genuine'));
spoofscores = scores(strcmp(spoofingKeys, 'spoof'));
[Pmiss, Pfa] = rocch(humanscores, spoofscores);
EER = rocch2eer(Pmiss, Pfa) * 100;
fprintf('All(human:%d + spoof:%d) EER is %.6f%%\n', length(humanscores), length(spoofscores), EER);


% techniquelist = unique(techniqueIds);
% techniquelist(strcmp(techniquelist, '-')) = [];
% techniquelist = sort(techniquelist);
% 
% alleer = zeros(length(techniquelist), 1);
% for i = 1 : length(techniquelist)
%     spoofscores = scores(strcmp(techniqueIds, techniquelist{i}));
%     [Pmiss, Pfa] = rocch(humanscores, spoofscores);
%     EER = rocch2eer(Pmiss, Pfa) * 100;
%     fprintf('%s(%d)  EER is %.6f%%\n', techniquelist{i}, length(spoofscores), EER);
%     alleer(i) = EER;
% end
% fprintf('AVG EER is %.6f%%\n', mean(alleer));

end

