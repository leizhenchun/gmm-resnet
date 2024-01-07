function [EER] = computeEERasvspoof2015(scores, spoofingKeys, techniqueIds)

if(isa(scores, 'char'))
    [~, techniqueIds, spoofingKeys, scores] = ReadASVspoofScore(scores);
end

humanscores = scores(strcmp(spoofingKeys, 'human'));
spoofscores = scores(strcmp(spoofingKeys, 'spoof'));
[Pmiss, Pfa] = rocch(humanscores, spoofscores);
EER = rocch2eer(Pmiss, Pfa) * 100;
fprintf('All(human:%d + spoof:%d) EER\t= %.6f%%\n', length(humanscores), length(spoofscores), EER);
    

% humanscores = scores(strcmp(spoofingKeys, 'human'));

techniquelist = unique(techniqueIds);
techniquelist(strcmp(techniquelist, 'human')) = [];
techniquelist = sort(techniquelist);

alleer = zeros(length(techniquelist), 1);
for i = 1 : length(techniquelist)
    spoofscores = scores(strcmp(techniqueIds, techniquelist{i}));
    [Pmiss, Pfa] = rocch(humanscores, spoofscores);
    EER = rocch2eer(Pmiss, Pfa) * 100;
    fprintf('%s(%d)  EER\t= %.6f%%\n', techniquelist{i}, length(spoofscores), EER);
    alleer(i) = EER;
end


fprintf('AVG EER \t= %.6f%%\n', mean(alleer));

end

