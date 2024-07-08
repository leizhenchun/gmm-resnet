function asvspoof21_evaluate_tDCF_score(cm_score, access_type)

% access_type = 'PA';

if ischar(cm_score)
    [eval_seg, cm_score] = textread(cm_score, '%s %f');
end

truth_dir = ['/home/lzc/lzc/ASVspoof2021/ASVspoof2021data/', access_type, '-keys-stage-1/keys'];
asv_key_file = fullfile(truth_dir, 'ASV/trial_metadata.txt');
asv_scr_file = fullfile(truth_dir, 'ASV/ASVTorch_Kaldi/score.txt');
cm_key_file  = fullfile(truth_dir, 'CM/trial_metadata.txt');

if strcmp(access_type, 'LA')
    [~, ak_seg, ~, ~, ~, ak_key, ~, ak_phase ]= textread(asv_key_file, '%s %s %s %s %s %s %s %s');
    [~, as_seg, as_score ]= textread(asv_scr_file, '%s %s %f');
    [~, ck_seg, ~, ~, ~, cm_key, ~, cm_phase ]= textread(cm_key_file, '%s %s %s %s %s %s %s %s');

    evaluate_phase(cm_key, cm_score, cm_phase, ak_key, as_score, ak_phase, 'progress');
    evaluate_phase(cm_key, cm_score, cm_phase, ak_key, as_score, ak_phase, 'eval');
    evaluate_phase(cm_key, cm_score, cm_phase, ak_key, as_score, ak_phase, 'hidden_track');
end

if strcmp(access_type, 'PA')
    [~, ak_seg, ~, ~, ak_key, ~, ak_phase ]= textread(asv_key_file, '%s %s %s %s %s %s %s');
    [~, as_seg, as_score ]= textread(asv_scr_file, '%s %s %f');
    [~, ck_seg, ~, ~, cm_key, ~, cm_phase ]= textread(cm_key_file, '%s %s %s %s %s %s %s');

    evaluate_phase(cm_key, cm_score, cm_phase, ak_key, as_score, ak_phase, 'progress');
    evaluate_phase(cm_key, cm_score, cm_phase, ak_key, as_score, ak_phase, 'eval');
    evaluate_phase(cm_key, cm_score, cm_phase, ak_key, as_score, ak_phase, 'hidden_track_1');
    evaluate_phase(cm_key, cm_score, cm_phase, ak_key, as_score, ak_phase, 'hidden_track_2');
end

if strcmp(access_type, 'DF')
    [~, ck_seg, ~, ~, ~, cm_key, ~, cm_phase ]= textread(cm_key_file, '%s %s %s %s %s %s %s %s');

    compute_eer_phase(cm_key, cm_score, cm_phase, 'progress');
    compute_eer_phase(cm_key, cm_score, cm_phase, 'eval');
    compute_eer_phase(cm_key, cm_score, cm_phase, 'hidden_track');
end

end

function evaluate_phase(cm_key, cm_score, cm_phase, ak_key, as_score, ak_phase, phase)

    ak_idx = strcmp(ak_phase, phase);
    ak_key2     = ak_key(ak_idx);
    as_score2 = as_score(ak_idx);
    cm_idx = strcmp(cm_phase, phase);
    cm_key2 = cm_key(cm_idx);
    cm_score2 = cm_score(cm_idx);

    fprintf('Phase : %-20s', phase);
    asvspoof_evaluate_tDCF_score(cm_key2, cm_score2, ak_key2, as_score2, false);

end

function compute_eer_phase(cm_key, cm_score, cm_phase, phase)
    cm_idx = strcmp(cm_phase, phase);
    cm_key2 = cm_key(cm_idx);
    cm_score2 = cm_score(cm_idx);
    
    bona_cm     = cm_score2(strcmp(cm_key2, 'bonafide'));
    spoof_cm    = cm_score2(strcmp(cm_key2, 'spoof'));

    [eer_cm, ~] = compute_eer(bona_cm, spoof_cm);
    fprintf('Performance : EER = %f%%\n', 100 * eer_cm);

end
