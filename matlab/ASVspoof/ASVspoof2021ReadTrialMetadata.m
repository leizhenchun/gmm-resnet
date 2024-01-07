function ASVspoof2021ReadTrialMetadata(filename, access_type)

if strcmp(access_type, 'LA')
    [~, ak_seg, ~, ~, ~, ak_key, ~, ak_phase ]= textread(filename, '%s %s %s %s %s %s %s %s');
end

if strcmp(access_type, 'PA')
    [~, ak_seg, ~, ~, ak_key, ~, ak_phase ]= textread(asv_key_file, '%s %s %s %s %s %s %s');
end

if strcmp(access_type, 'DF')
    [~, ck_seg, ~, ~, ~, cm_key, ~, cm_phase ]= textread(cm_key_file, '%s %s %s %s %s %s %s %s');
end

end