function tab = processSideGrouped(tuning, df, side, tuning_columns, df_columns)
    tab = [];
    
    side_tuning = tuning(contains({tuning.name}, side));
    side_df = df(contains({tuning.name}, side));
       
    for isub = 1:size(side_tuning, 1)
        xtab = readtable(fullfile(deblank(side_tuning(isub,:).folder), side_tuning(isub,:).name));
        [~,idxvx] = sort(xtab.voxel, 'ascend');
        xtab = xtab(idxvx,:);
        xtab = xtab(ismember(xtab.frequency_type, 'local_sf_magnitude'), :);
        xtab = xtab(:,  tuning_columns);
        xtab = addvars(xtab, strcmp(side, 'right')* ones(size(xtab, 1),1), 'NewVariableNames', 'Side');
        xtab = addvars(xtab, isub * ones(size(xtab, 1), 1), 'NewVariableNames', 'Subj');
        
        ytab = readtable(fullfile(deblank(side_df(isub,:).folder), side_df(isub,:).name));
        ytab = ytab(:, df_columns);
        ytab(max(ytab.voxel) - min(ytab.voxel)+2:end,:) = [];
        xtab = join(xtab, ytab, 'Keys', 'voxel');
        tab = [tab; xtab];
    end    
end