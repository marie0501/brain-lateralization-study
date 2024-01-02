function tab = processSide(tuning, df, side, tuning_columns, df_columns)
    tab = [];
    
    side_tuning = tuning(contains({tuning.name}, side));
    side_df = df(contains({tuning.name}, side));
       
    for isub = 1:size(side_tuning, 1)
        % read tuning table
        tuning_tab = readtable(fullfile(deblank(side_tuning(isub,:).folder), side_tuning(isub,:).name));
        [~,idxvx] = sort(tuning_tab.voxel, 'ascend');  % sort tuning table by voxel
        tuning_tab = tuning_tab(idxvx,:);
        tuning_tab = tuning_tab(ismember(tuning_tab.frequency_type, 'local_sf_magnitude'), :); % reduce types of frequency
        tuning_tab = tuning_tab(:,  tuning_columns); % keep useful fields
        [~, t_unique_rows, ~] = unique(tuning_tab, 'rows', 'stable');  % identify repeted rows
        tuning_tab = tuning_tab(t_unique_rows, :); % eliminate repeted rows
        tuning_tab = addvars(tuning_tab, strcmp(side, 'right')* ones(size(tuning_tab, 1),1), 'NewVariableNames', 'side');
        tuning_tab = addvars(tuning_tab, isub * ones(size(tuning_tab, 1), 1), 'NewVariableNames', 'subj');
        
        % read df table
        df_tab = readtable(fullfile(deblank(side_df(isub,:).folder), side_df(isub,:).name));
        df_tab=renamevars(df_tab, {'GLM_R2'}, {'gml_r2'});
        df_tab = df_tab(:, df_columns); % keep useful fields        
        [~, df_unique_rows, ~] = unique(df_tab, 'rows', 'stable'); % identify repeted rows
        df_tab = df_tab(df_unique_rows, :); % eliminate repeated rows
        
        % join tuning and df tables
        tuning_tab = join(tuning_tab, df_tab, 'Keys', 'voxel','LeftVariables',[1,3:7,8,9]);

        % append to general table
        tab = [tab; tuning_tab];
    end
end