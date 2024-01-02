% Main loop
tuning_columns = [1,2,3,5,13,14,18];
df_columns = [1,2,7,12,13];
rois = {'V1', 'V2', 'V3', 'hV4', 'VO1', 'VO2', 'LO1', 'LO2', 'TO1','TO2','V3b','V3a'};

for roi = 5:12
    disp(rois{roi})
    tab = [];
    tuning_folder = ['F:\ds003812-download\results\tuning\tuning_voxels\', num2str(roi), filesep];
    df_folder = ['F:\ds003812-download\results\df\', num2str(roi), filesep];

    % Process left side
    left_tab = processSide(dir(fullfile(tuning_folder, '*.csv')), dir(fullfile(df_folder, '*.csv')), 'left', tuning_columns, df_columns);
    
    % Process right side 
    right_tab = processSide(dir(fullfile(tuning_folder, '*.csv')), dir(fullfile(df_folder, '*.csv')), 'right', tuning_columns, df_columns);
    
    % Combine left and right tabs
    tab = [tab; left_tab; right_tab];

    % Additional cleaning steps
    tab(contains(tab.inf_warning, 'True'), :) = [];
    tab(tab.preferred_period > 29, :) = [];
    tab(tab.eccen > 10, :) = [];

    % Save table
    writetable(tab, ['tables\',rois{roi}, '_table_all_cleaned.csv'])
    save(['tables\',rois{roi}, '_table_all_cleaned.mat'],'tab')
end

%%
xtab = readtable("F:\ds003812-download\analisis\data\tables\all\1_all_tuning_voxels");

% Group by 'voxel' and 'stimulus_superclass'
 tab = groupsummary(xtab, {'voxel', 'stimulus_superclass','varea'}, 'mean', 'DataVariables', 'preferred_period');

% Save table
 writetable(tab, [num2str(roi), '_all_tuning_vbins_grouped.csv'])

 %%

 for iroi = 1:12
     disp(iroi)
     tab = readtable(['C:\Users\Marie\Desktop\test\',num2str(iroi),'_all_tuning_vbins_grouped.csv']);
    % indices = find(strcmp(tab.stimulus_superclass, 'radial'));
    % tab = tab(indices,:);
     mdl=fitlme(tab,'preferred_period~eccen*Side+(1|Subj) + (1|stimulus_superclass)');
     disp(mdl)
     % Obtener y organizar los resultados
     results = table(mdl.Coefficients.Estimate, mdl.Coefficients.SE, mdl.Coefficients.tStat, mdl.Coefficients.pValue, ...
         'RowNames', mdl.CoefficientNames, 'VariableNames', {'Estimate', 'SE', 'tStat', 'pValue'});

     % Guardar la tabla de resultados en un archivo CSV
     writetable(results, ['results_model_class',num2str(iroi),'.csv'], 'WriteRowNames', true);

 end
 