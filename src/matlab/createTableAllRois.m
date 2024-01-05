rois = {'V1', 'V2', 'V3', 'hV4', 'VO1', 'VO2', 'LO1', 'LO2', 'TO1','TO2','V3b','V3a'};
tab = [];

for iroi=1:12
    disp(rois{iroi})    
    folder = ['C:\Users\Marie\Documents\thesis\tables\', rois{iroi}, '_table_all_cleaned.csv'];
    data = readtable(folder);
    tab = [tab;data];
end

writetable(tab, ['tables\','all_rois_table_cleaned.csv'])
