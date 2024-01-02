clear
rois = {'V1', 'V2', 'V3', 'hV4', 'VO1', 'VO2', 'LO1', 'LO2', 'TO1','TO2','V3b','V3a'};
iroi = 12;

%%
csvFilePath = ['F:\ds003812-download\analisis\data\tables\all\',num2str(iroi),'_all_tuning_voxels.csv'];
data = readtable(csvFilePath);

xColumn = 'eccen';
yColumn = 'preferred_period';
xAxisLabel = 'Excentricidad';
yAxisLabel = 'Período Preferido';
titleText = rois{iroi};

% Extract X and Y data from specified columns

%all
% xData = data.(xColumn);
% yData = data.(yColumn);

% figure;
% densityScatterChart(xData, yData);
% title(titleText);
% xlabel(xAxisLabel);
% ylabel(yAxisLabel);

% preferref period < 2
% pf_2_indices = find(data.preferred_period < 2 & data.preferred_period > 0.16);
% xData = data.(xColumn)(pf_2_indices);
% yData = data.(yColumn)(pf_2_indices);

% figure;
% densityScatterChart(xData, yData);

% eccentricity histogram
figure;
histogram(data.eccen, 'BinEdges', linspace(min(data.eccen), max(data.eccen), 20), 'Normalization', 'probability');

title(titleText);
xlabel(xAxisLabel);
% ylabel(yAxisLabel);
grid on
% ylim([0.16,2]);

%fileName = [rois{iroi},'_',xAxisLabel,'_',yAxisLabel,'.png'];
%plotAndSaveScatter(titleText, csvFilePath, xColumn, yColumn, xAxisLabel, yAxisLabel, fileName);


%%
rois = {'V1', 'V2', 'V3', 'hV4', 'VO1', 'VO2', 'LO1', 'LO2', 'TO1','TO2','V3b','V3a'};
colors = {'red','darkorange','gold','lightskyblue','steelblue','lightpink','orchid','darkviolet','indigo','black', 'yellowgreen','limegreen'};
rgb = {[1.0, 0.0, 0.0], [1.0, 0.5490196078431373, 0.0], [1.0, 0.8431372549019608, 0.0], [0.5294117647058824, 0.807843137254902, 0.9803921568627451], [0.27450980392156865, 0.5098039215686274, 0.7058823529411765], [1.0, 0.7137254901960784, 0.7568627450980392], [0.8549019607843137, 0.4392156862745098, 0.8392156862745098], [0.5803921568627451, 0.0, 0.8274509803921568], [0.29411764705882354, 0.0, 0.5098039215686274], [0.0, 0.0, 0.0], [0.6039215686274509, 0.803921568627451, 0.19607843137254902], [0.19607843137254902, 0.803921568627451, 0.19607843137254902]};
columnName = 'eccen';
xAxisLabel = 'Excentricidad';

for iroi=1:12
    disp(iroi)
    tableDirectory = ['F:\ds003812-download\analisis\data\tables\all\',num2str(iroi),'_all_tuning_voxels.csv'];;  % Reemplazar con la ruta de tu archivo
    saveName = [rois{iroi},'_eccen_hist_all_count'];
    plotHistogramWithCurve(tableDirectory, columnName, rgb{iroi}, rois{iroi}, xAxisLabel, saveName);
end