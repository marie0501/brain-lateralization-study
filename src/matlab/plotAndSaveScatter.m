function plotAndSaveScatter(titleText, csvFilePath, xColumn, yColumn, xAxisLabel, yAxisLabel, fileName)
    % Read CSV file
    data = readtable(csvFilePath);
    
    % Extract X and Y data from specified columns
    xData = data.(xColumn);
    yData = data.(yColumn);

    % Create scatter plot
    figure;
    densityScatterChart(xData, yData);
    title(titleText);
    xlabel(xAxisLabel);
    ylabel(yAxisLabel);
    
    % Save the figure as an image
    saveas(gcf, fileName);
end



