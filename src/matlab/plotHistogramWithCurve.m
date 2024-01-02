function plotHistogramWithCurve(tableDirectory, columnName, histogramColor, plotTitle, xAxisLabel, saveFileName)
    % Load the table from the file
    myTable = readtable(tableDirectory);

    % Extract data from the specified column
    columnData = myTable.(columnName);

    % Create a histogram of the column
    figure;
    h = histogram(columnData, 'NumBins', 11, 'FaceColor', histogramColor);
    hold on;

    % Calculate density estimation (smooth curve)
    x_vals = linspace(min(columnData), max(columnData), 100);
    y_vals = ksdensity(columnData, x_vals);

    % Scale the density curve to match the count normalization
    scale_factor = sum(h.BinCounts) / trapz(x_vals, y_vals);
    y_vals_scaled = y_vals * scale_factor;

    % Plot the smooth curve over the histogram
    plot(x_vals, y_vals_scaled, 'LineWidth', 2, 'Color', histogramColor);

    % Customize the plot
    xlabel(xAxisLabel);
    ylabel('Frecuencia/densidad relativa');
    title(plotTitle);
    legend('Histograma', 'Curva de densidad');
    grid on;
    hold off;

    % Guardar la gráfica como un archivo PNG
    saveas(gcf, saveFileName, 'png');
end

