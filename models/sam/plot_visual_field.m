% Definir rangos de excentricidad y ángulo polar
r = linspace(0, 10, 100); % Rango de excentricidad de 0 a 10 grados
theta = linspace(0, 2*pi, 100);  % Rango de ángulo polar de 0 a 2*pi radianes

% Crear una cuadrícula de coordenadas r y theta
[R, Theta] = meshgrid(r, theta);

% Calcular la sensibilidad visual en función de r y theta (esto es un ejemplo, debes proporcionar tus propios datos)
sensibilidad = sin(Theta) .* exp(-R/5);  % Función de ejemplo

% Crear el mapa de calor polar usando polarplot
figure;
polarplot(theta, r, 'r.');  % Dibujar los puntos de excentricidad y ángulo polar
hold on;

% % Añadir el mapa de calor
% polar_ax = gca;
% sensibility_max = max(sensibilidad(:));
% polar_contour = polarplot(theta, r, 'k');  % Líneas de contorno
% polar_contour.LineWidth = 1.5;
% for i = 1:length(r)
%     polarplot([theta, theta(1)], [r(i), r(i)], 'color', [1, sensibilidad(i) / sensibility_max, 0], 'LineWidth', 2);
% end

% Añadir una retícula polar
thetaticks(0:45:315);  % Colocar marcas en la escala de ángulo polar
rticks(0:2:10); % Colocar marcas en la escala de excentricidad

% Añadir etiquetas y título
% title('Mapa de calor del campo visual');

% Mostrar el resultado


% Definir rangos de excentricidad y ángulo polar
r = linspace(0, 10, 100);  % Rango de excentricidad de 0 a 10 grados
theta = linspace(0, 2*pi, 100);  % Rango de ángulo polar de 0 a 2*pi radianes

% Crear una cuadrícula de coordenadas r y theta
[R, Theta] = meshgrid(r, theta);

% Calcular las intensidades en función de r y theta (esto es un ejemplo, debes proporcionar tus propios datos)
intensidades = sin(Theta) .* exp(-R/5);  % Función de ejemplo

% Crear el mapa de calor polar usando polarplot
figure;
polarplot(theta, r, 'r.');  % Dibujar los puntos de excentricidad y ángulo polar
hold on;

% Añadir el mapa de calor con intensidades
polar_ax = gca;
intensity_max = max(intensidades(:));
intensity_min = min(intensidades(:));

for i = 1:length(r)
    color = (intensidades(i) - intensity_min) / (intensity_max - intensity_min);  % Escalar la intensidad al rango [0, 1]
    polarplot([theta, theta(1)], [r(i), r(i)], 'color', [color, 0, 1 - color], 'LineWidth', 2);
end

% Añadir una retícula polar
thetaticks(0:45:315);  % Colocar marcas en la escala de ángulo polar
rticks(0:2:10); % Colocar marcas en la escala de excentricidad

% Añadir etiquetas y título
title('Mapa de campo visual con intensidades');

% Mostrar el resultado





