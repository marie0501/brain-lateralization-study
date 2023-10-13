% Cargar tus datos (supongamos que tienes una matriz llamada 'intensidades' de 360 x 60)
% Si no tienes tus datos reales, puedes crear una matriz de ejemplo como esta:
intensidades = rand(360, 60); % Cambia esto por tus propios datos

% Definir rangos de excentricidad y ángulo polar
r = linspace(0, 10, 60);  % Rango de excentricidad de 0 a 10 grados
theta = linspace(0, 2*pi, 360);  % Rango de ángulo polar de 0 a 2*pi radianes

% Crear una cuadrícula de coordenadas r y theta
[R, Theta] = meshgrid(r, theta);

% Convertir coordenadas polares a coordenadas cartesianas
X = R .* cos(Theta);
Y = R .* sin(Theta);

% Interpolar los datos en coordenadas cartesianas
[Xq, Yq] = meshgrid(linspace(-10, 10, 360), linspace(-10, 10, 60));  % Cuadrícula regular en coordenadas cartesianas
intensidades_interp = interp2(X, Y, intensidades, Xq, Yq, 'spline');

% Crear el mapa de calor polar usando polarplot
figure;
polarplot(Theta, R, 'r.');  % Dibujar los puntos de excentricidad y ángulo polar
hold on;

% Añadir el mapa de calor con intensidades interpoladas
polar_ax = gca;
intensity_max = max(intensidades_interp(:));
intensity_min = min(intensidades_interp(:));

for i = 1:length(r)
    color = (intensidades_interp(:, i) - intensity_min) / (intensity_max - intensity_min);  % Escalar la intensidad al rango [0, 1]
    polarplot(Theta(:, i), R(:, i), 'color', [color, 0, 1 - color], 'LineWidth', 2);
end

% Añadir una retícula polar
thetaticks(0:45:315);  % Colocar marcas en la escala de ángulo polar
rticks(0:2:10); % Colocar marcas en la escala de excentricidad

% Añadir etiquetas y título
title('Mapa de campo visual con intensidades');

% Mostrar el resultado

