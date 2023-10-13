
[X, Y] = meshgrid(-2:0.1:2, -2:0.1:2); % Genera una cuadrícula de coordenadas
intensidades = sin(X) .* cos(Y); % Ejemplo de intensidades basadas en funciones sinusoidales

% Encuentra la posición del punto con la mayor intensidad
[max_valor, indice_max] = max(intensidades(:));
[x_max, y_max] = ind2sub(size(intensidades), indice_max);

% Muestra el valor máximo y su posición
fprintf('El valor máximo es %f.\n', max_valor);
fprintf('Se encuentra en la fila %d y columna %d.\n', x_max, y_max);

% Define el centroide (reemplaza esto con tus coordenadas)
x_centroide = x_max;
y_centroide = y_max;

% Define un rango de intensidad dentro del cual deseas seleccionar los puntos
intensidad_centroide = max_valor; % Intensidad en el centroide (ejemplo)
rango_intensidad = 0.5; % Cambia este valor según tus necesidades

% Filtrar puntos por intensidad
indices_seleccionados = abs(intensidades - intensidad_centroide) < rango_intensidad;
x_seleccionados = X(indices_seleccionados);
y_seleccionados = Y(indices_seleccionados);

% Encontrar el contorno convexo de los puntos seleccionados
k = boundary(x_seleccionados(:), y_seleccionados(:));

% Graficar el meshgrid con contornos de colores y el contorno convexo
figure;
contourf(X, Y, intensidades, 20, 'LineColor', 'none'); % Gráfico de contornos con 20 niveles

% Resaltar el contorno convexo
hold on;
scatter(X(x_max, y_max), Y(x_max, y_max), 5, 'r', 'filled', 'MarkerEdgeColor', 'k'); % Punto con mayor intensidad en rojo

plot(x_seleccionados(k), y_seleccionados(k), 'r', 'LineWidth', 2); % Contorno convexo en rojo

% Personalizar el gráfico
xlabel('Coordenada X');
ylabel('Coordenada Y');
title('Área alrededor del Centroide con Intensidad Deseada');
colorbar; % Agregar una barra de colores