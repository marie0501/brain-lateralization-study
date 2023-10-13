% Ejemplo de matrices de coordenadas y valores (reemplaza con tus datos)
[X, Y] = meshgrid(-2:0.1:2, -2:0.1:2); % Genera una cuadrícula de coordenadas
Z = sin(X) .* cos(Y); % Ejemplo de intensidad basada en funciones sinusoidales

% Crea un gráfico de contorno con colores
figure;
contourf(X, Y, Z, 20, 'LineColor', 'none');

% Personaliza el gráfico
xlabel('Coordenada X');
ylabel('Coordenada Y');
title('Superficie 2D en MATLAB');

% Agrega una barra de colores
colorbar;

% Puedes personalizar la apariencia del gráfico según tus necesidades


