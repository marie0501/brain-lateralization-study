% Ejemplo de una superficie plana y su intensidad (reemplaza con tus datos)
[X, Y] = meshgrid(-2:0.1:2, -2:0.1:2); % Genera una cuadrícula de coordenadas
Z_superficie = sin(X) .* cos(Y); % Intensidad en la superficie (ejemplo)

% Calcula las coordenadas polares
R = sqrt(X.^2 + Y.^2);
Theta = atan2(Y, X);

% Calcula las coordenadas cartesianas en la esfera
radio_esfera = 1.0; % Cambia el radio de la esfera según tus necesidades
X_esfera = radio_esfera * sin(Theta) .* cos(R);
Y_esfera = radio_esfera * sin(Theta) .* sin(R);
Z_esfera = radio_esfera * cos(Theta);

% Usa la misma intensidad en la esfera
intensidad_esfera = Z_superficie;

% Crea un gráfico de superficie 3D en la esfera
figure;
surf(X_esfera, Y_esfera, Z_esfera, intensidad_esfera);

% Personaliza el gráfico
xlabel('Coordenada X');
ylabel('Coordenada Y');
zlabel('Coordenada Z');
title('Superficie Esférica en MATLAB');

% Puedes personalizar la apariencia del gráfico según tus necesidades

