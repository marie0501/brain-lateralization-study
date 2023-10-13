% Define el radio de la esfera
radio = 1.0; % Cambia el valor según el radio que desees

% Define la cantidad de puntos para la esfera (mayor número para una superficie más suave)
resolucion = 50; % Cambia el valor según la resolución que desees

% Genera una malla de puntos en la superficie de la esfera
[x, y, z] = sphere(resolucion);

% Escala la malla según el radio de la esfera
x = x * radio;
y = y * radio;
z = z * radio;

% Dibuja la esfera
figure;
surf(x, y, z);
axis equal; % Asegura una escala igual en todos los ejes
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Esfera en MATLAB');

% Personaliza la apariencia de la esfera según tus preferencias
