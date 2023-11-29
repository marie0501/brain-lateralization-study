import numpy as np

# Número de elementos en la matriz
num_elementos = 142997

# Generar tonalidades de rojo utilizando linspace
tonalidades_rojas = np.linspace(0, 1, num_elementos)

# Crear la matriz RGBA con tonalidades de rojo
colores_rgba = np.zeros((num_elementos, 4))
colores_rgba[:, 0] = tonalidades_rojas  # Componente roja
colores_rgba[:, 3] = 1.0  # Componente alfa (opacidad)

# Imprimir algunos valores de la matriz RGBA como ejemplo
print(colores_rgba[:5, :])
print(colores_rgba)