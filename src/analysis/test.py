import numpy as np
import matplotlib.pyplot as plt
from gaussian_processes import gaussian

# Ajustar una función gaussiana a los datos de un hemisferio
# (Estos datos de ejemplo deberían ser reemplazados por tus propios datos)
x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([2, 4, 6, 8, 10])

# Supongamos que ya has ajustado la función gaussiana a los datos y tienes los parámetros óptimos
# Estos parámetros deberían ser reemplazados por los parámetros obtenidos del ajuste
popt = [1, 3, 0.5]

# Generar puntos artificiales en el hemisferio original usando la distribución gaussiana ajustada
x_artificial = np.linspace(min(x_data), max(x_data), 1000)
y_artificial = gaussian(x_artificial, *popt)

# Aplicar simetría para obtener puntos artificiales correspondientes en el otro hemisferio
x_artificial_opposite = -x_artificial  # Invertir las coordenadas x
y_artificial_opposite = y_artificial  # Mantener las coordenadas y

# Visualizar los puntos artificiales en ambos hemisferios
plt.scatter(x_data, y_data, label='Datos Originales')
plt.plot(x_artificial, y_artificial, 'r-', label='Puntos Artificiales (Hemisferio Original)')
plt.plot(x_artificial_opposite, y_artificial_opposite, 'g-', label='Puntos Artificiales (Hemisferio Opuesto)')
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.title('Puntos Artificiales en Ambos Hemisferios')
plt.legend()
plt.grid(True)
plt.show()
