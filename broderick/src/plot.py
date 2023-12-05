import matplotlib.pyplot as plt
import pandas as pd
from analysis import analysis


df = analysis()

# Filtrar los datos
datos_filtrados = df[(df['freq'] == 1)]

# Crear dos DataFrames separados para "side" igual a 0 y 1
datos_side_0 = datos_filtrados[datos_filtrados['side'] == 0]
datos_side_1 = datos_filtrados[datos_filtrados['side'] == 1]


# Configurar la figura con dos subgráficas
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Graficar la variable "psf" para side = 0
axs[0].scatter(datos_side_0.index, datos_side_0['psf'])
axs[0].set_title('Variable "psf" con freq = 1 y side = 0')

# Graficar la diferencia de "psf" para side = 0
axs[1].scatter(datos_side_1.index, datos_side_1['psf'])
axs[1].axhline(0, color='red', linestyle='--', linewidth=2)  # Línea horizontal en y=0
axs[1].set_title('Variable de "psf" con freq = 1 y side = 1')

# Ajustar el diseño para evitar superposiciones
plt.tight_layout()

# Mostrar el gráfico
plt.show()