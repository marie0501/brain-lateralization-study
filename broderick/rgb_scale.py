import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def create_rgb_scale_from_array(data):
    # Normaliza los datos para mapearlos a colores
    norm = colors.Normalize(vmin=np.min(data), vmax=np.max(data), clip=True)
    
    # Crea un mapa de colores personalizado, en este caso, de cálido (rojo) a frío (azul)
    cmap = plt.cm.RdBu
    
    # Usa ScalarMappable para mapear los valores normalizados a colores RGB
    mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    # Convierte cada valor en el arreglo a un color RGB
    colors_rgb = [mapper.to_rgba(value)[:3] for value in data]

    # Crea un gráfico mostrando la escala de colores
    n = len(data)
    fig, ax = plt.subplots(1, 1, figsize=(n, 1))
    
    for i, color in enumerate(colors_rgb):
        ax.fill_betweenx(y=[0, 1], x1=i, x2=i + 1, color=color)

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")

    plt.show()

# Ejemplo de uso:
# Puedes cambiar los valores en 'mi_array' según tus datos.
# mi_array = np.linspace(0,10,20)
# create_rgb_scale_from_array(mi_array)
