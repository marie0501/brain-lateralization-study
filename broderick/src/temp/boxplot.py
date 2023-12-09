import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from analysis import analysis

df = analysis()

# Filtrar los datos
datos_filtrados = df[df['freq'] == 1]

# Configurar el estilo de seaborn (opcional)
sns.set(style="whitegrid")

# Crear un boxplot con dos cajas al lado de la otra
plt.figure(figsize=(10, 6))
sns.boxplot(x='side', y='psf', data=datos_filtrados)
plt.title('Boxplot de "psf" con freq = 1 para cada valor de "side"')
plt.show()
