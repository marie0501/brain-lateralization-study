import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("C:\\Users\\Marie\\Documents\\thesis\\broderick\\table_size_prf.csv")

# Filtrar los datos
datos_filtrados = df[(df['roi'] == 1) & (df['subj'] == 1)]

# Configurar el estilo de seaborn (opcional)
sns.set(style="whitegrid")

# Crear un boxplot con dos cajas al lado de la otra
plt.figure(figsize=(10, 6))
sns.boxplot(x='size', y='ecc',hue='side', data=datos_filtrados)
plt.title('Boxplot de "psf" con freq = 1 para cada valor de "side"')
plt.show()
