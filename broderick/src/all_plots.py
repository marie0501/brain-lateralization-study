import pandas as pd
import matplotlib.pyplot as plt

def plot_roi_values(csv_file):
    # Cargar el archivo CSV en un DataFrame
    df = pd.read_csv(csv_file)

    # Filtrar los valores de 'roi' igual a 1
    df_roi_1 = df[df['roi'] == 1]

    # Verificar si existe la columna 'subj'
    if 'subj' in df.columns:
        # Promediar los valores por 'subj'
        df_roi_1_avg = df_roi_1.groupby('subj')['freq'].mean()

        # Graficar los resultados
        plt.bar(df_roi_1_avg.index, df_roi_1_avg)
        plt.xlabel('Subj')
        plt.ylabel('Frecuencia Promedio para roi=1')
        plt.title('Frecuencia Promedio para roi=1 por Subj')
        plt.show()
    else:
        # Si no hay información sobre subj, simplemente graficar las frecuencias
        plt.bar(df_roi_1['freq'], df_roi_1['roi'])
        plt.xlabel('Frecuencia')
        plt.ylabel('Roi')
        plt.title('Valores de Roi=1 para Diferentes Frecuencias')
        plt.show()

# Ejemplo de uso
csv_file_path = 'ruta/a/tu/archivo.csv'
plot_roi_values(csv_file_path)
