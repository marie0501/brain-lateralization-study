import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

def model(directory):

    df= pd.read_csv(directory)

    rois = ['V1','V2','V3','hV4','VO1','VO2','V3a','V3b','LO1','LO2','TO1','TO2']

    for roi in range(1,13):
       df_roi = df[(df['roi'] == roi) & (df['frequency_class'] == 3)]
       modelo_mixto = smf.mixedlm("psf ~ side * eccentricity_mean", df_roi, groups=df_roi["subject"])
       resultados_modelo = modelo_mixto.fit(maxiter=10000)
       print(roi)
       # Imprimir los resultados del modelo
       print(resultados_modelo.summary())
       print(resultados_modelo.pvalues)        
       # Obtener el resumen del modelo
       summary_text = resultados_modelo.summary().as_text()
       # Especificar la ruta del archivo de texto
       ruta_archivo = 'C:\\Users\\Marie\\Documents\\thesis\\broderick\\results\\test_5_freq_3.txt'
       # Guardar el resumen en el archivo de texto
       with open(ruta_archivo, 'a') as file:
            file.write(f"{rois[roi-1]}\n")            
            file.write(summary_text)
            file.write('\n\n')

directory ="C:\\Users\\Marie\\Documents\\thesis\\broderick\\table1.csv"

model(directory)