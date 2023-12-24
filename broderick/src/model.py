import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt

def model(directory):

    df= pd.read_csv(directory)

    rois = ['V1','V2','V3','hV4','VO1','VO2','V3a','V3b','LO1','LO2','TO1','TO2']

    for roi in range(1,13):
       df_roi = df[(df['roi'] == roi)]
       modelo_mixto = smf.mixedlm("betas ~ side * ecc", df_roi, groups=df_roi["subj"])
       resultados_modelo = modelo_mixto.fit(maxiter=10000)

       # Verificar la normalidad de los residuos
       sm.qqplot(resultados_modelo.resid, line='s')
       plt.show()
       # Visualizar residuos frente a predicciones para homocedasticidad
       sns.scatterplot(x=resultados_modelo.fittedvalues, y=resultados_modelo.resid, hue=df_roi["subj"])
       plt.axhline(y=0, color='black', linestyle='--')
       plt.show()
       print(roi)
       # Imprimir los resultados del modelo
       print(resultados_modelo.summary())
       print(resultados_modelo.pvalues)        
       # Obtener el resumen del modelo
       summary_text = resultados_modelo.summary().as_text()
       # Especificar la ruta del archivo de texto
       ruta_archivo = f'C:\\Users\\Marie\\Documents\\thesis\\broderick\\results\\test_19_table_beta_freq_{freq}_data_full_inferred_varea.txt'
       # Guardar el resumen en el archivo de texto
       with open(ruta_archivo, 'a') as file:
            file.write(f"{rois[roi-1]}\n")            
            file.write(summary_text)
            file.write('\n\n')

freq=1

directory =f"C:\\Users\\Marie\\Documents\\thesis\\broderick\\table_beta_freq_{freq}_data_full_inferred_varea.csv"

model(directory)