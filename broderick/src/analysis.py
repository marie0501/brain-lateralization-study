from psf import preferred_spatial_frequency
from prf import load_all_prf_data
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

def analysis():

    psf= preferred_spatial_frequency("F:\\ds003812-download\\derivatives\\processed\\betas")
    ecc=load_all_prf_data("F:\\ds003812-download\\derivatives\\prf_solutions\\all", "eccen")
    roi=load_all_prf_data("F:\\ds003812-download\\derivatives\\prf_solutions\\all", "benson14_varea")
    print(psf[0])
    psf_result = []
    ecc_result=[]
    roi_result=[]
    side_result = []
    subj_result = []
    freq_result = []

    for sub in range(len(psf)):
        psf_l = psf[sub][:,:len(ecc[sub][0])]
        psf_r = psf[sub][:,len(ecc[sub][0]):]
        #psf_l = psf[sub][:,0:len(ecc[sub][0])]
        #psf_r = psf[sub][:,0:len(ecc[sub][1])]
        print(psf_r.shape)
        print(len(ecc[sub][1]))
        for freq in range(4):

            # left
            psf_result.extend(psf_l[freq,:])
            roi_result.extend(roi[sub][0])
            ecc_result.extend(ecc[sub][0])
            freq_result.extend(np.ones(ecc[sub][0].shape)*(freq+1))
            side_result.extend(np.zeros(ecc[sub][0].shape))
            subj_result.extend(np.ones(ecc[sub][0].shape)*(sub+1))

            # rigth
            psf_result.extend(psf_r[freq,:])
            roi_result.extend(roi[sub][1])
            ecc_result.extend(ecc[sub][1])
            freq_result.extend(np.ones(ecc[sub][1].shape)*(freq+1))
            side_result.extend(np.ones(ecc[sub][1].shape))
            subj_result.extend(np.ones(ecc[sub][1].shape)*(sub+1))


    data = {'psf':psf_result, 'roi':roi_result,'ecc':ecc_result,'freq':freq_result,'side':side_result,'subj':subj_result}

    df=pd.DataFrame(data)
    df.dropna()
    df.to_csv('all.csv', index=False)

    return df
            
def model():

    df = analysis()

    for roi in range(1,13):
        df_roi = df[(df['roi'] == roi) & (df['freq'] == 4)].dropna()
        modelo_mixto = smf.mixedlm("psf ~ side * ecc", df_roi, groups=df_roi["subj"])
        resultados_modelo = modelo_mixto.fit(maxiter=5000)

        print(roi)
        # Imprimir los resultados del modelo
        print(resultados_modelo.summary())
        print(resultados_modelo.pvalues)        

        # Obtener el resumen del modelo
        summary_text = resultados_modelo.summary().as_text()

        # Especificar la ruta del archivo de texto
        ruta_archivo = 'C:\\Users\\Marie\\Documents\\thesis\\broderick\\results\\freq_4_test_8.txt'

        # Guardar el resumen en el archivo de texto
        with open(ruta_archivo, 'a') as file:
            file.write(summary_text)
            file.write('\n\n')

    
#model()
analysis()