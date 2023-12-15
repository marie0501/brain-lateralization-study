import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from scipy.optimize import curve_fit
from src.prf import load_all_prf_data

import statsmodels.api as sm
import statsmodels.formula.api as smf

bins=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
bincenter= (np.array(bins[:-1]) + np.array(bins[1:]))/2


x = np.zeros((len(bincenter), 10))

# Calcular valores para cada fila de x
for i in range(len(bincenter)):
    x[i, :] = np.array([6, 8, 11, 16, 23, 32, 45, 64, 91, 128]) / bincenter[i]

rois_labels = ['V1','V2','V3','hV4','VO1','VO2','V3a','V3b','LO1','LO2','TO1','TO2']

betas= load_all_prf_data("F:\\ds003812-download\\derivatives\\processed\\betas","smoothed_betas")
ecc=load_all_prf_data("F:\\ds003812-download\\derivatives\\prf_solutions\\all", "full-eccen")
roi=load_all_prf_data("F:\\ds003812-download\\derivatives\\prf_solutions\\all", "inferred_varea")
print(len(betas))
nsub = 12

subj = []
bands = []
freq = []
side = []
rois = []


Lpeak = np.zeros((nsub, len(bincenter)))
Rpeak = np.zeros((nsub, len(bincenter)))

for iroi in range(len(rois_labels)):
    for isub in range(nsub):
        atlasL=roi[isub][0]
        atlasR = roi[isub][1]
        idxV1L = np.where(atlasL == iroi)
        idxV1R = np.where(atlasR == iroi)
        eccL= list(ecc[isub][0])
        eccR=list(ecc[isub][1])
        idxL = np.digitize(eccL, bins)
        idxR = np.digitize(eccR, bins)
        idxL = idxL[idxV1L]
        idxR = idxR[idxV1R]
        num_elements_eccL = len(eccL)
        # Dividir betas en betasR y betasL
        betasL = betas[isub][:,:num_elements_eccL]
        betasR = betas[isub][:,num_elements_eccL:]
        # Filtrar betasR y betasL según idxV1R e idxV1L respectivamente
        betasR = np.squeeze(betasR[:,idxV1R])
        betasL = np.squeeze(betasL[:,idxV1L])
        print(betasL.shape)
        # Número de elementos en bincenter
        num_bincenter = len(bincenter)
        # Inicializar matrices para bL y bR
        bL = np.zeros((num_bincenter, 10))
        bR = np.zeros((num_bincenter, 10))
        print(bL.shape)
        # Calcular bL y bR para cada valor de bincenter
        for i in range(num_bincenter):
            # Calcular medianas para bL y bR
            median_betasL_1_10 = np.median(betasL[:10,idxL == i + 1], axis=1)
            median_betasL_11_20 = np.median(betasL[10:20,idxL == i + 1], axis=1)
            median_betasR_1_10 = np.median(betasR[:10,idxR == i + 1], axis=1)
            median_betasR_11_20 = np.median(betasR[10:20,idxR == i + 1], axis=1)
            print(median_betasL_1_10.shape)
            # Calcular bL e bR
            bL[i, :] = (median_betasL_1_10 + median_betasL_11_20) / 2
            bR[i, :] = (median_betasR_1_10 + median_betasR_11_20) / 2


        # Calcular Lpeak y Rpeak para cada sujeto y valor de bincenter
    
        for i in range(num_bincenter):
            # Encontrar el índice del valor máximo en bL y bR
            ii_L = np.argmax(bL[i, :])
            ii_R = np.argmax(bR[i, :])
            # Asignar valores a Lpeak y Rpeak
            Lpeak[isub, i] = x[i, ii_L]
            Rpeak[isub, i] = x[i, ii_R]

   

    
        subj.extend([isub +1] * 14)
        bands.extend(list(range(1, 15)))
        freq.extend(np.sqrt(Lpeak[isub, :]))
        side.extend([0] * 14)
        rois.extend([iroi + 1] * 14)
    
        subj.extend([isub+1] * 14)
        bands.extend(list(range(1, 15)))
        freq.extend(np.sqrt(Rpeak[isub, :]))
        side.extend([1] * 14)
        rois.extend([iroi + 1] * 14)

    print(f"subj: {len(subj)}")
    print(f"bands: {len(bands)}")
    print(f"freq: {len(freq)}")
    print(f"side: {len(side)}")

data = {'subj': subj, 'bands': bands, 'freq': freq, 'side': side, 'rois':rois}
tab = pd.DataFrame(data)

tab.to_csv("matlab_table_smoothed_full_eccen_bayesian_area.csv", index=False)



# modelo_mixto = smf.mixedlm("freq ~ side * bands", tab, groups=tab["subj"])
# resultados_modelo = modelo_mixto.fit(maxiter=10000)
# print(rois[iroi-1])
# # Imprimir los resultados del modelo
# print(resultados_modelo.summary())
# print(resultados_modelo.pvalues)        

# summary_text = resultados_modelo.summary().as_text()
# # Especificar la ruta del archivo de texto
# ruta_archivo = 'C:\\Users\\Marie\\Documents\\thesis\\broderick\\results\\test_8_matlab_table.txt'
# # Guardar el resumen en el archivo de texto
# with open(ruta_archivo, 'a') as file:
#      file.write(f"{rois[iroi-1]}\n")            
#      file.write(summary_text)
#      file.write('\n\n')