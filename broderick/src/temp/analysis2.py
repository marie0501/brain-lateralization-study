from psf import preferred_spatial_frequency
from prf import load_all_prf_data
from betas import get_betas
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

def analysis():

    psf= load_all_prf_data("F:\\ds003812-download\\derivatives\\processed\\betas","betas")
    ecc=load_all_prf_data("F:\\ds003812-download\\derivatives\\prf_solutions\\all", "eccen")
    roi=load_all_prf_data("F:\\ds003812-download\\derivatives\\prf_solutions\\all", "benson14_varea")
    print(len(psf))
    print(psf[0].shape)
    psf_result = []
    ecc_result=[]
    roi_result=[]
    side_result = []
    subj_result = []
    freq_result = []

    arr=np.linspace(1.5,25.5,25)
    ecc_ranges = list(zip(arr, arr[1:]))
    
    freq_ring = []

    betas = []

    for sub in range(len(psf)):
        psf_l = psf[sub][:,:len(ecc[sub][0])]
        psf_r = psf[sub][:,len(ecc[sub][0]):]
        indices_rois_left = get_indices_rois(roi[sub][0])
        indices_rois_right = get_indices_rois(roi[sub][1])

        beta_left = get_all_values_betas(indices_rois_left,psf_l)
        beta_right = get_all_values_betas(indices_rois_right,psf_r)
        
        ecc_per_roi_left = get_all_values(indices_rois_left,ecc[sub][0])
        ecc_per_roi_right = get_all_values(indices_rois_right,ecc[sub][1])

        # arreglo dividido por rois, dividido por rangos de excentricidad
        indices_ecc_left = get_all_roi_indices_ecc(ecc_per_roi_left,ecc_ranges)
        indices_ecc_right = get_all_roi_indices_ecc(ecc_per_roi_right,ecc_ranges)

        betas_rings_left = []
        betas_rings_right = []


        for iroi in range(len(indices_ecc_left)):
            temp_left =[]
            temp_right =[]
            for ring in range(len(roi)):
                temp_left.append(beta_left[iroi][:,indices_ecc_left[iroi][ring]])
                temp_right.append(beta_right[iroi][:,indices_ecc_right[iroi][ring]])
            betas_rings_left.append(temp_left)
            betas_rings_right.append(temp_right)
        
        
        mean_betas_left = mean_all_rings(betas_rings_left)
        mean_betas_right = mean_all_rings(betas_rings_right)

        betas.append([mean_betas_left,mean_betas_right])
        
    #     for freq in range(4):

    #         # left
    #         psf_result.extend(psf_l[freq,:])
    #         roi_result.extend(roi[sub][0])
    #         ecc_result.extend(ecc[sub][0])
    #         freq_result.extend(np.ones(ecc[sub][0].shape)*(freq+1))
    #         side_result.extend(np.zeros(ecc[sub][0].shape))
    #         subj_result.extend(np.ones(ecc[sub][0].shape)*(sub+1))

    #         # right
    #         psf_result.extend(psf_r[freq,:])
    #         roi_result.extend(roi[sub][1])
    #         ecc_result.extend(ecc[sub][1])
    #         freq_result.extend(np.ones(ecc[sub][1].shape)*(freq+1))
    #         side_result.extend(np.ones(ecc[sub][1].shape))
    #         subj_result.extend(np.ones(ecc[sub][1].shape)*(sub+1))


    # data = {'psf':psf_result, 'roi':roi_result,'ecc':ecc_result,'freq':freq_result,'side':side_result,'subj':subj_result}

    # df=pd.DataFrame(data)
    # df.dropna()
    # df.to_csv('all.csv', index=False)

    # return df

    return betas

def data(betas):

    ring_result = []
    roi_result = []
    freq_result = []
    side_result = []
    freq_type_result = []       

    for sub in range(12):
        for hem in range(2):
            for roi in range(12):
                for ring in range(24):
                    for freq in range(4):
                        temp = np.argmax(betas[sub][hem][roi][ring][freq*10:freq*10+10,:])
                        
                        




def mean_all_rings(betas):

    mean_betas = []

    for roi in range(len(betas)):
        mean_roi = []
        for ring in range(len(betas[roi])):
            ring_mean = []
            for freq in range(40):
                mean_freq = np.mean(betas[roi][ring][freq,:])
                ring_mean.append(mean_freq)
            mean_roi.append(ring_mean)
        mean_betas.append(mean_roi)
    
    return mean_betas
        

    
    

def get_indices_rois(rois):

    rois_indices = []

    for i in range(1,13):
        indices = np.where(rois==i)[0]
        rois_indices.append(indices)
        
    return rois_indices

def get_all_roi_indices_ecc(vertices, ecc_ranges):

    all_ring_per_roi = []

    for roi in range(len(vertices)):
        temp = []
        for start,end in ecc_ranges:
            temp.append(get_indices_ecc(vertices[roi],start, end))
        all_ring_per_roi.append(temp)

    return all_ring_per_roi

def get_all_values_betas(indices,betas):
    result = []

    for i in range(len(indices)):
        result.append(betas[:,indices[i]])

    return result

def get_all_values(indices,values):

    result = []

    for i in range(len(indices)):
        result.append(values[indices[i]])

    return result    

def get_indices_ecc(ecc, start_ecc, end_ecc):    
    indices_ecc = np.where((ecc > start_ecc) & (ecc < end_ecc))[0]
    return indices_ecc

def local_spatial_frequency(w_r,w_a,ecc):
    return np.linalg.norm([w_r,w_a])/ecc


        

a = (1,2)

    
    

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
betas = analysis()
data(betas)