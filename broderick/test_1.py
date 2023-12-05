import numpy as np
import pandas as pd
import nibabel as nib
from scipy import io
from sklearn.linear_model import MixedLM
from statsmodels.regression.mixed_linear_model import MixedLMResults
from scipy.io import loadmat
import os
import re

# Specify the path to your MATLAB .mat file
mat_file_path = 'F:\\ds003812-download\\derivatives\\processed'

# Expresión regular para encontrar archivos que empiezan con "beta"
patron = re.compile(r'^beta.*')

# Lista para almacenar los nombres de archivos encontrados
archivos_beta = []

# Itera a través de los archivos en el directorio
for nombre_archivo in os.listdir(directorio):
    if patron.match(nombre_archivo):
        archivos_beta.append(nombre_archivo)

# Imprime la lista de archivos que cumplen con el patrón
print("Archivos que empiezan con 'beta':")
for archivo in archivos_beta:
    print(archivo)


# Load the .mat file
mat_data = loadmat(mat_file_path)

# Access variables in the loaded data
variable1 = mat_data['variable1']
variable2 = mat_data['variable2']

# Now you can use these variables in your Python code

# Set up file paths
processed_dir = 'F:\\ds003812-download\\derivatives\\processed\\'
maps_dir = 'F:\\ds003812-download\\derivatives\\prf_solutions\\'
names = ['sub-wlsubj001', 'sub-wlsubj006', 'sub-wlsubj007', 'sub-wlsubj045',
         'sub-wlsubj046', 'sub-wlsubj062', 'sub-wlsubj064', 'sub-wlsubj081',
         'sub-wlsubj095', 'sub-wlsubj114', 'sub-wlsubj121', 'sub-wlsubj115']

# Load and process beta files
for isub, name in enumerate(names):
    print(isub)
    beta_file_path = f'{processed_dir}betas{names[isub][5:13]}.mat'
    results = io.loadmat(beta_file_path)
    betas = np.squeeze(results['modelmd'][0][0][1])
    np.save(f'{processed_dir}betas{name[5:13]}.npy', betas)

# Summarize SF beta files
pSF = []
for isub, name in enumerate(names):
    print(isub)
    beta_file_path = f'{processed_dir}betas{name[5:13]}.npy'
    betas = np.load(beta_file_path)
    temp = np.zeros((4, betas.shape[0]))
    for ifreq in range(4):
        temp[ifreq, :] = np.argmax(betas[:, 1 + (ifreq - 1) * 10:ifreq * 10 + 10], axis=1)
    pSF.append(temp)

np.save(f'{processed_dir}allpSF.npy', pSF)

# Extract retinotopy data
subjmap = []
subjecc = []
for isub, name in enumerate(names):
    print(isub)
    a = nib.load(f'{maps_dir}{name}/atlas/lh.benson14_varea.func.gii')
    b = nib.load(f'{maps_dir}{name}/atlas/rh.benson14_varea.func.gii')
    subjmap.append([a.darrays[0].data, b.darrays[0].data])

    a = nib.load(f'{maps_dir}{name}/data/lh.full-eccen.func.gii')
    b = nib.load(f'{maps_dir}{name}/data/rh.full-eccen.func.gii')
    subjecc.append([a.darrays[0].data, b.darrays[0].data])

np.save(f'{processed_dir}subjmap.npy', subjmap)
np.save(f'{processed_dir}subjecc.npy', subjecc)

# Extract data for statistical analysis
wa = [6, 8, 11, 16, 23, 32, 45, 64, 91, 128]
wr = [4, 6, 8, 11, 16, 23, 32, 45, 64, 91]
pSF = np.load(f'{processed_dir}allpSF.npy')
subjmap = np.load(f'{processed_dir}subjmap.npy')
subjecc = np.load(f'{processed_dir}subjecc.npy')

ecc = []
psf = []
roi = []
side = []
subj = []
psfcorr = []
stype = []

for isub in range(12):
    for ifreq in range(4):
        print(isub)
        idx1 = np.arange(subjmap[isub, 0].size)
        idx2 = np.arange(subjmap[isub, 1].size)
        psfL = pSF[isub, ifreq, idx1]
        psfR = pSF[isub, ifreq, idx2]

        idxgood = np.where(subjmap[isub, 0] != 0)[0]
        roi = np.concatenate([roi, subjmap[isub, 0][idxgood]])
        ecc = np.concatenate([ecc, subjecc[isub, 0][idxgood]])
        psf = np.concatenate([psf, psfL[idxgood]])
        if ifreq == 0 or ifreq == 1:
            psfcorr = np.concatenate([psfcorr, wa[psfL[idxgood]] / subjecc[isub, 0][idxgood]])
            ww = np.sqrt(np.square(wa) + np.square(wr))
            psfcorr = np.concatenate([psfcorr, ww[psfL[idxgood]] / subjecc[isub, 0][idxgood]])
        side = np.concatenate([side, np.zeros(idxgood.size)])
        subj = np.concatenate([subj, np.full(idxgood.size, isub)])
        stype = np.concatenate([stype, np.full(idxgood.size, ifreq)])

        idxgood2 = np.where(subjmap[isub, 1] != 0)[0]
        roi = np.concatenate([roi, subjmap[isub, 1][idxgood2]])
        ecc = np.concatenate([ecc, subjecc[isub, 1][idxgood2]])
        psf = np.concatenate([psf, psfR[idxgood2]])
        psfcorr = np.con
