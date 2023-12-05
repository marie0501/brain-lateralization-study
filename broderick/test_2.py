import numpy as np
import pandas as pd
import nibabel as nib
from scipy import io
#from statsmodels.regression.mixed_linear_model import MixedLMResults, MixedLM
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
for nombre_archivo in os.listdir(mat_file_path):
    if patron.match(nombre_archivo):
        archivos_beta.append(nombre_archivo)

# Load the .mat file
mat_data = loadmat(f"{mat_file_path}\\{archivos_beta[0]}")


import h5py


# Open the file using h5py
with h5py.File(f"{mat_file_path}\\sub-wlsubj001_ses-04_task-sfprescaled_results.mat", 'r') as file:
    # Access the variables in the HDF5 file
    betas = file['results']['modelmd'][:]
    print(file['results']['modelmd'][:])
    print(file['results']['R2'].shape)
    print(np.sum(file['results']['R2'][:] > 0.90))
    # Add more variables as needed

file = h5py.File(f"{mat_file_path}\\sub-wlsubj001_ses-04_task-sfprescaled_results.mat", 'r')
betas = file['results']['modelmd'][:]
values = file[betas.flat[1]]
values_flat = np.squeeze(values)
print(values_flat.shape)
