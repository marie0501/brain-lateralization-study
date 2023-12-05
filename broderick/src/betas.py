import h5py
import os
import re
from scipy.io import loadmat
import numpy as np

def get_betas(directory, filename): 
    
    # Open the file using h5py
    file = h5py.File(f"{directory}\\{filename}", 'r')

    betas = file['results']['modelmd'][:]
    values = file[betas.flat[1]]
    values_flat = np.squeeze(values)

    return values_flat

    

def get_all_betas(directory, filename_prefix):

    # Regular expression pattern to match filenames starting with "beta"
    pattern = re.compile(rf"^{filename_prefix}.*")

    # List to store the found file names
    beta_files = []

    # Iterate through the files in the directory
    for filename in os.listdir(directory):
        if pattern.match(filename):
            beta_files = get_betas(directory,filename)
            # Save betas
            np.save(f"{directory}\\betas\\betas_{filename}", beta_files)

    


    
        
    