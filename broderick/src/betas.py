import h5py
import os
import re
from scipy.io import loadmat
import numpy as np
import scipy.io
from scipy.ndimage import gaussian_filter

def get_betas(directory, filename): 
    
    # Open the file using h5py
    file = h5py.File(f"{directory}\\{filename}", 'r')

    betas = file['results']['modelmd'][:]
    values = file[betas.flat[1]]
    values_flat = np.squeeze(values)

    return values_flat
    

def get_all_betas(directory, filename_prefix, smooth=False):

    # Regular expression pattern to match filenames starting with "beta"
    pattern = re.compile(rf"^{filename_prefix}.*")

    # List to store the found file names
    beta_files = []

    # Iterate through the files in the directory
    for filename in os.listdir(directory):
        if pattern.match(filename):
            beta_files = get_betas(directory,filename)
            # Save betas

            if smooth:
                np.save(f"{directory}\\betas\\smoothed_betas_{filename}", gaussian_filter(beta_files, sigma=3.0))
            else:
                np.save(f"{directory}\\betas\\betas_{filename}", beta_files)



# directory = 'F:\\ds003812-download\\derivatives\\processed'
# filename = "sub-wlsub"

# get_all_betas(directory=directory,filename_prefix=filename, smooth=True)







    
        
    