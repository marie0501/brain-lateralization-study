import os
import numpy as np
import scipy.io


def preferred_spatial_frequency(directory):

    psf_index = []
           
    for file in os.listdir(directory):        
        beta = np.load(f"{directory}\\{file}")
        index_sub = np.empty((4,beta.shape[1]))
        for freq in range(4):           
            for voxel in range(beta.shape[1]):
                temp = beta[freq*10:freq*10+10,voxel]            
                index = (np.argmax(temp),voxel)
                index_sub[freq,voxel] = beta[index]
        psf_index.append(index_sub)

    return psf_index
    
    

    

# directory = "F:\\ds003812-download\\derivatives\\processed\\betas"
# psf=preferred_spatial_frequency(directory)
    

# # Datos de ejemplo

# data = {
#     'psf': psf,    
# }

# # Ruta del archivo .mat
# mat_file_path = 'all_psf.mat'

# # Guardar en el archivo .mat
# scipy.io.savemat(mat_file_path, data)
