import nibabel as nib
import numpy as np
import os
import re

def get_prf_data(sub_directory, filename):

    hemisphere = ['lh','rh']
    subject_data = []

    for h in hemisphere:

        file_path = f"{sub_directory}\\{h}.{filename}"

        # Load the MGZ file
        img = nib.load(file_path)

        # Access the data array
        data = np.squeeze(img.get_fdata())
        print(data.shape)
        subject_data.append(data)

    return subject_data


def get_all_prf_data(directory, subdirectory, filename):

    pattern = re.compile(rf"^sub.*")

    for sub_filename in os.listdir(directory):
        if pattern.match(sub_filename):
            path = f"{directory}\\{sub_filename}\\{subdirectory}"
            data =get_prf_data(path,filename)            
            np.save(f"{directory}\\all\\{sub_filename}_{filename[:len(filename)-4]}",data)
    

def load_all_prf_data(directory,file):

    patron = re.compile(rf".*{file}.*", re.IGNORECASE)

    data = []

    for f in os.listdir(directory):
        
        if patron.match(f):
            print(f"{directory}\\{f}")
            data.append(np.load(f"{directory}\\{f}", allow_pickle=True))

    #np.save(f"{directory}\\all_{file}",data)
    return data
            
