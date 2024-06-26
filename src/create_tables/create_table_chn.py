import os
import nibabel as nib
import pandas as pd
import re
import numpy as np

dataset = 'CHNDataset'
files_path = '/Volumes/Elements/ds004698-download/derivatives/prf-estimation/'

prf_path='prfs/'
rois_path = 'rois/'
properties = ['angle', 'eccen','vexpl','x0','y0','sigma', 'subject', 'hem']

size = 0

data = {'angle':[], 'subject':[], 'sigma':[], 'eccen':[],'x0':[],'y0':[], 'hem':[], 'vexpl':[]}

pattern = r'sub-[0-9][1-9]_ses-all_task-all_hemi-([LR])_space-fsnative_(.+)\.mgz'
file_pattern = r'sub-[0-9][1-9]_ses-all_task-all_hemi-([LR])_space-fsnative_prf'


for subj in os.listdir(f'{files_path}'):
    print(subj)
    if subj.startswith('sub'):

        subj_prf_path = os.path.join(f'{files_path}{subj}/', f'{prf_path}')
       

        for prf_file in os.listdir(subj_prf_path):

            match = re.match(file_pattern, prf_file)

            if match:
                hem = match.group(1)

                for prop_file in os.listdir(f'{subj_prf_path}{prf_file}/'):

                    prop_math = re.match(pattern, prop_file)

                    if prop_math:
                        prop = prop_math.group(2)

                        prf_data = nib.load(f'{subj_prf_path}{prf_file}/{prop_file}')
                        prf_data = np.squeeze(prf_data.get_fdata())

                        data[prop].extend(prf_data)
                        size = len(prf_data)

                data['subject'].extend([subj[len(subj)-2:] for _ in range(size)])
                data['hem'].extend(["lh" if hem == "L" else "rh" for _ in range(size)])
         
        # for roi_file in os.listdir(subj_rois_path):
        
        #     if roi_file.endswith('.mgz'):
        #         roi_data = nib.load(f'{subj_rois_path}{roi_file}')
        #         roi_data = np.squeeze(roi_data.get_fdata())

        #         data['roi'].extend(roi_data)
                
        #         data['subject'].extend([subject_number for _ in range(len(roi_data))])
        #         print(roi_file[:2])
        #         data['hem'].extend([roi_file[:2] for _ in range(len(roi_data))])


for p in properties:
    print(f'{p}:{len(data[p])}')
df = pd.DataFrame(data)

df.dropna()
# Opcional: guarda el DataFrame en un archivo CSV
df.to_csv(f'{dataset}.csv', index=False)       



                





    

    





