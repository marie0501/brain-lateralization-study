import os
import nibabel as nib
import pandas as pd
import re
import numpy as np

dataset = 'NYUDataset'
files_path = '/Volumes/Elements/NYUDataset/derivatives/'

prf_path='prfanalyze-vista/'
rois_path = 'ROIs/'
properties = ['angle', 'eccen','vexpl','x','y','sigma', 'subject', 'hem', 'roi']
hem = ['lh','rh']

data = {'angle':[], 'subject':[], 'sigma':[], 'eccen':[],'x':[],'y':[],'roi':[],'hem':[], 'vexpl':[]}

pattern = r'^(rh|lh)\.(.+)\.mgz$'
subjects_pattern = r'sub-wlsubj(\d{3})'

for subj in os.listdir(f'{files_path}{prf_path}'):
    
    subj_match = re.match(subjects_pattern, subj)

    if subj_match:
        subject_number = subj_match.group(1)
        print(subject_number)
        subj_prf_path = os.path.join(f'{files_path}{prf_path}', f'{subj}/ses-nyu3t01/')
        subj_rois_path = os.path.join(f'{files_path}{rois_path}', f'{subj}/')

        for prf_file in os.listdir(subj_prf_path):

            match = re.match(pattern, prf_file)

            if match:
                prop = match.group(2)

                if prop in properties:
                    prf_data = nib.load(f'{subj_prf_path}{prf_file}')
                    prf_data = np.squeeze(prf_data.get_fdata())

                    data[prop].extend(prf_data)
                   

            

        for roi_file in os.listdir(subj_rois_path):
        
            if roi_file.endswith('.mgz'):
                roi_data = nib.load(f'{subj_rois_path}{roi_file}')
                roi_data = np.squeeze(roi_data.get_fdata())

                data['roi'].extend(roi_data)
                
                data['subject'].extend([subject_number for _ in range(len(roi_data))])
                print(roi_file[:2])
                data['hem'].extend([roi_file[:2] for _ in range(len(roi_data))])


for p in properties:
    print(f'{p}:{len(data[p])}')
df = pd.DataFrame(data)


# Opcional: guarda el DataFrame en un archivo CSV
df.to_csv(f'{dataset}.csv', index=False)       



                





    

    





