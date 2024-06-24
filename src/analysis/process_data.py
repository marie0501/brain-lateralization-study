import os
import pandas as pd

def read_mgz_file(filepath):
    import nibabel as nib
    data = nib.load(filepath).get_fdata()
    return data.flatten()

def process_subject_files(subject_path, subject_id, atlas):
    data_dict = {'subject_id': subject_id}
    for filename in os.listdir(subject_path):
        if filename.endswith('.mgz') and 'fit1' in filename:
            filepath = os.path.join(subject_path, filename)
            hemisphere, measure, _ = filename.split('.')
            data = read_mgz_file(filepath)
            data_dict[f'{hemisphere}_{measure[5:]}'] = data

    data_dict['lh_roi'] = atlas['lh']
    data_dict['rh_roi'] = atlas['rh']
            
    return data_dict

def process_atlas_file(atlas_dir, atlas_name):
    atlas = {}

    for filename in os.listdir(atlas_dir):
        if filename.endswith('.mgz') and atlas_name in filename:
            hemisphere,_,_ = filename.split('.')
            filepath = os.path.join(atlas_dir, filename)
            atlas[hemisphere] = read_mgz_file(filepath)

    return atlas


def build_dataframe(base_dir, atlas):
    all_data = []
    for subject_folder in os.listdir(base_dir):
        print(f"Processing subject: {subject_folder}")
        subject_path = os.path.join(base_dir, subject_folder)
        if os.path.isdir(subject_path):
            subject_data = process_subject_files(subject_path, subject_folder, atlas)
            subject_df = pd.DataFrame(subject_data)
            all_data.append(subject_df)
    final_df = pd.concat(all_data, ignore_index=True)
    return final_df


def main(base_dir, atlas_dir, atlas_name, output_file):
    atlas = process_atlas_file(atlas_dir, atlas_name)
    df = build_dataframe(base_dir, atlas)
    df.to_csv(output_file, index=False)
    print("DataFrame built and saved successfully.")


base_dir = '/Users/mariedelvalle/Documents/data/HCP/prfresultsmgz'
output_file = 'HCP_data.csv'
atlas_dir = '/Users/mariedelvalle/Documents/data/HCP/atlasmgz'
atlas_name = 'Wang2015'
main(base_dir, atlas_dir, atlas_name, output_file)
