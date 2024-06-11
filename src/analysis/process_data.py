import os
import pandas as pd

def read_mgz_file(filepath):
    import nibabel as nib
    data = nib.load(filepath).get_fdata()
    return data.flatten()

def process_subject_files(subject_path, subject_id):
    data_dict = {'subject_id': subject_id}
    for filename in os.listdir(subject_path):
        if filename.endswith('.mgz'):
            filepath = os.path.join(subject_path, filename)
            hemisphere, _, measure = filename.split('.')
            data = read_mgz_file(filepath)
            data_dict[f'{hemisphere}_{measure}'] = data
    return data_dict


def build_dataframe(base_dir):
    all_data = []
    for subject_folder in os.listdir(base_dir):
        subject_path = os.path.join(base_dir, subject_folder)
        if os.path.isdir(subject_path):
            subject_data = process_subject_files(subject_path, subject_folder)
            subject_df = pd.DataFrame(subject_data)
            all_data.append(subject_df)
    final_df = pd.concat(all_data, ignore_index=True)
    return final_df


def main(base_dir, output_file):
    df = build_dataframe(base_dir)
    df.to_parquet(output_file, compression='snappy')
    print("DataFrame built and saved successfully.")

# Ejecutar el proceso
base_dir = 'pfr_result'
output_file = 'datos_voxeles.parquet'
main(base_dir, output_file)
