from subject import Subject
from voxel import Voxel
from prf import load_all_prf_data
import numpy as np
import pandas as pd

def create_instances_from_data(beta_matrix, roi_data, eccentricity_data):
    # Assumption: beta_matrix is a 2D numpy array with rows representing voxels and columns representing frequencies

    # Assuming the order of data in roi_data and eccentricity_data corresponds to the rows of beta_matrix
    num_voxels = beta_matrix.shape[1]

    # Split beta_matrix into left and right hemispheres
    mid_col = len(roi_data[0])
    left_hemisphere_betas = beta_matrix[:, :mid_col]
    right_hemisphere_betas = beta_matrix[:, mid_col:]

    freq = np.arange(40)
    # Create instances for left hemisphere voxels
    left_hemisphere_voxels = [
        Voxel(roi_data[0][i], eccentricity_data[0][i], 0, dict(zip(freq, left_hemisphere_betas[:40,i])))
        for i in range(mid_col)
    ]

    # Create instances for right hemisphere voxels
    right_hemisphere_voxels = [
        Voxel(roi_data[1][i], eccentricity_data[1][i], 1, dict(zip(freq, right_hemisphere_betas[:40,i])))
        for i in range(num_voxels-mid_col)
    ]

    return np.concatenate((left_hemisphere_voxels,right_hemisphere_voxels))

def create_subjects_from_data(beta_matrix, roi_data, eccentricity_data):

    subject_ids = len(beta_matrix)
    subjects = [
        Subject(i, create_instances_from_data(beta_matrix[i],roi_data[i],eccentricity_data[i]))
        for i in range(subject_ids)
    ]
    return subjects

def spatial_frequency(freq, ecc, max_index):

    freq_1_2 = np.sqrt(np.array([6, 8, 11, 16, 23, 32, 45, 64, 91, 128])**2)
    freq_3_4 = np.sqrt((np.array([4, 6, 8, 11, 16, 23, 32, 45, 64, 91])**2)*2)

    if ecc > 10**(-8):
        if freq < 2:
            return freq_1_2[max_index]/ecc
        else:
            return freq_3_4[max_index]/ecc  
    else:
        return 0
    
def generate_dataframe(subjects, eccentricity_ranges):
    # Inicializar listas para almacenar los datos
    subject_list = []
    roi_list = []
    eccentricity_range_list = []
    frequency_class_list = []
    max_average_beta_list = []
    side_list = []

    # Definir las clases de frecuencia (suponiendo que tienes 4 clases de 10 frecuencias cada una)
    frequency_classes = [list(range(i, i+10)) for i in range(0, 40, 10)]

    # Iterar sobre los sujetos
    for subject in subjects:
        print(f"subject: {subject}")
        sides = subject.group_voxels_by_criteria(side = [0,1])
        for s,side in enumerate(sides):            
            # Iterar sobre los ROIs del sujeto
            for roi_name in range(12):
                roi = [voxel for voxel in side if voxel.roi == roi_name]
                # Iterar sobre los rangos de excentricidad
                for eccentricity_range in eccentricity_ranges:
                    # Filtrar voxels por ROI y rango de excentricidad
                    filtered_voxels = [voxel for voxel in roi if eccentricity_range[0] <= voxel.eccentricity <= eccentricity_range[1]]

                    # Calcular el máximo del promedio de betas por clase de frecuencia
                    if filtered_voxels:
                        max_average_betas = {}
                        for i,freq_class in enumerate(frequency_classes):
                            class_betas = []
                            for freq in freq_class:
                                temp = [voxel.beta_values[freq] for voxel in filtered_voxels]
                                class_betas.append(np.mean(temp))
                            max_index =  np.argmax(class_betas)
                            max_average_betas = spatial_frequency(i,np.mean([eccentricity_range[0],eccentricity_range[1]]),max_index)


                            # Almacenar los resultados en listas
                            subject_list.append(subject.subject_id)
                            roi_list.append(roi_name)
                            eccentricity_range_list.append(eccentricity_range)
                            frequency_class_list.append(i)
                            max_average_beta_list.append(max_average_betas)
                            side_list.append(s)

    # Crear un DataFrame a partir de las listas
    df = pd.DataFrame({
        'Subject': subject_list,
        'ROI': roi_list,
        'Eccentricity Range': eccentricity_range_list,
        'Frequency Class': frequency_class_list,
        'Max Average Beta': max_average_beta_list,
        'Side': side_list
    })

    df.to_csv("dataframe.csv", index=False)

    return df


psf= load_all_prf_data("F:\\ds003812-download\\derivatives\\processed\\betas","betas")
ecc=load_all_prf_data("F:\\ds003812-download\\derivatives\\prf_solutions\\all", "eccen")
roi=load_all_prf_data("F:\\ds003812-download\\derivatives\\prf_solutions\\all", "benson14_varea")



# Creating instances of Subjects
subjects = create_subjects_from_data(psf,roi,ecc)

arr=np.linspace(1.5,25.5,25)
eccentricity_ranges=list(zip(arr, arr[1:]))

# DataFrame resultante
result_df = generate_dataframe(subjects, eccentricity_ranges)

# Imprimir el DataFrame
print(result_df)




