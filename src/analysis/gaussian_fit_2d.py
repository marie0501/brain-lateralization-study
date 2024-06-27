from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def gaussian_2d(xy, a, x0, y0, sigma_x, sigma_y):
    x, y = xy
    return a * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2)))

def normalize_data(data):
    return (data - np.mean(data)) / np.std(data)

def fit_gaussian_2d(x_data, y_data, size_data):
    x_data_norm =x_data
    y_data_norm = y_data
    size_data_norm = size_data
    failure = False

    try:
        xy_data = np.vstack((x_data_norm, y_data_norm))
        bounds = ([0, -np.inf, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])
        initial_guess = [np.max(size_data_norm), 0, 0, 1, 1]
        popt, pcov = curve_fit(gaussian_2d, xy_data, size_data_norm, p0=initial_guess, maxfev=20000, bounds=bounds, method='trf')
        return popt, failure

    except:
        failure = True
        return [], failure

def generate_smooth_distribution_2d(x_data, y_data, popt):
    x_smooth = np.linspace(min(x_data), max(x_data), 100)
    y_smooth = np.linspace(min(y_data), max(y_data), 100)
    x_mesh, y_mesh = np.meshgrid(x_smooth, y_smooth)
    xy_mesh = np.vstack((x_mesh.ravel(), y_mesh.ravel()))
    z_smooth = gaussian_2d(xy_mesh, *popt).reshape(x_mesh.shape)
    return x_mesh, y_mesh, z_smooth

def plot_results_2d(x_data, y_data, size_data, x_mesh, y_mesh, z_smooth):
    plt.scatter(x_data, y_data, c=size_data, label='original', cmap='viridis')
    plt.scatter(x_mesh, y_mesh, z_smooth, alpha=0.5, label = 'predicted', cmap='viridis')
    plt.colorbar(label='Size')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.title('Ajuste de Distribución Gaussiana 2D a los Datos')
    plt.legend()
    plt.grid(True)
    plt.show()


df = pd.read_csv("/Users/mariedelvalle/Downloads/HCP_hV4.csv")
subjects = df['subj'].unique()

# fig, axes = plt.subplots(nrows=19, ncols=10, figsize=(20, 30))
# axes = axes.flatten()

# for i in subjects:
#     ax = axes[i]
#     sc = ax.scatter(df[df['subj']==i]['x'], df[df['subj']==i]['x'], c=df[df['subj']==i]['x'], cmap='viridis')
#     ax.set_title(f'Sujeto {i+1}')
#     ax.set_xlabel('Coordenada X')
#     ax.set_ylabel('Coordenada Y')

# # for j in range(subjects, len(axes)):
# #     fig.delaxes(axes[j])

# #Agregar una barra de color
# cbar = fig.colorbar(sc, ax=axes, orientation='vertical', fraction=0.02, pad=0.01)
# cbar.set_label('Size')

# plt.tight_layout()
# plt.show()

hemis = "L"

for subj in subjects:
    print(subj)
    # Supongamos que tienes los datos de x, y y size
    x_data = df[df['subj']==subj]['x']
    #x_data = df[df['hemis']==hemis]['x']
    y_data = df[df['subj']==subj]['y']
    #y_data = df[df['hemis']==hemis]['y']
    size_data = df[df['subj']==subj]['size']
    #size_data = df[df['hemis']==hemis]['size']

    # Ajustar la función gaussiana 2D a los datos
    popt, failure = fit_gaussian_2d(x_data, y_data, size_data)

    if failure:
        continue

    # Generar una distribución suavizada
    x_mesh, y_mesh, z_smooth = generate_smooth_distribution_2d(x_data, y_data, popt)

    # Graficar los resultados
    plt.scatter(x_data, y_data, c=size_data, label='original', cmap='viridis')
    plt.title(f'Subject {subj}')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.scatter(x_mesh, y_mesh, c=z_smooth, label = 'predicted', cmap='viridis')
    plt.title(f'Subject {subj}')
    plt.legend()
    plt.grid(True)
    plt.show()


# results = []

# # Supongamos que tienes los datos en un DataFrame llamado df
# # y una lista de sujetos y hemisferios
# subjects = df['subj'].unique()
# hemispheres = df['hemis'].unique()

# failures = []

# for subj in subjects:
#     for hemis in hemispheres:
#         print(f'Subject: {subj}, Hemisphere: {hemis}')
        
#         # Filtrar datos para el sujeto y hemisferio actuales
#         x_data = df[(df['subj'] == subj) & (df['hemis'] == hemis)]['x'].values
#         y_data = df[(df['subj'] == subj) & (df['hemis'] == hemis)]['y'].values
#         size_data = df[(df['subj'] == subj) & (df['hemis'] == hemis)]['size'].values

#         # Asegurarse de que hay datos para ajustar
#         if len(x_data) > 0 and len(y_data) > 0 and len(size_data) > 0:
#             # Ajustar la función gaussiana 2D a los datos
#             popt, failure = fit_gaussian_2d(x_data, y_data, size_data)

#             if failure:
#                 failures.append(f"{subj}-{hemis}")
#                 continue


#             # Generar una distribución suavizada
#             x_mesh, y_mesh, z_smooth = generate_smooth_distribution_2d(x_data, y_data, popt)

#             # Almacenar los resultados en la lista
#             for i in range(x_mesh.shape[0]):
#                 for j in range(x_mesh.shape[1]):
#                     results.append({
#                         'subj': subj,
#                         'hemis': hemis,
#                         'x': x_mesh[i, j],
#                         'y': y_mesh[i, j],
#                         'z': z_smooth[i, j]
#                     })

# # Convertir la lista de resultados en un DataFrame
# results_df = pd.DataFrame(results)

# # Mostrar el DataFrame
# print(results_df)
# print(failures)
# print(f"fallos: {len(failures)}")

# # Si quieres guardar el DataFrame a un archivo CSV
# results_df.to_csv('gaussian_results.csv', index=False)