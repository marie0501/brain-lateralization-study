# Import some standard/utility libraries:
import os, sys, six # six provides python 2/3 compatibility

# Import our numerical/scientific libraries, scipy and numpy:
import numpy as np
import scipy as sp

# The neuropythy library is a swiss-army-knife for handling MRI data, especially
# anatomical/structural data such as that produced by FreeSurfer or the HCP.
# https://github.com/noahbenson/neuropythy
import neuropythy as ny

# We also import ipyvolume, the 3D graphics library used by neurropythy, for 3D
# surface rendering (optional).
import ipyvolume as ipv

import nibabel as nib

from rgb_scale import create_rgb_scale_from_array

import matplotlib.pyplot as plt

# If you aren't running the tutorial in the docker-image, make sure to set this
# to a FreeSurfer subject directory that you have access to locally.
sub = ny.freesurfer_subject('C:\\Users\\Marie\\Documents\\freesurfer_subjects\\S1201')
print(len(sub.LH.indices))
ecc = nib.load("C:\\Users\\Marie\\Documents\\freesurfer_subjects\\S1201\\prfs\\lh.inferred_eccen.mgz")
vertices = ecc.get_fdata()

print(f"shape vertices: {vertices.shape}")
vertices_flatten = vertices.flatten()
print(f"vertices flatten shape: {vertices_flatten.shape}")
colores_normalizados = (vertices_flatten - np.min(vertices_flatten)) / (np.max(vertices_flatten) - np.min(vertices_flatten))
print(f"shape colores_normalizados: {colores_normalizados.shape}")
# Construye una matriz de colores RGBA para cada vértice
# En este ejemplo, se utiliza una escala de colores de matplotlib
colores_rgba = plt.cm.viridis(colores_normalizados)

# # Imprime los primeros 5 valores de colores RGBA como ejemplo
print(colores_rgba.shape)

# Número de elementos en la matriz
# num_elementos = 142997

# # Generar tonalidades de rojo utilizando linspace
# tonalidades_rojas = np.linspace(0, 1, num_elementos)

# # Crear la matriz RGBA con tonalidades de rojo
# colores_rgba = np.zeros((num_elementos, 4))
# colores_rgba[:, 0] = tonalidades_rojas  # Componente roja
# colores_rgba[:, 3] = 1.0  # Componente alfa (opacidad)
print(colores_rgba.shape)

ny.cortex_plot(sub.lh, surface='inflated', color=colores_rgba)