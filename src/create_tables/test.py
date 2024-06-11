import os

# Ruta al archivo ejecutable Unix
archivo_ejecutable_unix = '/Volumes/Elements/ds004489-download/derivatives/sub-114/Retinotopy/sub-114_task-Retinotopy_space-fsnative_hemi-L_aPRF_ang'

# Verificar que el archivo existe
if os.path.exists(archivo_ejecutable_unix):
    # Nueva ruta con la extensión .mgz
    nueva_ruta = 'sub-114_task-Retinotopy_space-fsnative_hemi-L_aPRF_ang' + '.mgz'
    
    # Renombrar el archivo
    os.rename(archivo_ejecutable_unix, nueva_ruta)
    
    print(f"Archivo renombrado a: {nueva_ruta}")
else:
    print("El archivo especificado no existe.")
