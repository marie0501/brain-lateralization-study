from scipy.io import loadmat

# Carga el archivo .mat
mat_data = loadmat('archivo.mat')

# El resultado es un diccionario con las variables del archivo .mat
# Puedes acceder a las variables por sus nombres
variable1 = mat_data['nombre_de_variable_1']
variable2 = mat_data['nombre_de_variable_2']

# Haz algo con las variables cargadas, por ejemplo, imprímelas
print(variable1)
print(variable2)
