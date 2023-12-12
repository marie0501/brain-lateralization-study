import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from scipy.optimize import curve_fit


df = pd.DataFrame(np.load("table_raw.csv"))

df_v1 = data[df['roi']==1 & df['frequency_class']==1]

unique_values = df_v1['eccentricity_range'].unique()

bL = []
bR = []

for sub in range(12):
    for values in unique_values:
        filtered_df = df[df['eccentricity_range'] == value]
        bL(i,:)=(median(betasL(idxL==i,1:10))+ median(betasL(idxL==i,11:20)))/2
        bR(i,:)=(median(betasR(idxR==i,1:10))+ median(betasR(idxR==i,11:20)))/2


# Crear DataFrames y ajustar el modelo mixto
data_L = pd.DataFrame({'subj': np.repeat(np.arange(1, nsub + 1), 14),
                       'bands': np.tile(np.arange(1, 15), nsub),
                       'freq': np.sqrt(Lpeak.flatten()),
                       'side': np.repeat([0, 1], nsub * 14)})

data_R = pd.DataFrame({'subj': np.repeat(np.arange(1, nsub + 1), 14),
                       'bands': np.tile(np.arange(1, 15), nsub),
                       'freq': np.sqrt(Rpeak.flatten()),
                       'side': np.repeat([0, 1], nsub * 14)})

# Ajustar modelos mixtos
model_L = smf.mixedlm("freq ~ bands * side", data_L, groups=data_L["subj"])
result_L = model_L.fit()

model_R = smf.mixedlm("freq ~ bands * side", data_R, groups=data_R["subj"])
result_R = model_R.fit()

# Imprimir resultados
print(result_L.summary())
print(result_R.summary())
