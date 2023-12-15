import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df_1 = pd.read_csv("C:\\Users\\Marie\\Documents\\thesis\\broderick\\table_size_prf.csv")
df_2 = pd.read_csv("C:\\Users\\Marie\\Documents\\thesis\\broderick\\table_size_prf_benson14.csv")
df_3 = pd.read_csv("C:\\Users\\Marie\\Documents\\thesis\\broderick\\table_size_prf_data_full.csv")
df_4 = pd.read_csv("C:\\Users\\Marie\\Documents\\thesis\\broderick\\table_size_prf_data_full_inferred_varea.csv")

dfs = [df_1,df_2,df_3,df_4]

rois = ['V1','V2','V3','hV4','VO1','VO2','V3a','V3b','LO1','LO2','TO1','TO2']

sub = 2

colors = ['blue','red','green','yellow']

for index_roi in range(1,len(rois)+1):
    for index_df in range(len(dfs)):

        current_df_0 = dfs[index_df][(dfs[index_df]['subj']==sub) & (dfs[index_df]['roi']==index_roi) & (dfs[index_df]['side']==0)]
        current_df_1 = dfs[index_df][(dfs[index_df]['subj']==sub) & (dfs[index_df]['roi']==index_roi) & (dfs[index_df]['side']==1)]

        # Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
        X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(current_df_0[['size']], current_df_0['ecc'], test_size=0.2, random_state=42)
        X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(current_df_1[['size']], current_df_1['ecc'], test_size=0.2, random_state=42)

        # Inicializar el modelo de regresión lineal
        model_0 = LinearRegression()
        model_1 = LinearRegression()

        # Entrenar el modelo con el conjunto de entrenamiento
        model_0.fit(X_train_0, y_train_0)
        model_1.fit(X_train_1, y_train_1)

        # Realizar predicciones en el conjunto de prueba
        y_pred_0 = model_0.predict(X_test_0)
        y_pred_1 = model_1.predict(X_test_1)


        plt.plot(X_test_0, y_pred_0, color=colors[index_df], linewidth=3, label=f'left {index_df + 1}')
        plt.plot(X_test_1, y_pred_1, color=colors[index_df], linewidth=3, linestyle='--', label=f"rigth {index_df + 1}")

    plt.xlabel('size')
    plt.ylabel('ecc')
    plt.title(f"{rois[index_roi-1]}")
    plt.show()