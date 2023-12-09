import statsmodels.api as sm
import pandas as pd

# Crear un DataFrame de ejemplo
data = pd.DataFrame({
    'Grupo': ['A', 'A', 'B', 'B', 'C', 'C'],
    'Variable_dependiente': [10, 12, 8, 9, 15, 14],
    'Variable_independiente': [5, 6, 4, 5, 7, 6]
})

# Crear un modelo mixto con efectos aleatorios en 'Grupo'
modelo_mixto = sm.MixedLM.from_formula("Variable_dependiente ~ Variable_independiente", 
                                       groups='Grupo', 
                                       data=data)

# Ajustar el modelo
resultado = modelo_mixto.fit()

# Imprimir resumen
print(resultado.summary())
