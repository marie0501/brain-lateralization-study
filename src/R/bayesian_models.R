# Instala y carga la librería

library(brms)

# Carga tus datos
# Supongamos que tienes un data.frame llamado 'datos' con las variables necesarias
data <- read.csv("C:\\Users\\Marie\\Documents\\thesis\\tables\\tables_rois\\hV4_table_all.csv")
data$preferred_period <-log(data$preferred_period)
data <- data[data$preferred_period > -6,]
data <- data[data$gml_r2 > 1,]
data <-data[data$eccen < 6,]

# Modelo Nulo
modelo_nulo <- brm(
  preferred_period ~ 1 + (1|subj) + (1|stimulus_superclass),
  data = data
)

# Modelo con Excentricidad
modelo_excentricidad <- brm(
  preferred_period ~ eccen + (1|subj) + (1|stimulus_superclass),
  data = data
)

# Modelo con Excentricidad y Hemisferio
modelo_excentricidad_hemisferio <- brm(
  preferred_period ~ eccen + hemisferio + (1|subj) + (1|stimulus_superclass),
  data = data
)

# Comparación de modelos utilizando el Factor de Bayes
comp <- compare_models(
  modelo_nulo,
  modelo_excentricidad,
  modelo_excentricidad_hemisferio
)

# Imprime la tabla de comparación
print(comp)
