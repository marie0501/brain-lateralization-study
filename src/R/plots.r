# Instalar y cargar el paquete lme4
install.packages("lme4")
library(lme4)

# Crear datos de ejemplo
set.seed(123)
datos <- data.frame(
  sujeto = rep(1:50, each = 4),
  tratamiento = rep(1:2, each = 100),
  respuesta = rbinom(200, 1, 0.6)
)

# Ajustar un modelo lineal mixto generalizado
modelo_glmm <- glmer(respuesta ~ tratamiento + (1|sujeto), data = datos, family = binomial)

# Resumen del modelo
summary(modelo_glmm)

# Obtener predicciones
nuevos_datos <- data.frame(tratamiento = c(1, 2))
predicciones <- predict(modelo_glmm, newdata = nuevos_datos, type = "response", re.form = NA)

# Imprimir las predicciones
print(predicciones)

# Instalar y cargar el paquete effects (si no está instalado)
install.packages("effects")
library(effects)

# Gráfico de efectos fijos
plot(allEffects(modelo_glmm))
