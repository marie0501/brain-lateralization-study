library(lme4)

path <- "/Users/mariedelvalle/Documents/projects/brain_lateralization_study/tables/HCP_prfresults.csv"

table <- read.csv(path)

model <- lmer(rfsize~, data = table)

# Mostrar el resumen del modelo
summary(modelo)


