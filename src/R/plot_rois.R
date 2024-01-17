library(ggplot2)
library(lme4)
library(sjPlot)

rois <- c('V1', 'V2', 'V3', 'hV4', 'VO1', 'VO2', 'LO1', 'LO2', 'TO1','TO2','V3b','V3a')
colors_se = c()
y_label <- "Período Preferido"
x_label <- "Tamaño de pRF"
leyend_label <- "Áreas"


l <- c("1" = "V1", "2" = "V2",'3'='V3','4'='hV4','5'='VO1','6'='V02','7'='LO1','8'='LO2','9'='TO1','10'='TO2','11'='V3b','12'='V3a')
colors <- c("1" = "red", "2" = "darkorange",'3'='gold','4'='lightskyblue','5'='navy','6'='yellowgreen','7'='forestgreen','8'='lightpink','9'='orchid','10'='darkviolet','11'="sienna",'12'='black')
data <- read.csv("C:\\Users\\Marie\\Documents\\thesis\\tables\\hV4_table_all_cleaned.csv")

data$varea <- as.factor(data$varea)
 
data$preferred_period <-log(data$preferred_period)
data <- data[data$preferred_period > -6,]
data <- data[data$gml_r2 > 1,]
data <-data[data$eccen < 6,]

formula <- sigma~eccen*side + (1|subj) + (1|stimulus_superclass)
model <- lmer(formula,data=data)
summary(model)
plot_model(model, type='pred',terms=c('eccen','side'), show.ci = TRUE) + theme_bw() + 
  labs(x = x_label, y = y_label, title = '') + 
  scale_color_manual(
    name = side_name,
    labels = side_labels,
    values = side_values
  ) +
  scale_fill_manual(
    name = side_name,
    labels = side_labels,
    values = side_values
  )#+
  #xlim(1, 6) +  # Establecer límites en el eje x
  #ylim(-1,1)

g <- ggplot(data,aes(x=sigma,y=preferred_period, color = varea)) + 
  geom_smooth(method='lm', se=TRUE, aes(fill=varea)) + 
  theme_bw() + labs(
    x = x_label,
    y = y_label
    ) + 
  scale_color_manual(
    name = leyend_label,
    labels = rois,
    values = colors
  ) +
  scale_fill_manual(
    name = leyend_label,
    labels = rois,
    values = colors
  ) + theme(axis.text.x = element_text(size = 12),   # Ajusta el tamaño de las etiquetas del eje x
           axis.text.y = element_text(size = 12),
           axis.title.x = element_text(size = 18),
           axis.title.y = element_text(size = 18),
           legend.text = element_text(size = 14),
           legend.title = element_text(size = 16)
  )
#+
#  xlim(1, 6) +  # Establecer límites en el eje x
#  ylim(-1,1)
 
show(g)

data <- read.csv("C:\\Users\\Marie\\Documents\\thesis\\tables\\tables_rois\\hV4_table_all.csv")
data$preferred_period <-log(data$preferred_period)
data <- data[data$preferred_period > -6,]
data <- data[data$gml_r2 > 1,]
data <-data[data$eccen < 6,]

data$side <- as.factor(data$side)
side_labels = c('0'='izquierdo','1'='derecho')
side_values = c('navy','red')
side_name = 'Hemisferios'

y_label <- "Log Período Preferido"#"Tamaño de pRF"
x_label <- "Excentridad"

g <- ggplot(data,aes(x=eccen,y=preferred_period, color = side)) + 
  geom_point() + 
  theme_bw() + labs(
    x = x_label,
    y = y_label,
  ) + 
  scale_color_manual(
    name = side_name,
    labels = side_labels,
    values = side_values
  ) +
  scale_fill_manual(
    name = side_name,
    labels = side_labels,
    values = side_values
  )+
  xlim(1, 6) + ylim(-2,2)

show(g)


