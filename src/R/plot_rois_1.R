library(ggplot2)

data <- read.csv("C:\\Users\\Marie\\Documents\\thesis\\tables\\merged_table.csv")

data$preferred_period <-log(data$preferred_period)
data <- data[data$preferred_period > -6,]
data <- data[data$gml_r2 > 1,]
data <-data[data$eccen < 6,]
data$varea <- as.factor(data$varea)

colors <- c("1" = "red", "2" = "darkorange",'3'='gold','4'='lightskyblue','5'='navy','6'='yellowgreen','7'='forestgreen','8'='lightpink','9'='orchid','10'='darkviolet','11'="sienna",'12'='black')
rois <- c('V1', 'V2', 'V3', 'hV4', 'VO1', 'VO2', 'LO1', 'LO2', 'TO1','TO2','V3b','V3a')
y_label <-  "Log Período Preferido"
x_label <- "Excentricidad"
leyend_label <- "Áreas"


g <- ggplot(data,aes(x=sigma_y,y=preferred_period, color = varea)) + 
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
