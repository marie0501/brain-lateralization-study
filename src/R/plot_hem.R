rm(list = ls())
graphics.off()
cat("\014")

library(ggplot2)
library(lme4)
library(sjPlot)


result_dir <- "C:\\Users\\Marie\\Documents\\thesis\\images\\"
tables_dir <- "C:\\Users\\Marie\\Documents\\thesis\\tables\\"
y_label <- "Tamaño de pRF"
x_label <- "Excentridad"

side_labels = c('0'='izquierdo','1'='derecho')
side_values = c('navy','red')
side_name = 'Hemisferios'

rois <- c('V1', 'V2', 'V3', 'hV4', 'VO1', 'VO2', 'LO1', 'LO2', 'TO1','TO2','V3b','V3a')

set_theme(
  title.align = 'center',
  axis.title.size = 1.5,
  axis.textsize = 1.3,
  legend.size = 1.5,
  legend.title.size = 1.5,
  title.size = 2,
  base = theme_bw()
)

for (iroi in 1:12)
{
  data <- read.csv(paste0(tables_dir,rois[iroi],"_table_all_cleaned.csv"))
  data$preferred_period <-log(data$preferred_period)
  data <- data[data$preferred_period > -6,]
  data <- data[data$gml_r2 > 1,]
  data <-data[data$eccen < 6,]
  
  data$side <- as.factor(data$side)
  
  formula_full <- preferred_period~eccen*side + (1|subj) + (1|stimulus_superclass)
  formula_null<- preferred_period~eccen + (1|subj) + (1|stimulus_superclass)
  formula_additive <- preferred_period~eccen + side + (1|subj) + (1|stimulus_superclass)
  
  model_full <- lmer(formula_full,data=data)
  summary(model_full)
  
  model_null <- lmer(formula_null,data=data)
  BF_BIC_1 = exp((BIC(model_null)-BIC(model_full))/2)
  print(BF_BIC_1)
  
  model_additive <- lmer(formula_additive,data=data)
  BF_BIC_2 = exp((BIC(model_additive)-BIC(model_full))/2)
  print(BF_BIC_2)
  
  BF_BIC_3 = exp((BIC(model_null)-BIC(model_additive))/2)
  print(BF_BIC_3)  
  
  
  p <- plot_model(model_full, type='pred',terms=c('eccen','side'), show.ci = TRUE) + 
    labs(x = x_label, y = y_label, title = rois[iroi], size = 20) +   scale_color_manual(
      name = side_name,
      labels = side_labels,
      values = side_values
    ) +
    scale_fill_manual(
      name = side_name,
      labels = side_labels,
      values = side_values
    ) 
  #+
   # xlim(1, 6) + ylim(-2,2)
  
  png(paste0(result_dir,rois[iroi],'_sigma_vs_eccen_hem_limitless.png'))
  show(p)
  dev.off()
  
}
