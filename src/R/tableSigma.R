rm(list = ls())
graphics.off()
cat("\014")

library(rempsyc)
library(ggplot2)
library(lme4)
library(sjPlot)

result_dir <- "C:\\Users\\Marie\\Documents\\thesis\\images\\"
tables_dir <- "C:\\Users\\Marie\\Documents\\thesis\\tables\\"
rois <- c('V1', 'V2', 'V3', 'hV4', 'VO1', 'VO2', 'LO1', 'LO2', 'TO1','TO2','V3b','V3a')

tab <- NULL

table <- read.csv("C:\\Users\\Marie\\Documents\\thesis\\tables\\sigma_table_all_rois.csv")

table <-table[table$eccen < 6,]

for (iroi in 1:12)
{
  print(iroi) 
  data <- table[table$area == iroi,]
  #data$preferred_period <-log(data$preferred_period)
  
  
  model0 <- lmer(sigma ~ 1  + (1 | subj), data = data)
  model1 <- lmer(sigma ~ eccen  + (1 | subj), data = data)
  model2 <- lmer(sigma ~ eccen + side + (1 | subj), data = data)
  model3 <- lmer(sigma ~ eccen * side + (1 | subj), data = data)
  
  BF_eccen = exp((BIC(model0)-BIC(model1))/2)
  print(BF_eccen)
  BF_side = exp((BIC(model1)-BIC(model2))/2)
  print(BF_side)
  BF_Int = exp((BIC(model2)-BIC(model3))/2)
  print(BF_Int)
  bf10 <- data.frame( BF10=c('', format(BF_eccen,scientific = TRUE, digits = 3), format(BF_side,scientific = TRUE, digits = 3), format(BF_Int,scientific = TRUE, digits = 3)))
  
  summodel<-summary(model3)
  coeff <- summodel$coefficients
  colnames(coeff)[1] <- 'Coef'
  colnames(coeff)[2] <- 'SE'
  colnames(coeff)[3] <- 't-val'
  coeff <- cbind(coeff,bf10)
  rn<-data.frame(áreas=rois[iroi])
  temp <- cbind(rn,coeff['eccen',],coeff['side',],coeff['eccen:side',])
  cn <- colnames(temp)
  cn[2:5] <- paste0('Excentricidad.',cn[2:5])
  cn[6:9] <- paste0('Hemisferio.',cn[6:9])
  cn[10:13] <- paste0('Excentricidad:Hemisferio.',cn[10:13])
  colnames(temp)<-cn
  tab <- rbind(tab,temp)
  
}
nt <- nice_table(tab,separate.header = TRUE)

ft <- flextable::autofit(nt, add_w = 0, add_h = 0)
w<-flextable::width(ft,width=0.3)
flextable::save_as_image(w, path = paste0(result_dir,'table_sigma',".png"))

#dt <- flextable::dim_pretty(nt)


#save(tab,file = paste0(result_dir,'tab_sigma_all_rois.csv'))
#
#df <- data.frame(x = factor(tab$áreas,levels = tab$áreas), y = c(tab$Excentricidad.Coef,tab$Lado.Coef,tab$`Excentricidad:Lado.Coef`),group=rep(c('Excentricidad','Lado','Excentricidad:Lado'),each=12))
#g <- ggplot(df, aes(x=x, y=y, group = group, color=group)) +  
#  geom_line(linewidth = 1.5) + 
#  geom_point(size=5, shape = 16) +
#  labs(
#    x = 'Áreas visuales',
#    y= 'Tamaño de efecto',
#    
#  ) + theme_bw()+
#  scale_color_manual(
#    name = '',
#    labels = c('Excentricidad','Lado','Excentricidad:Lado'),
#    values = c('red','navy','forestgreen')
#  )+theme(axis.text.x = element_text(size = 12),   # Ajusta el tamaño de las etiquetas del eje x
#          axis.text.y = element_text(size = 12),
#          axis.title.x = element_text(size = 16),
#          axis.title.y = element_text(size = 16),
#          legend.text = element_text(size = 14)
#  )
#
#
#png(paste0(result_dir,'effect_size_rois.png'),family = "ArialMT", units = "cm",
#    width = 17.35, height = 15, pointsize = 18, res = 300)
#show(g)
#dev.off()
#