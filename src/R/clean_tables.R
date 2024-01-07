
result_dir <- 'C:\\Users\\Marie\\Documents\\thesis\\tables\\tables_rois_cleaned\\'
rois <- c('V1', 'V2', 'V3', 'hV4', 'VO1', 'VO2', 'LO1', 'LO2', 'TO1','TO2','V3b','V3a')

for(iroi in 1:12){
  
  print(iroi)
  data <- read.csv(paste0('C:\\Users\\Marie\\Documents\\thesis\\tables\\tables_rois\\', rois[iroi],'_table_all.csv'))
  
  data$preferred_period <-log(data$preferred_period)
  data <- data[data$preferred_period > -6,]
  data <- data[data$gml_r2 > 1,]
  data <-data[data$eccen < 6,]
  
  
  save(data,file = paste0(result_dir,'table_all_cleaned.csv'))
}