

data <- read.csv("C:\\Users\\Marie\\Documents\\thesis\\tables\\all_rois_table_cleaned.csv")
data$preferred_period <-log(data$preferred_period)
data <- data[data$preferred_period > -6,]
data <- data[data$gml_r2 > 1,]
data <-data[data$eccen < 6,]
