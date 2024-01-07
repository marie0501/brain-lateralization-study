rm(list = ls())
graphics.off()
cat("\014")

library(rempsyc)
library(ggplot2)
library(lme4)
library(sjPlot)


table <- read.csv("C:\\Users\\Marie\\Documents\\thesis\\tables\\sigma_table_all_rois.csv")

table <-table[table$eccen < 6,]