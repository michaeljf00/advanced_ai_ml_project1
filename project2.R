rm(list=ls())
library(dplyr)
library(pls)
library(caret)
library(ranger)

stock_data = read.csv("Factors and Stock returns.csv")
indus_data = read.csv("Companies.csv")
names(stock_data)[names(stock_data) == "permno"] <- "PERMNO"
merged_data <- merge(stock_data, indus_data, by = "PERMNO", all.x = TRUE)
merged_data$SICCD <- as.character(merged_data$SICCD)
specific_siccd_data <- merged_data %>% 
  filter(substr(SICCD, 1, 2) == "35") %>%
  select(-SICCD, -NCUSIP, -TICKER, -COMNAM)
data_subset = na.omit(specific_siccd_data)
data_subset$Date = as.Date(data_subset$Date)
train_end_date = as.Date("2017-12-31")
validation_end_date = as.Date("2018-12-31")

train_data = subset(data_subset, Date <= train_end_date)
validation_data = subset(data_subset, Date > train_end_date & Date <= validation_end_date)
test_data = subset(data_subset, Date > validation_end_date)
