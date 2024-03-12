rm(list=ls())
library(caret)
library(dplyr)

setwd("/Users/z3r0/Documents/RPI/Spring2024/ML/advanced_ai_ml_project1/")
train_data <- read.csv("1000_unique_train.csv")
test_data <- read.csv("1000_unique_test.csv")
trControl <- trainControl(method = "LOOCV", classProbs = TRUE, summaryFunction = defaultSummary)
rf_model <- train(return ~ .,
                  data = train_data,
                  method = "rf",
                  trControl = trControl,
                  tuneLength = 5)
