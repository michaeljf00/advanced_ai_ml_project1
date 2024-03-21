rm(list=ls())
library(caret)
library(dplyr)

setwd("/Users/z3r0/Documents/RPI/Spring2024/ML/advanced_ai_ml_project1/")
train_data <- read.csv("1000_unique_train.csv")
test_data <- read.csv("1000_unique_test.csv")
set.seed(123)
sample_size <- round(0.1 * nrow(train_data))
sampled_train_data <- train_data[sample(nrow(train_data), sample_size), ]

trControl <- trainControl(method = "cv",
                          number = 5, 
                          allowParallel = TRUE,
                          verboseIter = TRUE)

rf_model <- train(return ~ .,
                  data = sampled_train_data,
                  method = "rf",
                  trControl = trControl,
                  tuneLength = 5)
tuneGrid <- expand.grid(.mtry = 1:6)
rf_model2 <- train(return ~ .,
                  data = train_data,
                  method = "rf",
                  trControl = trControl,
                  tuneGrid = tuneGrid)
train_data$date <- as.Date(train_data$Date)
# Random Forest
# 
# 63267 samples
# 23 predictor
# 
# No pre-processing
# Resampling: Cross-Validated (5 fold)
# Summary of sample sizes: 50613, 50615, 50612, 50614, 50614
# Resampling results across tuning parameters:
# 
#   mtry  RMSE       Rsquared   MAE
# 1     0.1235279  0.1459196  0.08605488
# 2     0.1189797  0.1712982  0.08230454
# 3     0.1167004  0.1787465  0.08034506
# 4     0.1156640  0.1823752  0.07947514
# 5     0.1150910  0.1852811  0.07900168
# 6     0.1146624  0.1881083  0.07868445
# 
# RMSE was used to select the optimal model using the smallest value.
# The final value used for the model was mtry = 6.

# Initialize a data frame to store complexity measures
complexity_measures <- data.frame(Year = integer(), Complexity = double())
# Top 10 R^2 values:
#   > print(top_10_r2)
# [1] 0.107558
# > cat("\nBottom 10 R^2 values:\n")
# 
# Bottom 10 R^2 values:
#   > print(bottom_10_r2)
# [1] 0.107558
# > cat("\nOverall R^2:", overall_r2, "\n")
# 
# Overall R^2: 0.107558 
r2_values <- rf_model$results$Rsquared

top_10_r2 <- head(sort(r2_values, decreasing = TRUE), 10)
bottom_10_r2 <- tail(sort(r2_values), 10)

overall_r2 <- rf_model$results$Rsquared[which.max(rf_model$results$Rsquared)]

cat("Top 10 R^2 values:\n")
print(top_10_r2)
cat("\nBottom 10 R^2 values:\n")
print(bottom_10_r2)
cat("\nOverall R^2:", overall_r2, "\n")
