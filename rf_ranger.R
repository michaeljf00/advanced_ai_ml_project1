rm(list=ls())
library(dplyr)
library(pls)
library(caret)
library(ranger)

stock_data = read.csv("Factors and Stock returns.csv")

## Sub-setting the Data Frame to be between 2009 and 2019 
start_date = as.Date("2009-01-31")
end_date = as.Date("2019-12-31")
data_subset = stock_data[stock_data$Date >=start_date  & stock_data$Date <=end_date, ]

## Dropping all NA values 
data_subset = na.omit(data_subset)

## Winsorizing the data leaving out Date and permno columns 
numeric_cols = sapply(data_subset[, !(names(data_subset) %in% c("Date", "permno"))], is.numeric)

# Loop through numeric columns and winsorize
for (col in names(data_subset)[numeric_cols]) {
  if (is.numeric(data_subset[[col]])) {
    q1 = quantile(data_subset[[col]], 0.01, na.rm = TRUE)
    q99 = quantile(data_subset[[col]], 0.99, na.rm = TRUE)
    
    data_subset[[col]][data_subset[[col]] < q1] = q1
    data_subset[[col]][data_subset[[col]] > q99] = q99
  }
}

set.seed(123)

## Time Based Train-Validation-Test split 
data_subset$Date = as.Date(data_subset$Date)
train_end_date = as.Date("2017-12-31")
validation_end_date = as.Date("2018-12-31")

train_data = subset(data_subset, Date <= train_end_date)
write.csv(train_data, file = "whole_data_train.csv", row.names = FALSE)

validation_data = subset(data_subset, Date > train_end_date & Date <= validation_end_date)
write.csv(validation_data, file = "whole_data_val.csv", row.names = FALSE)

test_data = subset(data_subset, Date > validation_end_date)
write.csv(test_data, file = "whole_data_test.csv", row.names = FALSE)
## -----------------------------------------------------------------------------------------------
## TOP AND BOTTOM 1000 STOCKS AND TRAIN-VAL-TEST SPLIT 

data_ch_top <- data_subset %>% 
  arrange(desc(mvel1)) %>%
  group_by(Date) %>%
  slice_head(n = 1000) %>%
  ungroup() %>%
  select(-Date)

data_ch_bot <- data_subset %>% 
  arrange(desc(mvel1)) %>%
  group_by(Date) %>%
  slice_tail(n = 1000) %>%
  ungroup() %>%
  select(-Date)


set.seed(123)

top_bottom_split <- function(dataset, train_ratio=0.7, validation_ratio=0.2, test_ratio=0.1) {
  dataset_name <- deparse(substitute(dataset))
  
  # Shuffle dataset
  # dataset <- dataset[sample(nrow(dataset)), ]
  
  # Compute sizes of train, validation, and test sets
  n <- nrow(dataset)
  train_size <- floor(train_ratio * n)
  validation_size <- floor(validation_ratio * n)
  test_size <- n - train_size - validation_size
  
  # Split dataset
  train_data <- dataset[1:train_size, ]
  write.csv(train_data, file = paste0(dataset_name, "_train.csv"), row.names = FALSE)
  #X_train <- as.matrix(train_data[, -which(names(train_data) %in% c("return"))])
  #X_train <- apply(X_train, 2, as.numeric)
  #y_train <- train_data$return
  
  validation_data <- dataset[(train_size + 1):(train_size + validation_size), ]
  write.csv(validation_data, file = paste0(dataset_name, "_validation.csv"), row.names = FALSE)
  #X_val <- as.matrix(validation_data[, -which(names(validation_data) %in% c("return"))])
  #X_val <- apply(X_val, 2, as.numeric)
  #y_val <- validation_data$return
  
  test_data <- dataset[(train_size + validation_size + 1):n, ]
  write.csv(test_data, file = paste0(dataset_name, "_test.csv"), row.names = FALSE)
  #X_test <- as.matrix(test_data[, -which(names(test_data) %in% c("return"))])
  #X_test <- apply(X_test, 2, as.numeric)
  #y_test <- test_data$return
  
  # Return split datasets
  return(list(train_data=train_data, test_data=test_data, validation_data=validation_data))
}

top_1000_data <- top_bottom_split(data_ch_top)
top_1000_data$train_data
top_1000_data$validation_data
top_1000_data$test_data


trControl <- trainControl(method = "cv",
                          number = 5, 
                          allowParallel = TRUE,
                          verboseIter = TRUE)
tuneGrid <- expand.grid(
  .mtry = 2:6,
  .splitrule = c("variance"),
  .min.node.size = 1
)
rf_model <- train(return ~ .,
                  data = top_1000_data$train_data,
                  method = "ranger",
                  trControl = trControl,
                  tuneGrid = tuneGrid)
# Predicting on the training data for in-sample R-squared
in_sample_predictions <- predict(rf_model, newdata = top_1000_data$train_data)

# Actual values from the training data
in_sample_actuals <- top_1000_data$train_data$return

# Calculating in-sample R-squared
in_sample_ss_res <- sum((in_sample_actuals - in_sample_predictions)^2)
in_sample_ss_tot <- sum((in_sample_actuals - mean(in_sample_actuals))^2)
in_sample_r_squared <- 1 - (in_sample_ss_res / in_sample_ss_tot)

# Displaying the in-sample R-squared value
in_sample_r_squared/ nrow(top_1000_data$train_data)

# Predicting on the test data
predicted_values <- predict(rf_model, newdata = top_1000_data$test_data)

# Actual values from the test data
actual_values <- top_1000_data$test_data$return

# Calculating R-squared
ss_res <- sum((actual_values - predicted_values)^2)
ss_tot <- sum((actual_values - mean(actual_values))^2)
r_squared <- 1 - (ss_res / ss_tot)

# Displaying the R-squared value
r_squared

bot_1000_data <- top_bottom_split(data_ch_bot)
bot_1000_data$train_data
bot_1000_data$validation_data
bot_1000_data$test_data

trControl <- trainControl(method = "cv",
                          number = 3, 
                          allowParallel = TRUE,
                          verboseIter = TRUE)
tuneGrid <- expand.grid(
  .mtry = 2:6,
  .splitrule = c("variance"),
  .min.node.size = 1
)
rf_model2 <- train(return ~ .,
                   data = bot_1000_data$train_data,
                   method = "ranger",
                   trControl = trControl,
                   tuneGrid = tuneGrid)
# Actual values from the training data
in_sample_predictions_bot <- predict(rf_model2, newdata = bot_1000_data$train_data)
in_sample_actuals_bot <- bot_1000_data$train_data$return

# Calculating in-sample R-squared
in_sample_ss_res_bot <- sum((in_sample_actuals_bot - in_sample_predictions_bot)^2)
in_sample_ss_tot_bot <- sum((in_sample_actuals_bot - mean(in_sample_actuals_bot))^2)
in_sample_r_squared_bot <- 1 - (in_sample_ss_res_bot / in_sample_ss_tot_bot)

# Displaying the in-sample R-squared value
in_sample_r_squared_bot

predicted_values_bot <- predict(rf_model2, newdata = bot_1000_data$test_data)

# Actual values from the test data
actual_values_bot <-  bot_1000_data$test_data$return

# Calculating R-squared
ss_res_bot <- sum((actual_values_bot - predicted_values_bot)^2)
ss_tot_bot <- sum((actual_values_bot - mean(actual_values_bot))^2)
r_squared_bot <- 1 - (ss_res_bot / ss_tot_bot)

# Displaying the R-squared value
r_squared_bot

rf_model3 <- train(return ~ .,
                   data = train_data,
                   method = "ranger",
                   trControl = trControl,
                   tuneGrid = tuneGrid)
predicted_values_whole <- predict(rf_model3, newdata = test_data)

# Actual values from the test data
actual_values_whole <-test_data$return

# Calculating R-squared
ss_res_whole <- sum((actual_values_whole - predicted_values_whole)^2)
ss_tot_whole <- sum((actual_values_whole - mean(actual_values_whole))^2)
r_squared_whole <- 1 - (ss_res_whole / ss_tot_whole)

# Displaying the R-squared value
r_squared_whole

library(lubridate)
time_period_fit <- function(dataset, start_date, end_date, train_ratio=0.9) {
  
  for (curr_year in year(start_date):year(end_date)) {
    
    # Get subset of dataset with data points from current year
    subset = data_subset[year(data_subset$Date) == curr_year, ]
    
    n = nrow(subset)
    train_size <- floor(train_ratio * n)
    
    # Fit and test model on this data
    train_data <- subset[1:train_size, ]
    test_data <- subset[(train_size + 1):n, ]
    
    # Fit OLS/Random Forest model here
    rf_model_split <- train(return ~ .,
                       data = train_data,
                       method = "ranger",
                       trControl = trControl,
                       tuneGrid = tuneGrid)
    predicted_values <- predict(rf_model_split, newdata = test_data)
    
    # Actual values from the test data
    actual_values <- test_data$return
    
    # Calculating R-squared
    ss_res <- sum((actual_values - predicted_values)^2)
    ss_tot <- sum((actual_values - mean(actual_values))^2)
    r_squared <- 1 - (ss_res / ss_tot)
    
    # Displaying the R-squared value
    
    ### Code Below is for manual R_squared calculations
    ### Total sum of squares
    ##TSS <- sum((y_test - mean(y_test))^2)
    ##
    ### Residual sum of Squares
    ##RSS <- sum((y_test - predictions)^2)
    ##
    ### Compute R-squared
    ##R_squared <- 1 - (RSS / TSS)
    ##
    
    # Record R_squared or current year
    print(curr_year)
    print(r_squared)
    
  }
  
}

# Run time_period_fit here
time_period_fit(data_subset, start_date, end_date)

saveRDS(rf_model, file = "rf_model.rds")
saveRDS(rf_model2, file = "rf_model2.rds")
saveRDS(rf_model3, file = "rf_model3.rds")
#==============================================================================================
# update
# =============================================================================================
train_and_evaluate <- function(train_data, validation_data, test_data) {
  trControl <- trainControl(method = "cv",
                            number = 5, 
                            allowParallel = TRUE,
                            verboseIter = TRUE)
  tuneGrid <- expand.grid(
    mtry = 2:6,
    splitrule = c("variance"),
    min.node.size = c(1,10,100)
  )
  
  model <- train(return ~ ., data = train_data, method = "ranger",  importance = 'impurity', trControl = trControl, tuneGrid = tuneGrid)
  print(paste("Best mtry:", model$bestTune$mtry))
  print(paste("Best splitrule:", model$bestTune$splitrule))
  var_importance <- varImp(model)
  print(var_importance)
  validation_predictions <- predict(model, newdata = validation_data)
  validation_actuals <- validation_data$return
  validation_ss_res <- sum((validation_actuals - validation_predictions)^2)
  validation_ss_tot <- sum((validation_actuals - mean(validation_actuals))^2)
  validation_r_squared <- 1 - (validation_ss_res / validation_ss_tot)
  combined_data <- rbind(train_data, validation_data)
  final_model <- train(return ~ ., data = combined_data, method = "ranger", trControl = trControl, tuneGrid = data.frame(mtry = model$bestTune$mtry, splitrule = model$bestTune$splitrule, min.node.size = model$bestTune$min.node.size))
  test_predictions <- predict(final_model, newdata = test_data)
  test_actuals <- test_data$return
  test_ss_res <- sum((test_actuals - test_predictions)^2)
  test_ss_tot <- sum((test_actuals - test_actuals)^2)
  test_r_squared <- 1 - (test_ss_res / test_ss_tot)
  list(
    model = final_model,
    validation_r_squared = validation_r_squared,
    test_r_squared = test_r_squared,
    variable_importance = var_importance
  )
}
results_top <- train_and_evaluate(top_1000_data$train_data, top_1000_data$validation_data, top_1000_data$test_data)
importance_data <- results_top$variable_importance$importance
top_features <- rownames(head(importance_data, 10))
results_bot <- train_and_evaluate(bot_1000_data$train_data, bot_1000_data$validation_data, bot_1000_data$test_data)
results_all <- train_and_evaluate(train_data, validation_data, test_data)
cat("Top 1000 Stocks Validation R-squared:", results_top$validation_r_squared, "\n")
cat("Top 1000 Stocks Test R-squared:", results_top$test_r_squared, "\n")
cat("Bottom 1000 Stocks Validation R-squared:", results_bot$validation_r_squared, "\n")
cat("Bottom 1000 Stocks Test R-squared:", results_bot$test_r_squared, "\n")
cat("All 1000 Stocks Validation R-squared:", results_all$validation_r_squared, "\n")
cat("All 1000 Stocks Test R-squared:", results_all$test_r_squared, "\n")
# Top 1000 Stocks Validation R-squared: -0.01339317 
# Top 1000 Stocks Test R-squared: 0.01454941 
# Bottom 1000 Stocks Validation R-squared: -0.004824673 
# Bottom 1000 Stocks Test R-squared: 0.01111166 
# All 1000 Stocks Validation R-squared: -0.06944591 
# All 1000 Stocks Test R-squared: -1.114231 

library(lubridate)
time_period_fit <- function(dataset, start_date, end_date, train_ratio=0.7, validation_ratio=0.2) {
  print("Year,Train Ratio,Validation Ratio,R-Squared")
  for (curr_year in year(start_date):year(end_date)) {
    
    # Get subset of dataset with data points from current year
    subset = dataset[year(dataset$Date) == curr_year, ]
    
    n = nrow(subset)
    train_size <- floor(train_ratio * n)
    validation_size <- floor(validation_ratio * n)
    
    # Split the data into training, validation, and testing
    train_data <- subset[1:train_size, ]
    validation_data <- subset[(train_size + 1):(train_size + validation_size), ]
    test_data <- subset[(train_size + validation_size + 1):n, ]
    
    # Fit OLS/Random Forest model here with tuning on validation set
    trControl <- trainControl(method = "cv",
                              number = 5, 
                              allowParallel = TRUE,
                              verboseIter = TRUE)
    tuneGrid <- expand.grid(mtry = 2:6, splitrule = c("variance"), min.node.size = c(1,10,100)
    
    rf_model_split <- train(return ~ .,
                            data = train_data,
                            method = "ranger",
                            trControl = trControl,
                            tuneGrid = tuneGrid)

    validation_predictions <- predict(rf_model_split, newdata = validation_data)
    validation_actuals <- validation_data$return
    test_predictions <- predict(rf_model_split, newdata = test_data)
    actual_values <- test_data$return
    # Calculating R-squared for test data
    ss_res <- sum((actual_values - test_predictions)^2)
    ss_tot <- sum((actual_values - actual_values)^2)
    r_squared <- 1 - (ss_res / ss_tot)
    
    # Print the R-squared for current year
    print(sprintf("%d,%f,%f,%f", curr_year, train_ratio, validation_ratio, r_squared))
  }
}

# Example usage of the function
time_period_fit(data_subset, as.Date("2009-01-01"), as.Date("2019-12-31"))
