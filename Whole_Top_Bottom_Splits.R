library(dplyr)
library(pls)
library(MASS)

## CREATE ENTIRE DATASET -- CALLED DATA_SUBSET
setwd("/Volumes/GoogleDrive/My Drive/RPI 2022-2026/5. Spring 2024/Adv AI ML Clarke/Project 1")
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
  shuffled_dataset <- dataset[sample(nrow(dataset)), ]
  
  # Compute sizes of train, validation, and test sets
  n <- nrow(shuffled_dataset)
  train_size <- floor(train_ratio * n)
  validation_size <- floor(validation_ratio * n)
  test_size <- n - train_size - validation_size
  
  # Split dataset
  train_data <- shuffled_dataset[1:train_size, ]
  write.csv(train_data, file = paste0(dataset_name, "_train.csv"), row.names = FALSE)
  #X_train <- as.matrix(train_data[, -which(names(train_data) %in% c("return"))])
  #X_train <- apply(X_train, 2, as.numeric)
  #y_train <- train_data$return
  
  validation_data <- shuffled_dataset[(train_size + 1):(train_size + validation_size), ]
  write.csv(validation_data, file = paste0(dataset_name, "_validation.csv"), row.names = FALSE)
  #X_val <- as.matrix(validation_data[, -which(names(validation_data) %in% c("return"))])
  #X_val <- apply(X_val, 2, as.numeric)
  #y_val <- validation_data$return
  
  test_data <- shuffled_dataset[(train_size + validation_size + 1):n, ]
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

bot_1000_data <- top_bottom_split(data_ch_bot)
bot_1000_data$train_data
bot_1000_data$validation_data
bot_1000_data$test_data


## ----------------------------------------------------------------------------------------------
## OLS MODEL WITH WHOLE DATASET -- WITHOUT HUBER LOSS 
ols_model = lm(return ~ . - Date - permno, data = train_data)
ols_insample_r2 = summary(ols_model)$r.squared
cat("In-sample R^2 in %:", ols_insample_r2*100, "\n")

# Validation Set Performance 
validation_data$predicted_returns = predict(ols_model, newdata = validation_data)
OLS_OOS_val_r2 = 1 - sum((validation_data$return - validation_data$predicted_returns)^2) / sum((validation_data$return - mean(validation_data$return))^2)
cat("Out-of-sample Validation R^2 in %:", OLS_OOS_val_r2*100, "\n")

# Test Set Performance 
test_data$predicted_returns = predict(ols_model, newdata = test_data)
OLS_OOS_test_r2 = 1 - sum((test_data$return - test_data$predicted_returns)^2) / sum((test_data$return - mean(test_data$return))^2)
cat("Out-of-sample Test R^2 in %:", OLS_OOS_test_r2*100, "\n")

## ----------------------------------------------------------------------------------------------
## TOP 1000 PERFORMANCE -- WITHOUT HUBER LOSS
## Train Performance 
ols_model_top1000 = lm(return ~ . - permno, data = top_1000_data$train_data)
ols_insample_r2 = summary(ols_model_top1000)$r.squared
cat("In-sample R^2 in %:", ols_insample_r2*100, "\n")

# Validation Performance 
top_1000_data$validation_data$predicted_returns = predict(ols_model_top1000, newdata = top_1000_data$validation_data)
OLS_OOS_val_r2 = 1 - sum((top_1000_data$validation_data$return - top_1000_data$validation_data$predicted_returns)^2) / sum((top_1000_data$validation_data$return - mean(top_1000_data$validation_data$return))^2)
cat("Out-of-sample Validation R^2 in %:", OLS_OOS_val_r2*100, "\n")

# Test Set Performance 
top_1000_data$test_data$predicted_returns = predict(ols_model_top1000, newdata = top_1000_data$test_data)
OLS_OOS_test_r2 = 1 - sum((top_1000_data$test_data$return - top_1000_data$test_data$predicted_returns)^2) / sum((top_1000_data$test_data$return - mean(top_1000_data$test_data$return))^2)
cat("Out-of-sample Test R^2 in %:", OLS_OOS_test_r2*100, "\n")

## ----------------------------------------------------------------------------------------------
## BOTTOM 1000 PERFORMANCE -- WITHOUT HUBER LOSS
## Train Performance 
ols_model_bot1000 = lm(return ~ . - permno, data = bot_1000_data$train_data)
ols_insample_r2 = summary(ols_model_bot1000)$r.squared
cat("In-sample R^2 in %:", ols_insample_r2*100, "\n")

# Validation Performance 
bot_1000_data$validation_data$predicted_returns = predict(ols_model_bot1000, newdata = bot_1000_data$validation_data)
OLS_OOS_val_r2 = 1 - sum((bot_1000_data$validation_data$return - bot_1000_data$validation_data$predicted_returns)^2) / sum((bot_1000_data$validation_data$return - mean(bot_1000_data$validation_data$return))^2)
cat("Out-of-sample Validation R^2 in %:", OLS_OOS_val_r2*100, "\n")

# Test Set Performance 
bot_1000_data$test_data$predicted_returns = predict(ols_model_bot1000, newdata = bot_1000_data$test_data)
OLS_OOS_test_r2 = 1 - sum((bot_1000_data$test_data$return - bot_1000_data$test_data$predicted_returns)^2) / sum((bot_1000_data$test_data$return - mean(bot_1000_data$test_data$return))^2)
cat("Out-of-sample Test R^2 in %:", OLS_OOS_test_r2*100, "\n")


## WHOLE DATASET WITH HUBER LOSS
huber_model <- rlm(return ~ . - Date - permno, data = train_data, method = "M", psi = psi.huber)
test_data$predicted_returns_huber = predict(huber_model, newdata = test_data)
TSS_huber = sum(test_data$return-mean(test_data$return)^2)
RSS_huber = sum(test_data$return-(test_data$predicted_returns_huber)^2)
insample_R_huber = 1 - (RSS_huber/TSS_huber)
insample_R_huber

## TOP 1000 WITH HUBER LOSS
huber_model_top1000 <- rlm(return ~ . - permno, data = top_1000_data$train_data, method = "M", psi = psi.huber)
top_1000_data$test_data$predicted_returns_huber = predict(huber_model_top1000, newdata = top_1000_data$test_data)
TSS_huber_top = sum(top_1000_data$test_data$return-mean(top_1000_data$test_data$return)^2)
RSS_huber_top = sum(top_1000_data$test_data$return-(top_1000_data$test_data$predicted_returns_huber)^2)
R_huber_top = 1 - (RSS_huber_top/TSS_huber_top)
R_huber_top


## BOTTOM 1000 WITH HUBER LOSS
huber_model_bot1000 <- rlm(return ~ . - permno, data = bot_1000_data$train_data, method = "M", psi = psi.huber)
bot_1000_data$test_data$predicted_returns_huber = predict(huber_model_bot1000, newdata = bot_1000_data$test_data)
TSS_huber_bot = sum(bot_1000_data$test_data$return-mean(bot_1000_data$test_data$return)^2)
RSS_huber_bot = sum(bot_1000_data$test_data$return-(bot_1000_data$test_data$predicted_returns_huber)^2)
R_huber_bot = 1 - (RSS_huber_bot/TSS_huber_bot)
R_huber_bot

bot_1000_data$validation_data$predicted_returns_huber = predict(huber_model_bot1000, newdata = bot_1000_data$validation_data)
TSS_huber_bot_val = sum(bot_1000_data$validation_data$return-mean(bot_1000_data$validation_data$return)^2)
RSS_huber_bot_val = sum(bot_1000_data$validation_data$return-(bot_1000_data$validation_data$predicted_returns_huber)^2)
R_huber_bot_val = 1 - (RSS_huber_bot_val/TSS_huber_bot_val)
R_huber_bot_val



