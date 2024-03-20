rm(list=ls())
library(dplyr)
library(data.table)
library(caret)
setwd("Documents/RPI/Spring2024/Risk analytics/advanced_ai_ml_project1/")
# Read the data
stock_data <- read.csv("factors_and_stock_returns.csv")

# Convert 'Date' column to Date format
stock_data$Date <- as.Date(stock_data$Date)

## Sub-setting the Data Frame to be between 2009 and 2019 
start_date = as.Date("2009-01-31")
end_date = as.Date("2019-12-31")
data_subset = stock_data[stock_data$Date >=start_date  & stock_data$Date <=end_date, ]

sum(is.na(data_subset))
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

# Get top 1000 and bottom 100 stocks
top_1000 <- data_subset %>% 
  arrange(desc(mvel1)) %>%
  group_by(Date) %>%
  slice_head(n = 1000) %>%
  ungroup() %>%
  select(-Date)

bottom_1000 <- data_subset %>% 
  arrange(desc(mvel1)) %>%
  group_by(Date) %>%
  slice_tail(n = 1000) %>%
  ungroup() %>%
  select(-Date)

set.seed(123)

## Sample Data-set with 1000 Unique permnos 
unique_stocks = unique(data_subset$permno)
sampled_permnos = sample(unique_stocks, size = 1000, replace = FALSE)
sampled_1000_unique = data_subset[data_subset$permno %in% sampled_permnos, ]

train_test_split <- function(sampled_data) {
  dataset_name <- deparse(substitute(sampled_data))
  
  ## Output Sampled Data as a CSV
  write.csv(sampled_data, file = paste0(dataset_name, "_stocks.csv"), row.names = FALSE)
  
  ## Time Based Train-Validation-Test split 
  sampled_data$Date = as.Date(sampled_data$Date)
  train_end_date = as.Date("2017-12-31")
  validation_end_date = as.Date("2018-12-31")
  
  X <- sampled_data[, -which(names(sampled_data) %in% c("Date", "return"))]
  y <- sampled_data$return
  
  train_data = subset(sampled_data, Date <= train_end_date)
  write.csv(train_data, file = paste0(dataset_name, "_train.csv"), row.names = FALSE)
  X_train <- as.matrix(train_data[, -which(names(train_data) %in% c("Date", "return"))])
  X_train <- apply(X_train, 2, as.numeric)
  y_train <- train_data$return
  
  validation_data = subset(sampled_data, Date > train_end_date & Date <= validation_end_date)
  write.csv(validation_data, file = paste0(dataset_name, "_validation.csv"), row.names = FALSE)
  X_val <- as.matrix(validation_data[, -which(names(validation_data) %in% c("Date", "return"))])
  X_val <- apply(X_val, 2, as.numeric)
  y_val <- validation_data$return
  
  test_data = subset(sampled_data, Date > validation_end_date)
  write.csv(test_data, file = paste0(dataset_name, "_test.csv"), row.names = FALSE)
  X_test <- as.matrix(test_data[, -which(names(test_data) %in% c("Date", "return"))])
  X_test <- apply(X_test, 2, as.numeric)
  y_test <- test_data$return
  
  
  return(list(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test))
}

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
  X_train <- as.matrix(train_data[, -which(names(train_data) %in% c("return"))])
  X_train <- apply(X_train, 2, as.numeric)
  y_train <- train_data$return
  
  validation_data <- shuffled_dataset[(train_size + 1):(train_size + validation_size), ]
  write.csv(validation_data, file = paste0(dataset_name, "_validation.csv"), row.names = FALSE)
  X_val <- as.matrix(validation_data[, -which(names(validation_data) %in% c("return"))])
  X_val <- apply(X_val, 2, as.numeric)
  y_val <- validation_data$return
  
  test_data <- shuffled_dataset[(train_size + validation_size + 1):n, ]
  write.csv(test_data, file = paste0(dataset_name, "_test.csv"), row.names = FALSE)
  X_test <- as.matrix(test_data[, -which(names(test_data) %in% c("return"))])
  X_test <- apply(X_test, 2, as.numeric)
  y_test <- test_data$return
  
  # Return split datasets
  return(list(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test))
}
sampled_data_splits <- train_test_split(sampled_1000_unique)
train_data <- sampled_data_splits$X_train
train_labels <- sampled_data_splits$y_train

trControl <- trainControl(method = "cv",
                          number = 5, 
                          allowParallel = TRUE,
                          verboseIter = TRUE)
tuneGrid <- expand.grid(.mtry = 1:6)

rf_model <- train(x = train_data, y = train_labels,
                  method = "rf",
                  trControl = trControl,
                  tuneGrid = tuneGrid)

print(rf_model)


# Prepare the test data
X_test <- sampled_data_splits$X_test
y_test <- sampled_data_splits$y_test

# Make predictions on the test data
predictions <- predict(rf_model, X_test)
rmse <- sqrt(mean((y_test - predictions)^2))
print(paste("RMSE:", rmse))

mae <- mean(abs(y_test - predictions))
print(paste("MAE:", mae))
R_squared <- 1 - sum((y_test - predictions)^2) / sum((y_test - mean(y_test))^2)
print(paste("R-squared:", R_squared))

