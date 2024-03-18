library(dplyr)
library(data.table)
library(tensorflow)
library(keras)

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

data_ch_top <- data_subset %>% 
  arrange(desc(mvel1)) %>%
  group_by(Date) %>%
  slice_head(n = 1000) %>%
  ungroup() %>%
  select(-Date)

# Sort data by 'mvel1' in descending order, group by 'DATE',
# select bottom 1000 rows for each group, and reset index
data_ch_bot <- data_subset %>% 
  arrange(desc(mvel1)) %>%
  group_by(Date) %>%
  slice_tail(n = 1000) %>%
  ungroup() %>%
  select(-Date)

set.seed(123)

## Sample Data-set with 1000 Unique permnos 
unique_stocks = unique(data_subset$permno)
sampled_permnos = sample(unique_stocks, size = 1000, replace = FALSE)
sampled_data = data_subset[data_subset$permno %in% sampled_permnos, ]

## Output Sampled Data as a CSV
write.csv(sampled_data, file = "1000_unique_stocks.csv", row.names = FALSE)

## Time Based Train-Validation-Test split 
sampled_data$Date = as.Date(sampled_data$Date)
train_end_date = as.Date("2017-12-31")
validation_end_date = as.Date("2018-12-31")

X <- sampled_data[, -which(names(sampled_data) %in% c("Date", "return"))]
y <- sampled_data$return

train_data = subset(sampled_data, Date <= train_end_date)
write.csv(train_data, file = "1000_unique_train.csv", row.names = FALSE)
X_train <- as.matrix(train_data[, -which(names(train_data) %in% c("Date", "return"))])
X_train <- apply(X_train, 2, as.numeric)
y_train <- train_data$return

validation_data = subset(sampled_data, Date > train_end_date & Date <= validation_end_date)
write.csv(validation_data, file = "1000_unique_validation.csv", row.names = FALSE)
X_val <- as.matrix(validation_data[, -which(names(validation_data) %in% c("Date", "return"))])
X_val <- apply(X_val, 2, as.numeric)
y_val <- validation_data$return

test_data = subset(sampled_data, Date > validation_end_date)
write.csv(test_data, file = "1000_unique_test.csv", row.names = FALSE)
X_test <- as.matrix(test_data[, -which(names(test_data) %in% c("Date", "return"))])
X_test <- apply(X_test, 2, as.numeric)
y_test <- test_data$return

model <- keras_model_sequential() %>%
  layer_lstm(units = 50, return_sequences = TRUE, input_shape = c(ncol(X_train), 1)) %>%
  layer_lstm(units = 50) %>%
  layer_dense(units = 1)
y_test
# Compile the model
model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

# Train model
history <- model %>% fit(
  X_train, y_train,
  epochs = 10,
  batch_size = 32,
  validation_data = list(X_val, y_val)
)

# Evlauate model

predicitons <- model %>% predict(X_test)
mse <- mean((predictions - y_test)^2)
