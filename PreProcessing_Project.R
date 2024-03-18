## Loading Relevant Libraries
library(dplyr)
library(pls)


setwd("/Volumes/GoogleDrive/My Drive/RPI 2022-2026/5. Spring 2024/Adv AI ML Clarke/Project 1")
stock_data <- read.csv("Factors and Stock returns.csv")  

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

train_data = subset(sampled_data, Date <= train_end_date)
write.csv(train_data, file = "1000_unique_train.csv", row.names = FALSE)

validation_data = subset(sampled_data, Date > train_end_date & Date <= validation_end_date)
write.csv(validation_data, file = "1000_unique_validation.csv", row.names = FALSE)

test_data = subset(sampled_data, Date > validation_end_date)
write.csv(test_data, file = "1000_unique_test.csv", row.names = FALSE)


## Dimensions 
dim(train_data)
dim(validation_data)
dim(test_data)

## Don't run OLS and PLS (Still working on it)
## ------------------------------------------------------------------------------------------------
## Model 1 - Basic OLS - Predict Return using all columns (except Date and permno)

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

## For the OLS model, the in-sample R^2 is 1.4% 
## while the validation and test R^2 is -6.6% and 0.6% respectively. 

## ------------------------------------------------------------------------------------------------
## Model 2 - Partial Least Squares 
## Trying up to 10 components with LOO validation to see which has the best val performance

pls_model_loo = plsr(return ~ . - Date - permno, ncomp = 10, data = train_data, validation = "LOO")
summary(pls_model)

plot(RMSEP(pls_model_loo), legendpos = "topright")

## Plotting the RMSEP against the components shows a minimum at 6, so that's what we chose 
## despite an increase in variance explained after 6 because as components increase 
## so does the complexity. 

## Exploring different aspects of the fit 

plot(pls_model_loo, ncomp = 1, asp = 1, line = TRUE)
## The plot shows that the model is passable, but there is a lot of room for improvement

explvar(pls_model_loo)



