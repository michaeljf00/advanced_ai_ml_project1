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
# ============================================================================================
train_and_evaluate <- function(train_data, validation_data, test_data) {
  trControl <- trainControl(method = "cv",
                            number = 3, 
                            allowParallel = TRUE,
                            verboseIter = TRUE)
  tuneGrid <- expand.grid(
    mtry = 2:6,
    splitrule = c("variance"),
    min.node.size = 1
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
  test_r_squared <-  1 - sum((test_data$return - test_predictions)^2) / sum((test_data$return))
  list(
    model = final_model,
    validation_r_squared = validation_r_squared,
    test_r_squared = test_r_squared,
    variable_importance = var_importance
  )
}
sampled_data <- top_1000_data$train_data[sample(nrow(top_1000_data$train_data), 100), ]
results_top <- train_and_evaluate(top_1000_data$train_data, top_1000_data$validation_data, top_1000_data$test_data)
importance_data <- results_top$variable_importance$importance
top_features <- rownames(head(importance_data, 10))

library(pdp)
library(ALEPlot)
library(gridExtra)

# ============
# pdp
# ============
model <- results_top$model
features_of_interest <- c('mom12m', 'indmom', 'baspread', 'chmom', 'retvol', 
                          'mom36m', 'idiovol', 'mvel1', 'mom1m', 'turn')
plot_list <- list()
for (feature in features_of_interest) {
  pdp_obj <- partial(model, pred.var = feature, grid.resolution = 30, train = sampled_data)
  plot_pdp <- autoplot(pdp_obj) +
    theme_light() +
    labs(x = feature, y = "Partial dependence") +
    theme(legend.position = "none")
  plot_list[[feature]] <- plot_pdp
}
grid.arrange(grobs = plot_list, ncol = 2)

sampled_data <- top_1000_data$train_data[sample(nrow(top_1000_data$train_data), 100), ]
# ============
# ICE
# ============
ice_plots <- list()
for (feature in features_of_interest) {
  ice_obj <- partial(model, pred.var = feature, grid.resolution = 30, 
                     train = sampled_data, ice = TRUE)
  ice_plot <- autoplot(ice_obj) +
    theme_light() +
    labs(x = feature, y = "Prediction") +
    theme(legend.position = "none")
  ice_plots[[feature]] <- ice_plot
}

gridExtra::grid.arrange(grobs = ice_plots, ncol = 2)
# ============
# ALE (i tired to use ALEPLOT but it keep running to an error that I can't fix so i used iml for ALE)
# https://cran.r-project.org/web/packages/iml/iml.pdf
# ============
library(iml)
predictor <- Predictor$new(model, data = top_1000_data$train_data, y = top_1000_data$train_data$return)
ale_plots <- list()
for (feature in features_of_interest) {
  feature_effect <- FeatureEffect$new(predictor, feature = feature)
  ale_plot <- feature_effect$plot()
  ale_plots[[feature]] <- ale_plot
}
gridExtra::grid.arrange(grobs = ale_plots, ncol = 2)
top_features_importance <- importance_data[rownames(importance_data) %in% top_features, ]
importance_df <- as.data.frame(importance_data)
importance_df$Feature <- rownames(importance_df)
ggplot(importance_df, aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_bar(stat = "identity") +
  theme_light() +
  labs(x = "Feature", y = "Importance") +
  coord_flip()

library(corrplot)
selected_data <- top_1000_data$train_data[features_of_interest]
correlation_matrix <- cor(selected_data, use="complete.obs")  # 'use' argument handles missing values
corrplot(correlation_matrix, method = "circle", type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black")
print(correlation_matrix)

