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
  filter(substr(SICCD, 1, 2) == "60") %>%
  select(-SICCD, -NCUSIP, -TICKER, -COMNAM)
data_subset = na.omit(specific_siccd_data)
data_subset$Date = as.Date(data_subset$Date)
train_end_date = as.Date("2017-12-31")
validation_end_date = as.Date("2018-12-31")

train_data = subset(data_subset, Date <= train_end_date)
validation_data = subset(data_subset, Date > train_end_date & Date <= validation_end_date)
test_data = subset(data_subset, Date > validation_end_date)
train_data <- select(train_data, -Date)
validation_data <- select(validation_data, -Date)
test_data <- select(test_data, -Date)
train_data <- select(train_data, -PERMNO)
validation_data <- select(validation_data, -PERMNO)
test_data <- select(test_data, -PERMNO)
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
results_top <- train_and_evaluate(train_data, validation_data, test_data)
importance_data <- results_top$variable_importance$importance
top_features <- rownames(head(importance_data, 10))

model <- results_top$model
train_data$Date <- as.numeric(train_data$Date)
features_of_interest <- c('mom1m', 'mom12m','chmom', 'indmom', 'mom36m', 'turn', 'mvel1', 'dolvol', 'ill', 'zerotrade','baspread', 'retvol', 'idiovol','beta','betasq','ep','sp','agr','nincr','return..t.1.')
library(iml)
predictor <- Predictor$new(model, data = train_data, y = train_data$return)
ale_plots <- list()
for (feature in features_of_interest) {
  if (feature %in% names(train_data)) {
    feature_effect <- FeatureEffect$new(predictor, feature = feature)
    ale_plot <- feature_effect$plot()
    ale_plots[[feature]] <- ale_plot
  } else {
    print(paste("Feature not found:", feature))
  }
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
selected_data <- train_data[features_of_interest]
correlation_matrix <- cor(selected_data, use="complete.obs")  # 'use' argument handles missing values
corrplot(correlation_matrix, method = "circle", type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black")
print(correlation_matrix)
