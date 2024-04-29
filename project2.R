rm(list=ls())
library(dplyr)
library(pls)
library(caret)
library(ranger)

setwd("/Users/michaeljoshua/Desktop/programming_projects/advanced_ai_ml_project1")
stock_data = read.csv("Factors and Stock returns.csv")
indus_data = read.csv("Companies.csv")
names(stock_data)[names(stock_data) == "permno"] <- "PERMNO"
merged_data <- merge(stock_data, indus_data, by = "PERMNO", all.x = TRUE)
merged_data$SICCD <- as.character(merged_data$SICCD)
industry <- 35
specific_siccd_data <- merged_data %>% 
  filter(substr(SICCD, 1, 2) == industry) %>%
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
# Specify the desired order of features
desired_order <- c("indmom", "mom12m", "baspread", "chmom", "mom1m", "mom36m",
                   "return..t.1.", "retvol", "idiovol", "mvel1", "ep", "sp", 
                   "betasq", "beta", "ill", "agr", "turn", "zerotrade", "dolvol", "nincr")

top_features <- desired_order

importance_df <- importance_df %>% 
  filter(Feature %in% top_features) %>%
  mutate(Feature = factor(Feature, levels = desired_order))

# Plot the data
ggplot(importance_df, aes(x = Feature, y = Overall)) +
  geom_bar(stat = "identity") +
  coord_flip() +  # Flips the axes so features are on the y-axis
  theme_light() +
  labs(x = "Feature", y = "Importance")


library(corrplot)
selected_data <- train_data[features_of_interest]
correlation_matrix <- cor(selected_data, use="complete.obs")  # 'use' argument handles missing values
corrplot(correlation_matrix, method = "circle", type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black")
print(correlation_matrix)

X <- train_data
model.inclass <- Predictor$new(model, data = X, class = "1")

get.lemon <- function(k=20,obs){
  x.interest <- X[obs, ]
  lemon <- LocalModel$new(model.inclass, 
                          x.interest = x.interest, 
                          k = k)
  return(lemon$results)
}

Q <- 5 # quintiles
qq = quantile(X$dolvol, probs = seq(0, 1, 1/Q))

# bins has the row numbers of the quantiles:
bins <- list(NA)
for (i in 1:Q){
  bins[[i]] <- which(X$dolvol >= qq[i] & X$dolvol < qq[i+1])
  
  # Print the min/mean/max of the bin values to make sure it worked:
  print(paste(min(X$dolvol[bins[[i]]]),
              round(mean(X$dolvol[bins[[i]]])),
              max(X$dolvol[bins[[i]]]),sep=","))
}

# Sample each bin and save the results:
set.seed(100)
k <- 20 # Number of Features
# The nuber of instances per bin:
N <- 10 

# Draw the random samples:
sample.rows <- as.data.frame(matrix(NA,ncol=Q,nrow=N))
for (i in 1:Q){
  sample.rows[,i] <- sample(bins[[i]],N,replace=F)
}

for (q in 1:Q){
  for (i in 1:N){
    tmp <- get.lemon(k=k,obs=sample.rows[i,q])
    tmp$Q <- q
    tmp$obs <- i
    print(tmp$feature)
    if (q == 1 & i == 1){
      lemon.results <- tmp
    }
    else{
      lemon.results <- rbind(lemon.results,tmp)
    }
  }
}


if (!file.exists(paste0("lemon_explanations_", industry))) {
  dir.create(paste0("lemon_explanations_", industry))
}

unique_features <- unique(lemon.results$feature)
unique_features

pdf_name <- paste0("lemon_explanations_", industry, "/explanation_", unique_features[i], ".pdf")
pdf(pdf_name)

indmom <- unique_features[4]
dolvol <- unique_features[8]

#Adjust this code here
df.cscore <- lemon.results[which(lemon.results$feature == unique_features[i]),]
df.cscore
df.cscore$Q <- as.factor(df.cscore$Q)

p.cscore <- ggplot(data=df.cscore,aes(x=obs,y=effect,colour=Q)) +
  geom_point() +
  scale_colour_manual(name="Q",values = c("red", "orange", "blue", "green","black"))

print(p.cscore)

p.cscore.Q <- ggplot(data=df.cscore,aes(x=Q,y=effect,colour=Q)) +
  geom_point() +
  scale_colour_manual(name="Q",values = c("red", "orange", "blue", "green","black"))

print(p.cscore.Q)

dev.off()

pdf_name <- paste0("lemon_explanations_", industry, "/explanation_dolvol_indmom.pdf")
pdf(pdf_name)

# Subset data for dolvol and indmom features
df_dolvol <- lemon.results[lemon.results$feature == "dolvol", ]
df_indmom <- lemon.results[lemon.results$feature == "indmom", ]

# Create ggplot
p <- ggplot() +
  geom_point(data = df_dolvol, aes(x = obs, y = effect, color = "dolvol")) +
  geom_point(data = df_indmom, aes(x = obs, y = effect, color = "indmom")) +
  scale_color_manual(name = "Feature", values = c("dolvol" = "red", "indmom" = "blue"), labels = c("dolvol", "indmom")) +
  labs(title = paste0("Lime Values for dolvol and indmom - SICID ", industry),
       x = "Observations", y = "Effect")

# Print and save plot
print(p)
dev.off()

length(lemon.results)
lemon.results
y <- as.numeric(train_data$return)
pred_wrapper <- function(object, newdata) {
  p <- predict.train(object, data = newdata)
}


# 2. generate a predictor conatiner to hold the model info:
predictor <- Predictor$new(model.inclass$model,
                           data=X,y=y,
                           predict.function = pred_wrapper)
model.inclass$model

# Get the shapley values for a single instance:
shapley_100 <- Shapley$new(predictor, x.interest = X[100,])

get.lshapley <- function(k=2,obs){
  x.interest <- X[obs, ]
  shapley <- LocalModel$new(model.inclass, 
                            x.interest = x.interest, 
                            k = k)
  return(lemon$results)
}

# Shapley values for number of instances for each bin 
indmom_phis <- c()
dolvol_phis <- c()

for (q in 1:Q){
  for (i in 1:N){
    tmp <- Shapley$new(predictor, x.interest = X[sample.rows[i, q],])
    features <- tmp$results$feature
    phis <- tmp$results$phi
    print(features[5])
    print(phis[5])
    print(features[9])
    print(phis[9])
    indmom_phis <- rbind(indmom_phis, phis[5])
    dolvol_phis <- rbind(dolvol_phis, phis[9])
  }
}

# Graph feature values for both industries
# Assuming you have already computed the Shapley values for dolvol and indmom for industries with SICID starting with 35 and 60
# Let's call them dolvol35, indmom35, dolvol60, indmom60

library(ggplot2)

# Create data frames for plotting
data <- data.frame(phi_index = seq_along(dolvol_phis),
                     dolvol = dolvol_phis,
                     indmom = indmom_phis)

# Reshape data for ggplot
data_long <- tidyr::pivot_longer(data, cols = c(dolvol, indmom), names_to = "Feature", values_to = "Phi")

if (!file.exists(paste0("explainability_plots_", industry))) {
  dir.create(paste0("explainability_plots_", industry))
}

# Plot
pdf_name <- paste0("shapley_explanation_", industry, ".pdf")
pdf(pdf_name)

plot <- ggplot(data = data_long, aes(x = phi_index, y = Phi, color = Feature)) +
  geom_point() +
  geom_line() +
  labs(title = paste("Shapley Values for SICID", industry),
       x = "Index",
       y = "Shapley Value") +
  scale_color_manual(values = c("dolvol" = "blue", "indmom" = "red")) +
  theme_minimal()
plot
dev.off()
