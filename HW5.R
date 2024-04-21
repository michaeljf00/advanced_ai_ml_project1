rm(list=ls())
library(dplyr)
library(iml)
library(shapr)
library(fastshap)
library(ranger)
library(data.table)
library(ggplot2)
library(caret)

set.seed(123)

setwd("/Users/michaeljoshua/Desktop/programming_projects/advanced_ai_ml_for_finance")
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

# Subset the data to 10000 data points
data_subset <- data_subset[sample(nrow(data_subset), 10000), ]

# Fit a random forest model with ranger:
x <- data_subset[, c("mom1m", "mom12m", "chmom", "indmom", "mom36m", "turn", "mvel1", "dolvol", "ill", "zerotrade", "baspread", "retvol", "idiovol", "beta", "betasq", "ep", "sp", "agr", "nincr")]
y <- as.numeric(data_subset$return)


my.args <- list("x"=x,"y"=y,
                "metric" = "MSE",
                "maximize" = TRUE,
                "method" = "ranger",
                "tuneGrid" = tuneGrid <- expand.grid( 
                  mtry = 2:6,
                  splitrule = c("variance"),
                  min.node.size = c(1,10,100)
                )
)

model_rf <- do.call(caret::train,my.args)

X <- x

model.inclass <- Predictor$new(model_rf, data = X, class = "1")

get.lemon <- function(k=2,obs){
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
k <- 5 # Number of Features
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


if (!file.exists("lemon_explanations")) {
  dir.create("lemon_explanations")
}

unique_features <- unique(lemon.results$feature)
unique_features
for (i in 1:length(unique_features)){
  print(i)
  
  pdf_name <- paste0("lemon_explanations/explanation_", unique_features[i], ".pdf")
  pdf(pdf_name)
  
  df.cscore <- lemon.results[which(lemon.results$feature == unique_features[i]),]
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
}

length(lemon.results)
lemon.results
# Print the result
print(average_obs)
# Shaple values using iml package

pred_wrapper <- function(object, newdata) {
  p <- predict.train(object, data = newdata)
}


# 2. generate a predictor conatiner to hold the model info:
predictor <- Predictor$new(model.inclass$model,
                           data=x,y=y,
                           predict.function = pred_wrapper)
model.inclass$model

# Get the shapley values for a single instance:
shapley_100 <- Shapley$new(predictor, x.interest = x[100,])

get.lshapley <- function(k=2,obs){
  x.interest <- X[obs, ]
  shapley <- LocalModel$new(model.inclass, 
                          x.interest = x.interest, 
                          k = k)
  return(lemon$results)
}

if (!file.exists("shapley_plots")) {
  dir.create("shapley_plots")
}

# Shapley values for number of instances for each bin 
for (q in 1:Q){
  for (i in 1:N){
    tmp <- Shapley$new(predictor, x.interest = X[sample.rows[i, q],])
    pdf_name <- paste0("shapley_plots/shapley_plot_Q:", q, "_i:", i, ".pdf")
    pdf(pdf_name)
    
    # Generate and save Shapley plot for specific Q value and index
    print(tmp$plot())  # Replace with your actual code for generating the Shapley plot

    dev.off()
  }
}

shapley.results

tmp <- Shapley$new(predictor, x.interest = X[sample.rows[1, 1],])
print(tmp$plot())

