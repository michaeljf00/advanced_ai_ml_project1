# Load the required library
library(data.table)

# Read the data
data <- read.csv("factors_and_stock_returns.csv")  # Replace "your_data.csv" with the path to your data file

# Convert 'Date' column to Date format
data$Date <- as.Date(data$Date)

# Calculate market equity
data$market_equity <- data$mvel1 * data$sp

# Group data by month
grouped_data <- split(data, format(data$Date, "%Y-%m"))

# Initialize empty lists to store results
top_1000_stocks_per_month <- list()
bottom_1000_stocks_per_month <- list()

# Loop through each month
for (month in names(grouped_data)) {
  group <- grouped_data[[month]]
  
  # Sort the data by market equity within each month's group
  sorted_group <- group[order(-group$market_equity), ]
  
  # Get top 1000 stocks for the current month
  top_1000_stocks <- head(sorted_group, 1000)
  
  # Get bottom 1000 stocks for the current month
  bottom_1000_stocks <- tail(sorted_group, 1000)
  
  # Append results to the overall list
  top_1000_stocks_per_month[[month]] <- top_1000_stocks
  bottom_1000_stocks_per_month[[month]] <- bottom_1000_stocks
}

# Combine results into data frames
top_1000_stocks_per_month_df <- do.call(rbind, top_1000_stocks_per_month)
bottom_1000_stocks_per_month_df <- do.call(rbind, bottom_1000_stocks_per_month)

# Print the top and bottom 10 stocks for verification
print("Top 10 Stocks Each Month:")
print(head(top_1000_stocks_per_month_df, 10))
print("\nBottom 10 Stocks Each Month:")
print(head(bottom_1000_stocks_per_month_df, 10))

SimpleRNN <- keras_model_sequential()
SimpleRNN %>%
  layer_simple_rnn(units=20, 
                   activation='tanh', 
                   kernel_initializer="glorot_uniform", 
                   bias_initializer="glorot_uniform", 
                   recurrent_initializer="orthogonal", 
                   kernel_regularizer="l1", 
                   input_shape=c(ncol(x_train_reg),1), 
                   unroll=T, 
                   stateful=F) %>%
  layer_dense(units=1, 
              kernel_initializer="glorot_uniform", 
              bias_initializer="glorot_uniform", 
              kernel_regularizer="l1") %>%
  compile(loss = 'mean_squared_error', optimizer = 'adam')
SimpleRNN