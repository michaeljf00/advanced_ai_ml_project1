# Load required libraries
library(ggplot2)

# Sample data (replace with your actual R-squared values)
data <- data.frame(
  Model = rep(c("OLS", "Random-Forest", "LSTM"), each = 3),
  Dataset = rep(c("all_dataset", "top_1000_dataset", "bottom_1000_dataset"), times = 3),
  R_squared = c(0.7, 0.65, 0.6, 0.8, 0.75, 0.7, 0.6, 0.55, 0.5) # Replace with your R-squared values
)

# Create bar chart
ggplot(data, aes(x = Dataset, y = R_squared, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "R-squared for Stock Prediction Models",
       x = "Dataset",
       y = "R-squared") +
  scale_fill_manual(values = c("blue", "#40E0D0", "yellow")) +  # Define colors for each model
  theme_minimal()

