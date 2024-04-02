# Load required libraries
library(ggplot2)

# Sample data (replace with your actual R-squared values)
data <- data.frame(
  Model = rep(c("OLS", "OLS w Huber", "Random-Forest", "LSTM"), each = 3),
  Dataset = rep(c("All", "Top 1000", "Bottom 1000"), times = 4),
  R_squared = c(0.1744, 0.25, -7.2586, 0.018, 0.024, 0.124, -0.7559416, 0.2663911, -7.15049, 0.16823, 0.25515, -7.23703)
)

# Create bar chart
ggplot(data, aes(x = Model, y = R_squared, fill = Dataset)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "R-squared vs. Model Performance",
       x = "Model",
       y = "R-squared") +
  scale_fill_manual(values = c("blue", "#40E0D0", "yellow")) +  # Define colors for each model
  theme_minimal()


# Years from 1995 to 2015
years <- 2009:2019

r_squared_values <- c(0.630228860825841, 0.778032949968669, -1.78437662132587, 0.582760387177426, 0.649718499363029, 0.302157888662528, 1.45705029452722, 0.57436940226569, 0.0589806967406192, 1.27572653897096, 0.570140923154199)

# Create a line plot for R-squared values
plot(years, r_squared_values, type = "l", col = "blue", xlab = "Year", ylab = "R-squared Value", main = "OLS R^2")


r_squared_values <- c(0.641232611336465, 0.775282646468715, -1.75459889582575, 0.587505052038391, 0.652189238218439, 0.291811152572536, 1.45251009098568, 0.570107841733111, 0.0436169566976332, 1.27068535752895, 0.568817840565043)


# Create a line plot for R-squared values
plot(years, r_squared_values, type = "l", col = "blue", xlab = "Year", ylab = "R-squared Value", main = "OLS w/ Huber R^2")

r_squared_values <- c(0.716417, 0.811248, -3.058784, 0.595045, 0.644761, 0.013169, 1.435495, 0.572351, -0.120566, 1.263048, 0.526902)

# Create a line plot for R-squared values
plot(years, r_squared_values, type = "l", col = "blue", xlab = "Year", ylab = "R-squared Value", main = "Random-Forest R^2")

r_squared_values <- c(0.7151613, .7452231, -2.370242, 0.5862915, 0.6520201, 0.299481, 1.448716, 0.5893982, 0.06290385, 1.301173, 0.5511483)

# Create a line plot for R-squared values
plot(years, r_squared_values, type = "l", col = "blue", xlab = "Year", ylab = "R-squared Value", main = "LSTM R^2")


