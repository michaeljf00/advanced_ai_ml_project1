# Load required libraries
library(ggplot2)

# Sample data (replace with your actual R-squared values)
data <- data.frame(
  Model = rep(c("OLS", "OLS w Huber", "Random-Forest", "LSTM"), each = 3),
  Dataset = rep(c("All", "Top 1000", "Bottom 1000"), times = 4),
  R_squared = c(0.0829, 0.0025, 0.0031, -0.0028, 0.0115, 0.1263, 0.017, 0.0073, 0.0034, 0.0024, 0.0016, 0.0032)
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

r_squared_values <- c(-0.02833381, -0.05994907,  0.1274594, -0.009538831,  0.0275336, -0.005341961, 0.02664486, -0.02432666, 0.01272472, 0.09356996, 
                      -0.01845706)

# Create a line plot for R-squared values
plot(years, r_squared_values, type = "l", col = "blue", xlab = "Year", ylab = "R-squared Value", main = "OLS R^2")

r_squared_values <- c(-0.02920133, -0.06038982, 0.09717653, -0.012885,  0.01639624, 0.004670284, 0.02517152, -0.02450865, 0.01072242, 0.09353364, 
                      -0.01449914)

# Create a line plot for R-squared values
plot(years, r_squared_values, type = "l", col = "blue", xlab = "Year", ylab = "R-squared Value", main = "OLS w/ Huber R^2")

r_squared_values <- c(-0.08998296, -0.1090994, -0.007030313, -0.0784509, -0.05699684, -0.001592706, -0.3118906,-0.09692005, -0.05532857, -0.5603918, 
                      -0.05986084)

# Create a line plot for R-squared values
plot(years, r_squared_values, type = "l", col = "blue", xlab = "Year", ylab = "R-squared Value", main = "Random-Forest R^2")

r_squared_values <- c(-0.1224982, -0.3698727, -0.2608575, -0.06747795, -0.01394878, -0.03975618, -0.04293517,-0.04174138, -0.002450364, -0.7677409, 
                      -0.04301312)

# Create a line plot for R-squared values
plot(years, r_squared_values, type = "l", col = "blue", xlab = "Year", ylab = "R-squared Value", main = "LSTM R^2")


