# Load required libraries
library(ggplot2)

# Sample data (replace with your actual R-squared values)
data <- data.frame(
  Model = rep(c("OLS", "OLS w Huber", "Random-Forest", "LSTM"), each = 3),
  Dataset = rep(c("All", "Top 1000", "Bottom 1000"), times = 4),
  R_squared = c(0.00829, -0.0016, 0.0011, -0.0028, -0.0095, 0.139, 0.017, 0.0073, 0.0034, 0.0024, 0.0016, 0.0032)
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

r_squared_values <- c(-0.65901047546989, -0.361633125065703,  -0.127389721739456, -0.149096203259179,  -0.0140088785045793, -0.0625184327543851, 0.0159722748435088, -0.140255849331574, -0.00405752795992753, -1.09401190149182, 
                      -0.120278166231536)

# Create a line plot for R-squared values
plot(years, r_squared_values, type = "l", col = "blue", xlab = "Year", ylab = "R-squared Value", main = "OLS R^2")


r_squared_values <- c(-0.392964865535784, -0.37223910336796, -0.152649743655255, -0.0941695987012028,  -0.0240521589747462, -0.0882119604060143, 0.0282757582000654, -0.131408316811164, -0.0188185514335872, -1.04948306007856, 
                      -0.145693234556018)


# Create a line plot for R-squared values
plot(years, r_squared_values, type = "l", col = "blue", xlab = "Year", ylab = "R-squared Value", main = "OLS w/ Huber R^2")

r_squared_values <- c(-0.08998296, -0.1090994, -0.007030313, -0.0784509, -0.05699684, -0.001592706, -0.3118906,-0.09692005, -0.05532857, -0.5603918, 
                      -0.05986084)

# Create a line plot for R-squared values
plot(years, r_squared_values, type = "l", col = "blue", xlab = "Year", ylab = "R-squared Value", main = "Random-Forest R^2")

r_squared_values <- c(-0.08998296, -0.1090994, -0.2608575, -0.06747795, -0.01394878, -0.03975618, -0.04293517,-0.04174138, -0.002450364, -0.7677409, 
                      -0.04301312)

# Create a line plot for R-squared values
plot(years, r_squared_values, type = "l", col = "blue", xlab = "Year", ylab = "R-squared Value", main = "LSTM R^2")


