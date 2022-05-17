
import random
from AdrianPack.Aplot import Default

a, b = 4, 1

# Create a list with 200 x values between -10 and 10
x = [val * 0.1 for val in list(range(-10, 10))]
# Calculate values of the function y = sin(x) between 0 and 20
y = [(a * val + b) + random.random() for val in x]

# Defining x and y errors
x_err = [random.randrange(-10, 10) * 0.01 for val in list(range(-10, 10))]
y_err = [random.randrange(-10, 10) * 0.01 for val in list(range(-10, 10))]

# Make a Default object, include x and y errors; degree; labels
plot = Default(x, y, x_err=x_err, y_err=y_err, degree=1
               , x_label="x-axis X [-]", y_label="y-axis Y [-]",
               save_as="simple_fit.png")

# Run the plot
plot()
