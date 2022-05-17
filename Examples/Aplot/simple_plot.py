
from math import sin
from AdrianPack.Aplot import Default

# Create a list with 200 x values between -10 and 10
x = [val * 0.1 for val in list(range(-100, 100))]
# Calculate values of the function y = sin(x) between 0 and 20
y = [sin(val) for val in x]

# Make a Default object
plot = Default(x, y, save_as="simple_plot.png")

# Run the plot
plot()

