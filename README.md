# AdrianPack

This is a small complilation of functions that have been made to make life easier. These functions can be applied in a variety of ways, mainly they are used to visualise data, parse data from files to python objects or simply manipulate data. The project is constantly updated and functions are continuesly being added. 

# Contents
1. [How to install?](#How-to-install?)
2. [Dependecies](#Dependecies)
3. [A Short overview](#A-Short-overview)
4. [Aplot.py](#Aplot.py)

    4.1 [Description](##Description)
   
    4.2 [How to use?](##How-to-use?)
   
    4.3 [Examples](##Examples)

      * [Creating a simple plot](###Creating-a-simple-plot)
      * [Simple plot with fit and errors](###Simple-plot-with-fit-and-errors)
      * [Using ODE.py to approximate the speed of falling objects with air resistance](###Using-ODE.py-to-approximate-the-speed-of-falling-objects-with-air-resistance) 
5.

    5.1 [Description](##Description)
   
    5.2 [How to use?](##How-to-use?)
   
    5.3 [Examples](##Examples)
6.

# How to install?

# Dependecies
* Numpy
* Pandas
* Matplotlib
* Scipy

# A Short overview
The package, currently, consists out of 4 main files.
* Aplot.py
* fileread.py
* ODE.py
* Extra.py

These files work indepent from each other, although Aplot.py accepts fileread objects, and can be imported on their own. **All** files depend on Helper.py, this file should be included when using files on their own. The package as a whole is subject to constant change, the goal is to keep the readme as updated as possible. Update logs are always more accurate.

The Aplot and fileread files do as their name suggest, currently Aplot.py has 2 different useable classes "Default" and "Histogram". A "Default" plot is a simple figure that can consist out of multiple attributes which are all, excluding the datapoints, optional. At the base a Default plot consists out of two arrays, x and y. It is possible to add a variety of attributes which are further specified in the Aplot.py documentation. The used class object needs to be called to construct and show/save the plot. The fileread.py file consists out of a single class "fileread" this class needs to be called to be able to convert the data file into a numpy.ndarray, pandas.df or dict object. It is possible to call multiple files although this function is not fully optimised. Next to these functions the ability to choose specific columns/rows/cells is also an option and can be specified in the input.

Extra.py is a collection of functions that are used for a varied range of applications. Functions included are:
* calc_err_DMM, TTI DMM 1604 error calculator
* trap_int, trapezium integral
* dep_trap, trap_int with dependant error
* derive, numerical derivatives
* Compress_array
* gauss_elim

Most of the functions have test cases which are placed in Extra_test.py and for Compress_array in Helper_test.py

ODE.py consists out of three approximation methods for numerically calculating ODE's
* Euler approximation
* Runga-Kutta 2nd order
* Runga-Kutta 4th order

All functions accept arrays/lists/tuples to make it easier to preform a parameter test or approximate with different stepsizes.

# Aplot.py
## Description
Aplot.py currently supports the plotting of a "Default" plot and a "Histogram". The default plot consists out x and y data arrays that are shown in a matplotlib.pyplot plot. It is possible to add a variety of options to these plots, like error bars custom labels fits and more. Currently the plots are limited to a single data set, if more than one data set is needed to be plotted the function can return fig and ax objects that contain the plot made by the Default class.

## How to use?
The easiest way to plot files it to simply create a Default object and passing 2 aruments through "x" and "y". This creates a simple plot with datapoints. Things can get more complicated by adding errorbars, labels and other plots. Errorbars are supported by adding arguments "x_err" and "y_err" when creating the object in the same way an "x_label" "y_label" and a "data_label" can be included. 

Adding fits is done by specifying a function with the "fx" paramter, the fit needs to follow where the first input argument needs to be x followed by constants these variables can have all names. To add a correct fit label specify the label with the "func_format" parameter which accepts a string constants are called with curly brackest per example "y = {0}x + {1}" would contain 2 constants A and B or {0} and {1} the order of constants is the same a defined in the "fx" parameter. Another way of adding fits is by specifying the degree of the polynomial to be fitted using the "degree" parameter a custom fit label can be added but standard ones are pre-included.

Plots can be added to each other to have multiple graphs in a single plot to do this add the "add_plot" parameter to the plot that needs to be **added** to a base plot and set this parameter to True. To show/save the plot run the base plot to which other plots have been added to. It is important to add them with the "+=" operator and not only use the "+" operator unless addition is specified as "plot_a = plot_a + plot_b".

## Examples
### Creating a simple plot
```python

from math import sin
from AdrianPack.Aplot import Default

# Create a list with 200 x values between -10 and 10
x = [val * 0.1 for val in list(range(-100, 100))]
# Calculate values of the function y = sin(x) between 0 and 20
y = [sin(val) for val in x]

# Make a Default object
plot = Default(x, y)

# Run the plot
plot()
```
Running this code results into the following plot

![alt text](https://github.com/AdrianvEik/AdrianPack/blob/main/Examples/Aplot/Plots/simple_plot.png?raw=true)

### Simple plot with fit and errors
```python

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

```
Running this code results into the following plot

![alt text](https://github.com/AdrianvEik/AdrianPack/blob/main/Examples/Aplot/Plots/simple_fit_img.png?raw=true)

### Using ODE.py to approximate the speed of falling objects with air resistance
The velocity of a falling object with air resistance can be described by the following equation:

![equation](https://latex.codecogs.com/png.image?\dpi{110}\bg{white}m&space;\dfrac{\mathrm{d}v}{\mathrm{d}t}&space;=&space;mg&space;-&space;k_2&space;v^2)

We can, after a small rewrtiting, add this ODE to python. Notice that the function is not dependant on t yet it is still included, **all** functions in ODE.py require an input function that takes to variables an x, y pair or in our case a v, t pair.
```python
def fx(v: float, t: float) -> float:
    return g - k_2/m * v**2
```

Which analytical solution is given by

![equation](https://latex.codecogs.com/png.image?\dpi{110}\bg{white}v(t)&space;=&space;\sqrt{\dfrac{mg}{k_2}}&space;\cdot&space;\mathrm{tanh}\left(&space;t&space;\cdot&space;\sqrt{\dfrac{k_2&space;g}{m}}&space;&plus;&space;C&space;\right))

To calculate this we write it in the following python code
```python
def theory(t: float) -> float:
    # The initial velocity is equal to 0 thus + C can be left out
    return np.sqrt(m*g/k_2) * np.tanh(t * np.sqrt(k_2*g/m))
```

Or more generally written as

![equation](https://latex.codecogs.com/png.image?\dpi{110}\bg{white}v(t)&space;=&space;A&space;\cdot&space;\tanh{(B&space;\cdot&space;t)&space;&plus;&space;C})

This is consequently also the function we will fit our approximations to! In python the function is formatted as follows, the first input is **always** the variable followed by our constants.

```python
def fit(x: float, a: float, b: float, c: float) -> float:
    return a * np.tanh(x * b + c)
```

It is possible to numerically determine the constants A, B and C and compare these to the analytical solution using ODE.py and Aplot.py. For this problem it is important to define certain constants. The following values and restrictions will be used for solving the problem
* *m* = 2 kg
* *k2* = 0,04 kg/m
* *g* = 9.81 m/s^2

The function *v*(*t*) is restricted by
* *v*(0) = 0

Let's define these restrictions, place the restrictrions above the defined functions as these are required by the functions.
```python
# Define constants
m = 0.5
k_2 = 0.04
g = 9.81

# Define the limiting factor
v_0 = 0
```

Now it is time to start making the plots. First calculate the data points for the Euler and Runga-Kutta approximations, these functions require 3 extra arguments besides the function fx we defined earlier. The 3 extra required arguments are a lower, upper bound and a timestep take 0, 5 and 0,5 for these respectively.
```python
runga_kutta4_data = runga_kutta_4(fx=fx, lower=0, upper=5, dt=0.5, x0=0)
euler_data = euler(fx=fx, lower=0, upper=5, dt=0.5, x0=0)
```

Using these data points we can make a graph for the 4th order Runga-Kutta, this will be the **base** graph in which the axis labels and plot limits are defined.
```python
runga_kutta4_plot = Default(x=runga_kutta4_data[0], y=runga_kutta4_data[1],
                            fx=fit, func_format="v = {0} $\cdot$ tanh({1} t + {2})",
                            x_label="Time t (s)", y_label="Velocity v ($\mathrm{ms^{-1}}$)",
                            data_label="4th order Runga-Kutta", save_as="air_res.png")
```
Note the usage of latex code in the labels, the current version (0.0.2) only supports simple latex code and because of the current formatter inputs like "$e^{4 \cdot x + 3}$" are **not** allowed.

All other graphs will take the axis labels and limits of the base graph thus we do not need to redefine these. The code for Euler's approximation is then
```python
euler_plot = Default(x=euler_data[0], y=euler_data[1],
                     fx=fit, func_format="v = {0} $\cdot$ tanh({1} t {2})",
                     add_mode=True, colour="C1",
                     data_label="Euler's approximation")
```

The theoretical line should only contain a line, to realise this use the "line_mode" attribute and set this to True. To make the line more smooth it is usefull to define a new time array with more points than the runga-kutta and euler arrays.

```python
t = np.linspace(0, 5, 1000)
theory_plot = Default(x=t, y=theory(t),
                      line_mode=True, add_mode=True, colour="gray",
                      connecting_line_label="Theory")
```

Now add the plots together and run the base plot
```python
runga_kutta4_plot += euler_plot
runga_kutta4_plot += theory_plot

runga_kutta4_plot()
```
We should get the following result

![alt text](https://github.com/AdrianvEik/AdrianPack/blob/main/Examples/Aplot/Plots/air_res.png?raw=true)


Full code:

```python

import numpy as np

from AdrianPack.Aplot import Default
from AdrianPack.ODE import runga_kutta_4, euler

# Define constants
m = 0.5
k_2 = 0.04
g = 9.81

# Define the limiting factor
v_0 = 0

# Define the ODE
def fx(v, t):
    return g - k_2/m * v**2

# Define the theoretical function
def theory(t):
    # The initial velocity is equal to 0 thus + C can be left out
    return np.sqrt(m*g/k_2) * np.tanh(t * np.sqrt(k_2*g/m))

# Define the fit function
def fit(x, a, b, c):
    return a * np.tanh(x * b + c)

# For the approximations a timestep dt is needed, in this example we will
# use 0,5 seconds

# We take the starting point at t=0 s and endpoint at t=5 s.
runga_kutta4_data = runga_kutta_4(fx=fx, lower=0, upper=5, dt=0.5, x0=0)
euler_data = euler(fx=fx, lower=0, upper=5, dt=0.5, x0=0)

# Initialising graphs for runga_kutta, euler and the theoretical line
runga_kutta4_plot = Default(x=runga_kutta4_data[0], y=runga_kutta4_data[1],
                            fx=fit, func_format="v = {0} $\cdot$ tanh({1} t + {2})",
                            x_label="Time t (s)", y_label="Velocity v ($\mathrm{ms^{-1}}$)",
                            data_label="4th order Runga-Kutta", save_as="air_res.png")

# Do not forget to enable "add_mode" when combining plots!
euler_plot = Default(x=euler_data[0], y=euler_data[1],
                     fx=fit, func_format="v = {0} $\cdot$ tanh({1} t {2})",
                     add_mode=True, colour="C1",
                     data_label="Euler's approximation")

t = np.linspace(0, 5, 1000)
theory_plot = Default(x=t, y=theory(t),
                      line_mode=True, add_mode=True, colour="gray",
                      connecting_line_label="Theory")

# Adding the plots together
runga_kutta4_plot += euler_plot
runga_kutta4_plot += theory_plot

runga_kutta4_plot()

```

# fileread.py
## Description
fileread.py Currently works with three file types, .txt, .csv and .xlsx sometimes using .xlsx files can cause issues when converting cells to floats this issue is most commonly resolved by converting to a .csv or .txt file. It is possible to read out multiple files too the columns/rows are then specified with a dictionary in the format {file_nr (int): columns (list, int), ...} all files need to be assigned columns to read it is not yet possible to read out all files without specifying a coll or row dictionary. Columns/rows can also be assigned custom labels to do so instead of specifying the position of the column/row use a tuple in the format (int, str) wherein the int and str are position and label respectively. This label will only be useable when the returned object type is a dictionary. The function has three return options "numpy", "dataframe" and "dictionary" the default option is the "dictionary" option.  
## How to use?

## Examples


# ODE.py

# Extra.py
The Extra.py file consists out of a varied range of functions. Each function is listed with a short explanation and example.
#### Functions
* calc_err_DMM

calculate the AC/DC error of a TTI DMM 1604. 
##### Example
```python
# Data array y consists out data points for a DC signal measured in milli volts.
y = np.array([0, 1.54, 2.01, 2.89, 3.71, 4.65], dtype=float)
error = cacl_err_DMM(unit="milli volt DC", val=y)
```

* trap_int

# Future changes/to-do list

# Known bugs/issues
### Aplot.py
* Custom fit labels with latex code in curly brackets crashes the code, this is because of an issue with the formatter used to format the constants
### csvread.py
* When reading xlsx files columns/rows that are specified with a tuple in the format "(int, str)" crashes the code.


