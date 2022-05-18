
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
    return np.sqrt(m*g/k_2) * np.tan(t * np.sqrt(m*g/k_2))

# Define the fit function
def fit(x, a, b, c):
    return a * np.tanh(x * b + c)

# For the approximations a timestep dt is needed, in this example we will
# use 0,1 seconds

# We take the starting point at t=0 s and endpoint at t=5 s.
runga_kutta4_data = runga_kutta_4(fx=fx, lower=0, upper=5, dt=0.1, x0=0)
euler_data = euler(fx=fx, lower=0, upper=5, dt=0.1, x0=0)

runga_kutta4_plot = Default(x=runga_kutta4_data[0], y=runga_kutta4_data[1],
                            fx=fit, func_format="v = {0} $\cdot$ tanh({1} t + {2})",
                            x_label="Time t (s)", y_label="Velocity v ($\mathrm{ms^{-1}}$)")
euler_plot = Default(x=runga_kutta4_data[0], y=runga_kutta4_data[1],
                     fx=fit, func_format="v = {0} $\cdot$ tanh({1} t + {2})",
                     x_label="Time t (s)", y_label="Velocity v ($\mathrm{ms^{-1}}$)",
                     add_mode=True)