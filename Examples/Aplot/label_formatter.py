from __future__ import unicode_literals

import numpy as np

from AdrianPack.Aplot import Default
from AdrianPack.ODE import runga_kutta_4, euler

import matplotlib as mpl
mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.unicode']=True

def f(x, a, b, c):
    return a * np.cos(x * b) + a**(x + c)

x = np.linspace(-2, 2, int(1e1))
y = f(x, 0.314, 0.272, 0.161)

plot = Default(x, y, fx=f, x_label=r"\LaTeX", y_label=r"$ \beta $",
               func_format="$y_{{result}} = {0} \cdot \cos{{ (x_{{func}} \cdot {1}) }} + {0}^{{x_{{func}} + {2}}}$")
plot()

