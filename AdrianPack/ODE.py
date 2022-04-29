
import numpy as np
from csvread import csvread
from typing import Callable

test_input = csvread.test_inp

class Solve():
    def __init__(self, *args, **kwargs):
        """"
        Base function with definitions of self and shared functions
        """


        # TODO: UMPDATE ERROR THROWS
        if "fx" in kwargs:
            self.fx: Callable = kwargs["fx"]
        elif len(args) > 1:
            self.fx: Callable = args[0]
        else:
            raise AttributeError("Function requires fx")

        if "dt" in kwargs:
            self.dt: float = kwargs["dt"]
        elif len(args) > 2:
            self.dt: float = args[1]
        else:
            self.dt = 0.01

    def import_cuda(self):
        pass


class Euler(Solve):
    """"

    :param: fx
        ObjectType -> function

        Function fx which is the first derivative of function

    :param: dt
        ObjectType -> float

        Stepsize used when solving the ODE

    :param: lower
        ObjectType -> float

    :param: upper
        ObjectType -> float

    :param: mode
        ObjectType -> str

        Method to calculate the result either "single", "multi" or "gpu"
        uses single core processing, multi core processing or gpu accelerated
        processing.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if "mode" in kwargs:
            self.mode: str = kwargs["mode"]
        else:
            self.mode = "single"

        if "lower" in kwargs:
            self.lower: float = kwargs["lower"]
        else:
            raise AttributeError("Function requires lower")

        if "upper" in kwargs:
            self.upper: float = kwargs["upper"]
        else:
            raise AttributeError("Function requires upper")

        if "dt" in kwargs:
            self.dt: float = kwargs["dt"]
        else:
            raise AttributeError("Function requires dt")

        if "x0" in kwargs:
            self.x0: float = kwargs["x0"]
        else:
            self.x0 = 0

        self.N = round(np.abs((self.lower - self.upper)) / self.dt)

        self.tpoints = np.linspace(self.lower, self.upper, self.N)
        self.xpoints = np.zeros(self.N)

        self.xpoints[0] = self.x0

    def __call__(self):
        """
        return calculated values as (t, x)
        """
        getattr(self, self.mode)()
        return self.tpoints, self.xpoints

    def single(self):
        for i in range(len(self.tpoints) - 1):
            self.xpoints[i + 1] = self.xpoints[i] +\
                                  self.dt * self.fx(self.xpoints[i], self.tpoints[i])

    def multi(self):
        pass

    def gpu(self):
        pass


class Runga_kutta(Solve):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self):
        pass

    def singe(self):
        pass

    def multi(self):
        pass

    def gpu(self):
        # Add import function in super to import cuda packages
        # Finish later
        pass

if __name__ == "__main__":
    from Aplot import Default
    def f(x, t):
        return -x**3 + np.sin(t)

    data = Euler(fx=f, lower=0, upper=10, dt=0.001, x0=10)()
    print(data)
    Default(data[0], data[1], degree=1)()


