import types
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from typing import Sized, Iterable, Union, Optional, Any, Type, Tuple, List
from matplotlib.ticker import EngFormatter

try:
    from TISTNplot import TNFormatter
except ImportError:
    # TODO: add a replacement for TNFormatter that replaces "." with ","
    #  notation but doesnt change decimal places
    TNFormatter = False

try:
    from csvread import csvread
except ImportError:
    from .csvread import csvread

# TODO: plot straight from files
# TODO: plot normal distrubtion over histogram
# TODO: plot bodeplots (maybe?)

test_inp = csvread.test_inp


class Aplot: # TODO: expand the docstring
    """"
    Plotting tool to plot files (txt, csv or xlsx), numpy arrays or
    pandas table with fit and error.

    Input arguments:
        Required:
        Either x and y or x or file is required for the input
        :param: x
        Array with values that correspond to the input values or controll values.
        :param: y
        Array with values that correspond to the output values f(x) or response
        values.
        :param: file

        Either degree or fx is required for the input
        :param: degree


        Optional:
        :param: mode

        :param: save_as

        kwargs:
        :param: xerr

        :param: yerr

        :param: x_label

        :param: y_label

        :param: fx

        :param: func_format
            Objecttype -> str.
            The format of the function shown in the figure legend, this is
            will be overwritten when used in combination with the degree
            parameter.
            EXAMPLE:
            fx Contains a function which returns
                f(x) = a * exp(-b*x)
            For a correct label func_form should be equivalent to
                r"${0} e^{-{1} \cdot x}$"
            the input doesnt necessarily need to be in a latex format.

    Usage:

    Examples:
    """

    def __init__(self, x: Union[tuple, list, np.ndarray] = None,
                 y: Union[tuple, list, np.ndarray] = None,
                 save_as: str = '', file: str = '', mode: str = 'default',
                 degree: Union[list, tuple, int] = None, **kwargs):

        self.save_as = save_as
        test_inp(self.save_as, str, "save path")
        self.mode = mode
        test_inp(self.mode, str, "mode")

        self.func = None
        self.degree = degree
        if degree is not None:
            test_inp(self.degree, (list, tuple, int, type(None)), "x values")
        elif 'fx' in kwargs:
            self.func = kwargs["fx"]
            test_inp(self.func, types.FunctionType, "f(x)")
        else:
            raise ValueError(
                "Input arguments must include degree or fx")

        self.response_var = "y"
        if "response_var" in kwargs:
            self.response_var = kwargs["response_var"]
            test_inp(self.response_var, str, "response variable")

        self.control_var = "x"
        if "control_var" in kwargs:
            self.control_var = kwargs["control_var"]
            test_inp(self.control_var, str, "control variable")

        self.func_format = ''
        if 'func_format' in kwargs:
            self.func_format = kwargs['func_format']

        self.kwargs = kwargs

        # X AND Y ARRAYS
        if x is not None and y is not None:
            test_inp(x, (list, tuple, np.ndarray), "x values")
            test_inp(y, (list, tuple, np.ndarray), "y values")
            self.x = np.asarray(x, dtype=np.float32)
            self.y = np.asarray(y, dtype=np.float32)

            try:
                assert self.x.shape == self.y.shape
            except AssertionError:
                raise ValueError("Arrays x and y should have same size but are"
                                 " size %s and %s" % (self.x.size, self.y.size))

            try:
                assert self.x.ndim == self.y.ndim == 1
            except AssertionError:
                raise NotImplementedError("Multi dimensional plotting is not"
                                          " supported.")

        elif x is not None:
            test_inp(x, (list, tuple, np.ndarray), "x values")
            self.x = x
            self.mode = 'hist'
            try:
                assert self.x.ndim == 1
            except AssertionError:
                # TODO: Rework this one to be more compact
                # TODO: Rework to utilize Format function used by file_read
                if self.x.ndim == 2:
                    self.mode = 'default'
                    if x.shape[1] == 2:
                        self.x = x[:, 0]
                        self.y = x[:, 1]
                        print('\x1b[33m' +
                              'WARNING: Input array x has 2 dimensions and %s'
                              ' columns, the first and second column has been used as x and'
                              'y respectfully.' % str(x.shape[1])
                              + '\x1b[0m' )
                    elif x.shape[1] == 3:
                        self.x = x[:, 0]
                        self.y = x[:, 1]
                        self.y_err = x[:, 2]
                        print('\x1b[33m' +
                              'WARNING: Input array x has 2 dimensions and %s columns,'
                              ' the matrix has been formatted as follows'
                              ' "x y yerr"' % str(x.shape[1])
                              + '\x1b[0m' )
                    elif x.shape[1] == 4:
                        self.x = x[:, 0]
                        self.y = x[:, 1]
                        self.x_err = x[:, 2]
                        self.y_err = x[:, 3]
                        print('\x1b[33m' +
                              'WARNING: Input array x has 2 dimensions and %s columns,'
                              ' the matrix has been formatted as follows'
                              ' "x y xerr yerr"' % str(x.shape[1])
                              + '\x1b[0m' )
                    else:
                        raise NotImplementedError(
                            "Multi dimensional plotting is not supported.")
                else:
                    raise NotImplementedError("Multi dimensional plotting is not"
                                          " supported.")

        elif file is not None:
            # TODO: self.read_file() func returning 4 arrays
            #  (or changing the self. vals)
            raise NotImplementedError("Function has not been implemented yet")
        else:
            raise ValueError("Input argument must include x or x and y or file")

        # ERROR ARRAYS
        if 'x_err' in kwargs:
            self.x_err = kwargs["x_err"]
            test_inp(self.x_err, (int, tuple, np.ndarray, list, float), "x error")
            try:
                if isinstance(self.x_err, (int, float)):
                    self.x_err = np.full(self.x.size, self.x_err)
                assert self.x_err.size == self.x.size
            except AssertionError:
                raise IndexError("The error")
        else:
            self.x_err = []

        if 'y_err' in kwargs:
            self.y_err = kwargs["y_err"]
            test_inp(self.y_err, (int, tuple, np.ndarray, list, float), "y error")
            try:
                if isinstance(self.y_err, (int, float)):
                    self.y_err = np.full(self.y.size, self.y_err)
                assert self.y_err.size == self.y.size
            except AssertionError:
                raise IndexError("The error")
        else:
            self.y_err = []

    def __call__(self, *args, **kwargs) -> Tuple[plt.figure, plt.axes]:
        """"
        OPTIONAL:
            :param: save_path
                ObjectType -> str
                Path to save the file to, default is the directory of the
                .py file.
            :param: return_object
                ObjectType -> Bool
                Default false, if true returns only the fig, ax object in a
                tuple.
        :returns
            Tuple consisting of fig, ax objects

        """
        # TODO: implement save_path
        if "save_path" in kwargs:
            test_inp(kwargs["save_path"], str, "save path")
            save_path = kwargs['save_path']
        else:
            try:
                test_inp(args[0], str, "save path")
                save_path = args[0]
            except IndexError:
                save_path = ''

        self.return_object = False
        if "return_object" in kwargs:
            test_inp(kwargs["return_object"], bool, "return object")
            self.return_object = kwargs["return_object"]
        else:
            # Maybe restructure this? Idk if I want to implement this option
            # just a waste of time.
            try:
                test_inp(args[1], bool, "return object")
                self.return_object = args[1]
            except IndexError:
                try:
                    if save_path == '':
                        test_inp(args[0], bool,
                                 "return object")
                        self.return_object = args[0]
                except IndexError:
                    pass

        if self.mode == 'default':
            self.default_plot()

        # TODO: Implement multi fig support
        elif self.mode == 'multi':
            raise NotImplementedError("Multi plot mode not supported yet")

        elif ["side", "under"] in self.mode.split(' '):
            if self.mode.split(' ')[1].lower() == "side":
                raise NotImplementedError("Multi side mode not supported yet")
            else:
                raise NotImplementedError("Multi under mode not supported yet")

        elif self.mode == 'norm_dist':
            raise NotImplementedError("Norm_dist mode not supported yet")

        return self.fig, self.ax

    def default_plot(self, show_error: bool = None) -> None:
        """
        Plot a 2D data set with errors in both x and y axes. The data
        will be fitted according to the input arguments in __innit__.

        Requires a Aplot object to plot with x, y or file input.

        OPTIONAL
        :param: show_error
            ObjectType -> boolean
            Default, True when true prints out the error in the
            coefficients. When changed from default overwrites the print_error
            statement in __innit__.

        EXAMPLE
        plot and show fig
        x, y = data
        plot_obj = Aplot(x, y, degree=1) # Linear fit to x and y data
        plot_obj.default_plot()

        plot and save fig as plot.png
        x, y = data
        plot_obj = Aplot(x, y, degree=1, save_as='plot.png') # Linear fit to x and y data
        plot_obj.default_plot()

        :return: None
        """

        self.fig, self.ax = plt.subplots()
        self.fit()

        # DATA PLOTTING
        # TODO: add these extra kwargs to the docs
        if "data_label" in self.kwargs:
            test_inp(self.kwargs["data_label"], str, "data label")
            label = self.kwargs["data_label"]
        else:
            label = "Data"

        if "data_color" in self.kwargs:
            test_inp(self.kwargs["data_color"], str, "data color")
            color = self.kwargs["data_color"]
        else:
            color = "C0"

        if len(self.y_err) == 0 and len(self.x_err) == 0:
            self.ax.scatter(self.x, self.y, label=label, color=color)
        elif len(self.y_err) == 0 or len(self.x_err) == 0:
            if len(self.y_err) == 0:
                self.ax.errorbar(self.x, self.y, xerr=self.x_err,
                                 label=label, fmt=color+'o', linestyle='',
                                 capsize=4)
            else:
                self.ax.errorbar(self.x, self.y, yerr=self.y_err,
                                 label=label, fmt=color+'o', linestyle='',
                                 capsize=4)
        else:
            self.ax.errorbar()

        # FIT PLOTTING
        # If both the show_error atr in this function and the show error in the
        # __innit__ are turned off the error is printed as mentioned in the

        if show_error:
            print(self.fit_errors)

        n_points = 1000
        if "custom_fit_spacing" in self.kwargs:
            test_inp(self.kwargs["custom_fit_spacing"], int, "fit array size")
            n_points = self.kwargs["custom_fit_spacing"]
        fit_x = np.linspace(min(self.x), max(self.x), n_points)

        fit_pr = 3
        if "fit_precision" in self.kwargs:
            test_inp(self.kwargs["fit_precision"], int, "fit precision")
            fit_pr = self.kwargs["fit_precision"]

        str_fit_coeffs = [str(np.around(c, fit_pr)).replace(".", ",") for c in self.fit_coeffs]
        if self.func is not None:
            self.ax.plot(fit_x, self.func(fit_x, *self.fit_coeffs), linestyle="--",
                         label=(lambda _: self.func_format.format(*str_fit_coeffs)
                         if self.func_format != "" else "Fit")(None))
        elif self.degree is not None:
            self.ax.plot(fit_x,
                         sum([fit_x ** (c) * self.fit_coeffs[abs(c - self.degree)]
                              for c in range(self.degree + 1).__reversed__()]),
                         linestyle="--",
                         label=("Fit with function %s = " % self.response_var +
                                self.degree_dict[self.degree].format(*str_fit_coeffs)))
        else:
            raise ValueError("Either 'func' or 'degree' should be in the input"
                             "params.")

        self.ax.plot()


        #TODO: add these to the docs
        y_label = ''
        if "y_label" in self.kwargs:
            test_inp(self.kwargs["y_label"], str, "y_label")
            y_label = self.kwargs["y_label"]

        x_label = ''
        if "x_label" in self.kwargs:
            test_inp(self.kwargs["x_label"], str, "x_label")
            x_label = self.kwargs["x_label"]

        grid = True
        if "grid" in self.kwargs:
            test_inp(self.kwargs["grid"], bool, "grid")
            grid = self.kwargs["grid"]

        self.single_form(x_label, y_label, grid=grid)

        if not self.return_object:
            (lambda save_as:
             plt.show() if save_as == '' else plt.savefig(save_as,
                                                          bbox_inches='tight')
             )(self.save_as)

        return None

    def fit(self) -> None:
        """

        :return:
        """

        self.degree_dict = {
            0: '{0}',
            1: '${0}%s + {1}$' % self.control_var,
            2: r'${0}%s^2 + {1}%s + {2}$' % (self.control_var, self.control_var),
            3: r'${0}%s^3 + {1}%s^2 + {2}%s + {3}$' % tuple([self.control_var for _ in range(3)])
        }

        if self.degree is not None:
            self.label_format = self.degree_dict[self.degree]
            if self.degree == 1:
                fit = scipy.stats.linregress(self.x, self.y)
                self.fit_coeffs = np.array([fit.slope, fit.stderr], dtype=np.float32)
                self.fit_errors = np.array([fit.intercept, fit.intercept_stderr], dtype=np.float32)
            elif isinstance(self.degree, int):
                fit = np.polyfit(self.x, self.y, deg=self.degree, cov=True)
                self.fit_coeffs = fit[0]
                self.fit_errors = np.sqrt(np.diag(fit[1]))
        elif self.func is not None:
            fit, pcov = curve_fit(self.func, self.x, self.y)
            self.fit_coeffs = fit
            self.fit_errors = np.sqrt(np.diag(pcov))
        return None

    def single_form(self, x_label: str, y_label: str, grid: bool = True, **kwargs)\
            -> Union[Tuple[plt.figure, plt.axes], None]:
        """"
        Format a figure with 1 x-axis and y-axis.

        REQUIRED:
        :param: x_label
            ObjectType -> str
            Label placed on the x-axis, usually uses input from __init__ kwargs
        :param: y_label
            ObjectType -> str
            Label placed on the y-axis, usually uses input from __init__ kwargs

        OPTIONAL:
        :param: grid
            ObjectType -> bool
            True to show grid and False to turn the grid off, default True.
            Takes input from __innit__ kwargs.

        KWARGS:
        :param: fig_ax
            ObjectType -> Tuple
            The tuple should contain an fig, ax pair with fig and ax being
                fig:
                ObjectType -> matplotlib.pyplot.fig object
                Use this input to apply formatting on input fig
                ax:
                ObjectType -> matplotlib.pyplot.Axes.ax object
                Use this input to apply formatting on input ax

            EXAMPLE
                single_form("x_label", "y_label", fig_ax=(plt.subplots()))

         :returns:
            ObjectType -> Union[Tuple[matplotlib.pyplot.fig, matplotlib.pyplot.Axes.ax], NoneType]
            When fig_ax input is part of the input this function will return
            the fig, ax pair
            if not the return is of type NoneType

        EXAMPLES:
            # Initiate a fig, ax pair
            fig, ax = plt.subplots()
            # Plot the data
            ax.plot(x_data, y_data)
            # format the plot
            Aplot().single_form("x_label", "y_label", (fig, ax))
            # Show the formatted plot
            plt.show()

            #TODO: Add example with Aplot object

        NOTES ON PARAMS  x_label, y_label and grid:
            Direct input in this function will overwrite __innit__ inputs.
        """

        if "fig_ax" in kwargs:
            # TODO: test these tests and test the usability of the object
            test_inp(kwargs["fig_ax"], tuple, "fig ax pair")
            test_inp(kwargs["fig_ax"][0], type(plt.figure), "fig")
            test_inp(kwargs["fig_ax"][1], type(plt.axes.ax), "ax")

            self.fig = kwargs["fig_ax"][0]
            self.ax = kwargs["fig_ax"][1]

        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)

        if TNFormatter is not False:
            self.ax.xaxis.set_major_formatter(TNFormatter(3))
            self.ax.yaxis.set_major_formatter(TNFormatter(3))

        if grid:
            plt.grid()

        plt.legend(loc='lower right')
        plt.tight_layout()
        if "fig_ax" in kwargs:
            return self.fig, self.ax
        else:
            return None

    def multi_plot(self):
        """

        :return:
        """
        self.fig, self.ax = plt.subplots() # more plots in subplots
        return None

    def norm_dist_plot(self):
        """

        :return:
        """
        self.fig, self.ax = plt.subplots()
        return None


    def file_ceck(self):
        return None


if __name__ == "__main__":
    import time
    t_start = time.time()
    def f(x: float, a: float, b: float) -> float:
        return a * x**2 + b*x

    points = 25
    x = np.linspace(-5, 5, points)
    noise = np.random.randint(2, 5, points)
    Aplot(x=x, y=np.array([i**3 * 4.32 + 9.123*i for i in x]) + noise, degree=3, fit_precision=2,
          y_err=10, control_var="z")()
    print("t: ", time.time() - t_start)