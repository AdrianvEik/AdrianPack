import types
import time

import numpy.random
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.optimize import curve_fit
from typing import Sized, Iterable, Union, Optional, Any, Type, Tuple, List, \
    Dict, Callable
from inspect import signature

try:
    from TN_code.plotten.TISTNplot import TNFormatter
except ImportError:
    try:
        from .TN_code.plotten.TISTNplot import TNFormatter
    except ImportError:
        TNFormatter = False

try:
    from Helper import test_inp, compress_ind
except ImportError:
    from .Helper import test_inp, compress_ind


# TODO: plot straight from files
# TODO: plot normal distrubtion over histogram
# TODO: plot bodeplots (maybe?)

# TODO: Recheck all examples and upload to git with an example.py file
# TODO: restructure to use base class from which to inherit standardized methods

class Base:
    """
    Base class of Aplot object
    """

    def __init__(self, *args, **kwargs):
        """"
        :param: x
            ObjectType -> numpy.ndarry
            Array with values that correspond to the input values or control
             values.
        :param: y
            ObjectType -> numpy.ndarry
            Array with values that correspond to the output values f(x) or
             response values.
        """
        self.decimal_comma = True
        self.TNFormatter = TNFormatter
        if 'decimal_comma' in kwargs:
            test_inp(kwargs["decimal_comma"], bool, "decimal_comma")
            self.decimal_comma = kwargs['decimal_comma']
            # If the decimal comma is set to False the TNformatter should
            # if available be deactivated.
            if not self.decimal_comma:
                self.TNFormatter = self.decimal_comma

        add_mode = False
        if "add_mode" in kwargs:
            add_mode = kwargs["add_mode"]
            test_inp(add_mode, bool, "add_mode")

        if not add_mode:
            plt.clf()
            plt.close()
            self.fig, self.ax = plt.subplots()

        self.response_var = "y"
        if "response_var" in kwargs:
            self.response_var = kwargs["response_var"]
            test_inp(self.response_var, str, "response variable")

        self.control_var = "x"
        if "control_var" in kwargs:
            self.control_var = kwargs["control_var"]
            test_inp(self.control_var, str, "control variable")

        self.label = "Data"
        if "data_label" in kwargs:
            test_inp(kwargs["data_label"], str, "data label")
            self.label = kwargs["data_label"]

        self.x_lim = False
        if "x_lim" in kwargs:
            self.x_lim = kwargs["x_lim"]
            test_inp(self.x_lim, (list, tuple, np.ndarray), "x lim")

            try:
                assert len(self.x_lim) == 2
            except AssertionError:
                raise IndexError(
                    "Xlim should only contain xmin and xmax but the"
                    "length of xlim does not equal 2.")

            test_inp(self.x_lim[0], (float, int), "xmin")
            test_inp(self.x_lim[1], (float, int), "xmax")

        self.y_lim = False
        if "y_lim" in kwargs:
            self.y_lim = kwargs["y_lim"]
            test_inp(self.y_lim, (list, tuple, np.ndarray), "x lim")

            try:
                assert len(self.y_lim) == 2
            except AssertionError:
                raise IndexError(
                    "Ylim should only contain ymin and ymax but the"
                    "length of ylim does not equal 2.")

            test_inp(self.y_lim[0], (float, int), "ymin")
            test_inp(self.y_lim[1], (float, int), "ymax")

        self.colour = "C0"
        if "colour" in kwargs:
            test_inp(kwargs["colour"], str, "colour")
            self.colour = kwargs["colour"]

        self.marker = "o"
        if "marker" in kwargs:
            test_inp(kwargs["marker"], str, "marker")
            self.marker = kwargs["marker"]

        self.linestyle = "-"
        if "linestyle" in kwargs:
            test_inp(kwargs["linestyle"], str, "linestyle")
            self.linestyle = kwargs["linestyle"]

        self.capsize = 4
        if "capsize" in kwargs:
            test_inp(kwargs["capsize"], (int, float, str), "capsize")
            self.capsize = kwargs["capsize"]
        self.capsize = int(self.capsize)

        self.linewidth = 1
        if "linewidth" in kwargs:
            test_inp(kwargs["linewidth"], (int, float, str), "linewidth")
            self.linewidth = kwargs["linewidth"]

        # Read x and y values
        self.file_format = kwargs[
            "file_format"] if "file_format" in kwargs else ["x", "y", "x_err",
                                                            "y_err"]
        test_inp(self.file_format, list, "File format")

        # Initialise x and y as None
        self.x, self.y = None, None

        if len(args) >= 2 and args[1] is not None:
            self.x = args[0]
            self.y = args[1]
        elif len(args) == 1:
            self.x = args[0]
            self.y = None
        elif "file" in kwargs:
            self.x = kwargs["file"]
        elif "x" in kwargs:
            self.x = kwargs["x"]

        if self.x.__class__.__name__ in ["ndarray", "list", "tuple"]:
            # either matrix or normal x and y
            self.x = np.asarray(self.x, dtype=np.float32)
            if self.y is not None:
                self.y = np.asarray(self.y, dtype=np.float32)
            # 1 Dimensional array
            if self.x.ndim == 1:
                # if self.y is None:
                #     self.y = self.x
                #     self.x = range(len(self.x))
                pass

            # Matrix
            else:
                self.array_parse(self.x)

        # File
        elif self.x.__class__.__name__ == "str" or self.x.__class__.__name__ == "Fileread":
            # File
            try:
                from .Fileread import Fileread
            except ImportError:
                try:
                    from AdrianPack.Fileread import Fileread
                except ImportError:
                    raise ImportError("Fileread not available for use, include"
                                      " Fileread in the script folder to read files.")

            # If the type is a string turn it in to a Fileread object else use
            # the Fileread object
            data_obj = Fileread(path=self.x, **kwargs) if type(
                self.x) is str else self.x
            data = data_obj()

            for key in data.keys():
                if key in "x":
                    self.x = data[key]
                if key in "y":
                    self.y = data[key]
                if key in "x_err":
                    self.x_err = data[key]
                if key in "y_err":
                    self.y_err = data[key]

            if type(self.x) in [None, str] or self.y is None:
                data_obj.output = "numpy"
                self.array_parse(data_obj())

        # Only y input should also be possible
        if "y" in kwargs and self.y is None:
            self.y = kwargs["y"]

            if self.y.__class__.__name__ in ["list", "ndarray", "tuple"]:
                self.y = np.asarray(self.y, dtype=np.float32)
                # 1 Dimensional array
                if self.y.ndim == 1:
                    self.x = range(len(self.y))
                else:
                    raise Exception("Error") # TODO: Fix this error pls

        self.plots = []
        self.kwargs = kwargs

    def __add__(self, other):
        if other.__class__.__name__ == "Default":
            self.add_default_plot(other)
        elif other.__class__.__name__ == "Histogram":
            raise NotImplementedError

        self.plots.append(other)
        return self

    def __len__(self):
        return len(self.plots) + 1

    def __repr__(self):
        return "{0}, {1}".format(self.x, self.y)

    def array_parse(self, data):
        """
        Parse an array to x, y and optional x_err and y_err attributes from
        given format ('file_format').
        :param data: Array to be parsed
        :return: None
        """
        self.x = data[:, self.file_format.index("x")]
        self.y = data[:, self.file_format.index("y")]

        if data.shape[1] == 3:
            if "x_err" in self.file_format:
                self.x_err = data[:, self.file_format.index("x_err")]
            else:
                self.y_err = data[:, self.file_format.index("y_err")]
        elif data.shape[1] > 4:
            self.x_err = data[:, self.file_format.index("x_err")]
            self.y_err = data[:, self.file_format.index("y_err")]
        return None

    def single_form(self, x_label: str, y_label: str, grid: bool = True,
                    **kwargs) \
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

        :param: x_lim
            ObjectType -> Union[Tuple, List, np.ndarray]
            The limits of the horizontal axes, contains a xmin (xlim[0]) and
            xmax (xlim[1]) pair. Both xmin and xmax should be of type
            float, int or numpy float.

            EXAMPLE
                single_form("x_label", "y_label", xlim=[0, 2.4])

        :param: y_lim
            ObjectType -> Union[Tuple, List, np.ndarray]
            The limits of the vertical axes, contains a ymin (ylim[0]) and
            ymax (ylim[1]) pair. Both ymin and ymax should be of type
            float, int or numpy float.

            EXAMPLE
                single_form("x_label", "y_label", ylim=[-15.4, 6.9])

         :returns:
            ObjectType -> Union[Tuple[matplotlib.pyplot.fig, matplotlib.pyplot.Axes.ax], NoneType]
            When fig_ax input is part of the input this function will return
            the fig, ax pair
            if not the return is of type NoneType

        EXAMPLE:
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
            test_inp(kwargs["fig_ax"], (tuple, list), "fig ax pair")
            test_inp(kwargs["fig_ax"][0], plt.Figure, "fig")
            test_inp(kwargs["fig_ax"][1], plt.Axes, "ax")

            fig = kwargs["fig_ax"][0]
            ax = kwargs["fig_ax"][1]
        else:
            fig = self.fig
            ax = self.ax

        if "x_lim" in self.kwargs:
            ax.set_xlim(self.x_lim)
        elif "x_lim" in kwargs:
            x_lim = kwargs["x_lim"]

            test_inp(x_lim, (list, tuple, np.ndarray), "x lim")

            try:
                assert len(x_lim) == 2
            except AssertionError:
                raise IndexError(
                    "Xlim should only contain xmin and xmax but the"
                    "length of xlim does not equal 2.")

            test_inp(x_lim[0], (float, int), "xmin")
            test_inp(x_lim[1], (float, int), "xmax")

            ax.set_xlim(x_lim)

        if "y_lim" in self.kwargs:
            ax.set_ylim(self.y_lim)
        elif "y_lim" in kwargs:
            y_lim = kwargs["y_lim"]

            test_inp(y_lim, (list, tuple, np.ndarray), "y lim")

            try:
                assert len(y_lim) == 2
            except AssertionError:
                raise IndexError(
                    "Ylim should only contain ymin and ymax but the"
                    "length of ylim does not equal 2.")

            test_inp(y_lim[0], (float, int), "ymin")
            test_inp(y_lim[1], (float, int), "ymax")

            ax.set_ylim(y_lim)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        if self.TNFormatter:
            x_pr = 3
            if "x_precision" in self.kwargs:
                test_inp(self.kwargs["x_precision"], int, "x_precision",
                         True)
                x_pr = self.kwargs["x_precision"]
            y_pr = 3
            if "y_precision" in self.kwargs:
                test_inp(self.kwargs["y_precision"], int, "y_precision",
                         True)
                y_pr = self.kwargs["y_precision"]

            ax.xaxis.set_major_formatter(TNFormatter(x_pr))
            ax.yaxis.set_major_formatter(TNFormatter(y_pr))

        if grid:
            ax.grid()

        if "legend_loc" in self.kwargs:
            test_inp(self.kwargs["legend_loc"], str, "legenc loc")
            ax.legend(loc=self.kwargs["legend_loc"])
        elif "legend_loc" in kwargs:
            test_inp(self.kwargs["legend_loc"], str, "legenc loc")
            ax.legend(loc=kwargs["legend_loc"])
        else:
            ax.legend(loc='lower right')

        if "fig_ax" in kwargs:
            return fig, ax
        else:
            return None

    def add_default_plot(self, other):
        """
        Render another Aplot.Default graph on a Aplot.Base object
        :param other:
            Other object to be added
        """
        other.fit()

        if "colour" in other.kwargs:
            test_inp(other.kwargs["colour"], str, "colour")
            color = other.kwargs["colour"]
        else:
            color = "C1"

        if other.scatter:
            if len(other.y_err) == 0 and len(other.x_err) == 0:
                self.ax.plot(other.x, other.y,
                        label=other.label, marker=other.marker, c=other.colour,
                        linestyle=other.linestyle,
                        ms=other.capsize,
                        linewidth=other.linewidth
                )
            elif len(other.y_err) == 0 or len(other.x_err) == 0:
                if len(other.y_err) == 0:
                    self.ax.errorbar(other.x, other.y, xerr=other.x_err,
                                     label=other.label,
                                     fmt=color + other.marker,
                                     linestyle=other.linestyle,
                                     capsize=other.capsize)
                else:
                    self.ax.errorbar(other.x, other.y, yerr=other.y_err,
                                     label=other.label,
                                     fmt=color + other.marker,
                                     linestyle=other.linestyle,
                                     capsize=other.capsize)
            else:
                self.ax.errorbar(other.x, other.y, xerr=other.x_err,
                                 yerr=other.y_err,
                                 label=other.label, fmt=color + other.marker,
                                 linestyle=other.linestyle,
                                 capsize=other.capsize)

        if other.sigma_uncertainty:
            self.sigma_uncertainty(base=other)

        fit_x = np.linspace(min(other.x), max(other.x), other.n_points)

        fit_pr = 3
        if "fit_precision" in other.kwargs:
            test_inp(other.kwargs["fit_precision"], int, "fit precision")
            fit_pr = other.kwargs["fit_precision"]

        if other.degree is not None or other.func is not None:
            if self.decimal_comma:
                str_fit_coeffs = [str(np.around(c, fit_pr)).replace(".", ",")
                                  for c
                                  in
                                  other.fit_coeffs]
            else:
                str_fit_coeffs = [str(np.around(c, fit_pr)) for c in
                                  other.fit_coeffs]

        if other.func is not None:
            self.ax.plot(fit_x, other.func(fit_x, *other.fit_coeffs),
                         linestyle=other.fit_line_style,
                         c=color if not other.fit_colour else other.fit_colour,
                         label=(
                             lambda _: other.func_format.format(
                                 *str_fit_coeffs)
                             if other.func_format != "" else "Fit")(None))
        elif other.degree is not None:
            if other.func_format != '':
                self.ax.plot(fit_x,
                             sum([fit_x ** (c) * other.fit_coeffs[
                                 abs(c - other.degree)]
                                  for c in
                                  range(other.degree + 1).__reversed__()]),
                             linestyle=other.fit_line_style,
                             c=color if not other.fit_colour else other.fit_colour,
                             label=(other.func_format.format(*str_fit_coeffs)))
            else:
                self.ax.plot(fit_x,
                             sum([fit_x ** (c) * other.fit_coeffs[
                                 abs(c - other.degree)]
                                  for c in
                                  range(other.degree + 1).__reversed__()]),
                             linestyle=other.fit_line_style,
                             c=color if not other.fit_colour else other.fit_colour,
                             label=(
                                     "Fit with function %s = " % other.response_var +
                                     other.degree_dict[other.degree].format(
                                         *str_fit_coeffs)))
        return None


class Default(Base):  # TODO: expand the docstring #TODO x and y in args.
    """"
    Plotting tool to plot files (txt, csv or xlsx), numpy arrays or
    pandas table with fit and error.

    Input arguments:
        :param: file
            # TODO: add os library based Path object support.

            ObjectType -> str
            Path to the file in str.
            EXAMPLE:
                plot_obj = Aplot(x, y, )

        :param: degree
            ObjectType -> int
            N-th degree polynomial that correlates with input data. Currently
            only 1st to 3rd degree polynomials are supported. For higher degree
            polynomials (or any other function) use your own function as input.
            This can be done with the fx and func_format params.

            Example
                plot_obj = Aplot(x, y, degree=1) # Linear fit

        :param: save_as
            # TODO: Implement save_as

        :param: x_err
            ObjectType -> Union[np.ndarray, float, int]
            Array with error values or single error value that is applied
            to all values for x.

            EXAMPLE
            # For an array of error values
            plot_obj = Aplot(x, y, x_err=x_err)
            plot_obj()

            # For a single error value
            plot_obj = Aplot(x, y, x_err=0.1)
            plot_obj()

        :param: y_err
            ObjectType -> Union[np.ndarray, float, int]
            Array with error values or single error value that is applied
            to all values for y.

        :param: x_label
            ObjectType -> str
            String with label placed on the horizontal axis, can contain
            latex.

            EXAMPLE
            # To include latex in the label the string (or part that contains
            # latex) should start and end with "$"

            label = r"$x$-axis with $\latex$"
            plot_obj = Aplot(x, y, x_label=label)
            # Show plot
            plot_obj()


        :param: y_label
            ObjectType -> str
            String with label placed on the horizontal axis, can contain
            latex.

        :param: fx
            ObjectType -> function
            The function to fit the data to.

            Typical fx function example
            def f(x: float, a: float, b: float) -> float:
                '''
                Input
                    x: float
                        Variabale input
                    a: float
                        1st fit paramater
                    b: float
                        2nd fit parameter
                Returns
                    rtype: float
                    Output of function f(x).
                '''
                return a*x + b

            The first input argument of the function needs to be x and this
            argument must take a float-like input and return a float like object.
            Other function parameters will be the fit parameters, these can
            have custom names. Fit parameters will return in the same order as
            their input, this order is also used in the formatter.

            EXAMPLES
                # Function that is equal to y = a exp(-bx)
                def f(x: float, a: float, b: float) -> float:
                    return a * np.exp(-1 * b * x)

                format_str = r"${0} e^{-{1} \cdot x}$" # Str containing the format
                # Initialize the plot with data and function/format pair
                plot_obj = Aplot(x, y, fx = f)
                # Show the plot
                plot_obj()

                # Function that is equal to y = z * x^2 + h * x^(-4/7)
                def g(x: float, h: float, z: float) -> float
                    return z * x**2 + h * x**(4/7)
                # Initialize the plot with data and function/format pair
                plot_obj = Aplot(x, y, fx = g)
                # Show the plot
                plot_obj()

        :param: func_format
            Objecttype -> str.
            The format of the function shown in the figure legend, this is
            will overwrite the default label when used in combination with
            the degree parameter.

            EXAMPLE:
                fx Contains a function which returns
                    f(x) = a * exp(-b*x)
                For a correct label func_form should be equivalent to
                    r"${0} e^{-{1} \cdot x}$"
                the input doesnt necessarily need to be in a latex format.

            CODE EXAMPLE:
                # Function that is equal to y = a exp(-bx)
                def f(x: float, a: float, b: float) -> float:
                    return a * np.exp(-1 * b * x)

                format_str = r"${0} e^{-{1} \cdot x}$" # Str containing the format
                # Initialize the plot with data and function/format pair
                plot_obj = Aplot(x, y, fx = f, func_format = format_str)
                # Show the plot
                plot_obj()

        :param: data_label
            ObjectType -> str
            Label of the data in the legend. Can include latex.

        :param: colour
            ObjectType -> str
            Color of the data in one of the matplotlib accepted colors.
            See matplotlib documentation for accepted colors.

        :param: marker_fmt
            ObjectType -> str
            Marker of data points.

        :param: custom_fit_spacing
            ObjectType -> int
            Length of the fitted data array. Min and Max are determined
            by the min and max of the provided x data. Default is 1000 points.

        :param: fit_precision


        :param: grid


        :param: response_var


        :param: control_var

        :param: add_mode
            Default False when true doesnt initiate plt.subplots to make it
            possible to add these graphs to another Aplot.Default plot.

        :param: connecting_line
            Default False, when True adds a connecting line in between
            data points

        :param: line_only
            Default False, when True ONLY draws a connecting line between
            data points.

        :param: connecting_line_label
            Default "Connection" object type str, is used as a label for the
            connecting line only applicable when the connecting_line or line_mode
            parameters are set to True.

        :param: decimal_comma
            Default True, if set to false

        :param: file

        :param: file_format
            The format of the columns within the files, default is set to:
             ["x", "y", "x_err", "y_err"]
            Indicating that the first column is the x column, the second y, etc.

            If the headers within the file are equal to the names in the file_format
            the reader will detect these columns within the file note that
            the headers of the file will need to be "x", "y", "x_err", "y_err"
            for this to work.

        :param: sigma_uncertainty
            Either false or an Integer,
            Adds an n-sigma uncertainty 'field' to the plot.

        :param: uncertainty colour
            Colour of the uncertainty lines

    Usage:

    Examples:
    """

    def __init__(self, x: Union[tuple, list, np.ndarray, str] = None,
                 y: Union[tuple, list, np.ndarray] = None, save_as: str = '',
                 degree: Union[list, tuple, int] = None,
                 *args, **kwargs):
        super().__init__(x, y, *args, **kwargs)
        self.save_as = save_as
        test_inp(self.save_as, str, "save as")

        self.func = None
        self.degree = degree
        if degree is not None:
            test_inp(self.degree, (list, tuple, int, type(None)), "x values")
        elif 'fx' in kwargs:
            self.func = kwargs["fx"]
            test_inp(self.func, types.FunctionType, "f(x)")

        self.fit_coeffs = 0
        self.fit_errors = 0

        self.fit()

        self.fit_colour = False
        if 'fit_colour' in kwargs:
            test_inp(kwargs['fit_colour'], str, "fit_colour")
            self.func_format = kwargs['fit_colour']

        self.fit_line_style = "--"
        if 'fit_line_style' in kwargs:
            test_inp(kwargs['fit_line_style'], str, "fit_line_style")
            self.func_format = kwargs['fit_line_style']

        self.line_only = False
        self.scatter = True

        self.func_format = ''
        if 'func_format' in kwargs:
            test_inp(kwargs['func_format'], str, "func_format")
            self.func_format = kwargs['func_format']

        self.n_points = 1000
        if "custom_fit_spacing" in kwargs:
            test_inp(kwargs["custom_fit_spacing"], int, "fit array size")
            self.n_points = kwargs["custom_fit_spacing"]

        self.sigma_uncertainty = False
        if "sigma_uncertainty" in kwargs:
            self.sigma_uncertainty = kwargs["sigma_uncertainty"]
            test_inp(self.sigma_uncertainty, (bool, int), "sigma_uncertainty")

        # ERROR ARRAYS
        if 'x_err' in kwargs:
            self.x_err = kwargs["x_err"]
            test_inp(self.x_err, (int, tuple, np.ndarray, list, float),
                     "x error")
            try:
                if isinstance(self.x_err, (int, float)):
                    self.x_err = np.full(self.x.size, self.x_err)

                if isinstance(self.x_err, (tuple, list)):
                    self.x_err = np.asarray(self.x_err)

                assert self.x_err.size == self.x.size
            except AssertionError:
                raise IndexError(
                    "x Error and y list are not of the same size.")
        else:
            self.x_err = []

        if 'y_err' in kwargs:
            self.y_err = kwargs["y_err"]
            test_inp(self.y_err, (int, tuple, np.ndarray, list, float),
                     "y error")
            try:
                if isinstance(self.y_err, (int, float)):
                    self.y_err = np.full(self.y.size, self.y_err)

                if isinstance(self.y_err, (tuple, list)):
                    self.y_err = np.asarray(self.y_err)

                assert self.y_err.size == self.y.size
            except AssertionError:
                raise IndexError(
                    "y Error and y list are not of the same size.")
        else:
            self.y_err = []

        self.kwargs = kwargs
        self.return_object = False

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

        self.default_plot()

        return self.fig, self.ax

    def __repr__(self):
        # len(x, y, xerr, yerr); fitcoeffs; file format
        return f"Default({len(self.x)}, {len(self.y)}, {len(self.x_err)}," \
               f" {len(self.y_err)}; {self.fit_stats()}; {self.file_format})"

    def __str__(self):
        return self.__repr__()

    def default_plot(self, show_error: bool = None,
                     return_error: bool = None,
                     fig: bool = False, ax: bool = False,
                     fig_format: bool = True):
        """
        Plot a 2D data set with errors in both x and y axes. The data
        will be fitted according to the input arguments in __innit__.

        Requires an Aplot object to plot with x, y or file input.

        OPTIONAL
        :param: show_error
            ObjectType -> bool
            Default, True when true prints out the error in the
            coefficients. When changed from default overwrites the print_error
            statement in __innit__.

        :param: return_error
            Objecttype -> bool
            Default, False when true returns a dictionary with coefficients,
            "coeffs" and error "error" of the fit parameters.

        EXAMPLES
        plot and show fig
        x, y = data
        plot_obj = Aplot(x, y, degree=1) # Linear fit to x and y data
        plot_obj.default_plot()

        plot and save fig as plot.png
        x, y = data
        plot_obj = Aplot(x, y, degree=1, save_as='plot.png') # Linear fit to x and y data
        plot_obj.default_plot()


        :return: Optional[dict[str, Union[Union[ndarray, Iterable, int, float], Any]]]
        """
        # DATA PLOTTING
        # TODO: add these extra kwargs to the docs

        if not ax:
            ax = self.ax

        if not fig:
            fig = self.fig

        if self.scatter:
            if len(self.y_err) == 0 and len(self.x_err) == 0:
                ax.plot(self.x, self.y,
                        label=self.label, marker=self.marker, c=self.colour,
                        linestyle=self.linestyle,
                        ms=self.capsize,
                        linewidth=self.linewidth
                )
            elif len(self.y_err) == 0 or len(self.x_err) == 0:
                if len(self.y_err) == 0:
                    ax.errorbar(self.x, self.y, xerr=self.x_err,
                                label=self.label, fmt=self.marker + self.colour,
                                linestyle=self.linestyle,
                                capsize=self.capsize)
                else:
                    ax.errorbar(self.x, self.y, yerr=self.y_err,
                                label=self.label, fmt=self.marker + self.colour,
                                linestyle=self.linestyle,
                                capsize=self.capsize)
            else:
                ax.errorbar(self.x, self.y, xerr=self.x_err, yerr=self.y_err,
                            label=self.label, fmt=self.marker + self.colour,
                            linestyle=self.linestyle,
                            capsize=self.capsize)

        fit_x = np.linspace(min(self.x), max(self.x), self.n_points)

        fit_pr = 3
        if "fit_precision" in self.kwargs:
            test_inp(self.kwargs["fit_precision"], int, "fit precision")
            fit_pr = self.kwargs["fit_precision"]

        if TNFormatter:
            if self.degree is not None or self.func is not None:
                str_fit_coeffs = [str(np.around(c, fit_pr)).replace(".", ",")
                                  for c
                                  in
                                  self.fit_coeffs]
        else:
            if self.degree is not None or self.func is not None:
                str_fit_coeffs = [str(np.around(c, fit_pr)) for c in
                                  self.fit_coeffs]

        if self.func is not None:
            ax.plot(fit_x, self.func(fit_x, *self.fit_coeffs),
                    linestyle=self.fit_line_style, c=self.colour if not self.fit_colour else self.fit_colour,
                    label=(
                        lambda _: self.func_format.format(*str_fit_coeffs)
                        if self.func_format != "" else "Fit")(None))
        elif self.degree is not None:
            if self.func_format != '':
                ax.plot(fit_x,
                        sum([fit_x ** (c) * self.fit_coeffs[
                            abs(c - self.degree)]
                             for c in
                             range(self.degree + 1).__reversed__()]),
                        linestyle=self.fit_line_style, c=self.colour if not self.fit_colour else self.fit_colour,
                        label=(self.func_format.format(*str_fit_coeffs)))
            else:
                ax.plot(fit_x,
                        sum([fit_x ** (c) * self.fit_coeffs[
                            abs(c - self.degree)]
                             for c in
                             range(self.degree + 1).__reversed__()]),
                        linestyle=self.fit_line_style, c=self.colour if not self.fit_colour else self.fit_colour,
                        label=(
                                "Fit with function %s = " % self.response_var +
                                self.degree_dict[self.degree].format(
                                    *str_fit_coeffs)))

        if self.sigma_uncertainty:
            self.add_sigma_uncertainty()

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

        if fig_format:
            self.single_form(x_label, y_label, grid=grid, fig_ax=(fig, ax))

        # dead code?
        if not return_error:
            plt.tight_layout()
            (lambda save_as:
             plt.show() if save_as == '' else plt.savefig(save_as,
                                                          bbox_inches='tight')
             )(self.save_as)

        else:
            return None

    def fit(self) -> Optional[
        Dict[str, Union[np.ndarray, Iterable, int, float]]]:
        """
        Calculate the fit parameters of an Aplot object.

        This function calculates the fit parameters based on either a polyfit or
        function fit. Where the function fit takes in a python function type,
        with first argument x and other arguments being the parameters.

        param show_fit -> moved to the fit_stats function

        :returns:
            None

        EXAMPLE WITH FUNCTION:
            # Function that depicts y = exp(-x + sqrt(a * x)) + b
            def f(x: float, a: float, b: float) -> float:
                return np.exp(-1*x + np.sqrt(a * x)) + b

            # load the data into an Aplot object
            plot_obj = Aplot(x, y, fx=f)

            # Run the fit attr with show_fit set to True.
            plot_obj.fit(True)

        EXAMPLE WITHOUT FUNCTION:
            # load the data into an Aplot object and set the degree variable
            plot_obj = Aplot(x, y, degree=1) # Linear data

            # Run the fit attr with show_fit set to True.
            plot_obj.fit(True)

        """

        self.degree_dict = {
            0: '{0}',
            1: '${0}%s + {1}$' % self.control_var,
            2: r'${0}%s^2 + {1}%s + {2}$' % (
                self.control_var, self.control_var),
            3: r'${0}%s^3 + {1}%s^2 + {2}%s + {3}$' % tuple(
                [self.control_var for _ in range(3)]),
            4: r'${0}%s^4 + {1}%s^3 + {2}%s^2 + {3}%s + {4}$' % tuple(
                [self.control_var for _ in range(4)]),
            5: r'${0}%s^5 + {1}%s^4 + {2}%s^3 + {3}%s^2 + {4}%s + {5}$' % tuple(
                [self.control_var for _ in range(5)])
        }

        if self.degree is not None:
            self.label_format = self.degree_dict[self.degree]
            if isinstance(self.degree, int):
                fit = np.polyfit(self.x, self.y, deg=self.degree, cov=True)
                self.fit_coeffs = fit[0]
                self.fit_errors = np.sqrt(np.diag(fit[1]))
        elif self.func is not None:
            fit, pcov = curve_fit(self.func, self.x, self.y)
            self.fit_coeffs = fit
            self.fit_errors = np.sqrt(np.diag(pcov))

        return None

    def lambdify_fit(self) -> Callable:
        """
        Turn the fitted line into a lambda function to manually calculate
        desired values.
        :return: lambda function, Object type Callable
        """
        if self.func is not None:
            return lambda x: self.func(x, *self.fit_coeffs)
        elif self.degree is not None:
            return lambda x: sum(
                [x ** (c) * self.fit_coeffs[abs(c - self.degree)]
                 for c in range(self.degree + 1).__reversed__()])
        else:
            raise AttributeError("Missing parameters to compute fit"
                                 " coefficients, add either 'degree' or 'fx'"
                                 "to the parameters.")

    def fit_stats(self) -> Dict:
        """
        Return fit values in a dictionary.

        The dictionary format is {"c": list of coefficients, "e": list of errors}

        Where the order of coefficients and errors is the same as defined in the
        given fit function or from highest order to lowest order when defined
        by the degree parameter.

        :return: Dictionary consisting out of fit coefficients and error in the
                 coefficients.
        """
        return {"c": self.fit_coeffs, "e": self.fit_errors}

    def add_sigma_uncertainty(self, base=None, linestyle: str = "--"):
        """
        Uses the sum of the errors in the fit to calculate an n-sigma
        fit boundary. The method uses the SUM meaning this method is only feasible
        for linear regressions.
        :param base:
            Base object from which the data is being pulled
        :param linestyle:
            Linestyle of the sigma line.
        :return:
        """
        if base is None:
            base = self

        sigma_label: str = base.kwargs["sigma_label"] if "sigma_label" in \
                                                         base.kwargs else \
            "%s sigma boundary" % base.sigma_uncertainty
        sigma_colour: str = base.kwargs["sigma_colour"] if "sigma_colour" in \
                                                           base.kwargs else "gray"

        x_vals = np.linspace(min(self.x), max(self.x), self.n_points)
        if self.func is not None:
            # Plot the lower bound
            self.ax.plot(x_vals,
                         y=base.func(x_vals, *base.fit_coeffs) - sum(
                             base.fit_errors),
                         linestyle=linestyle,
                         c=sigma_colour, label=sigma_label)
            self.ax.plot(x_vals,
                         base.func(x_vals, *base.fit_coeffs) + sum(
                             base.fit_errors),
                         linestyle=linestyle,
                         c=sigma_colour)
        else:
            lower_y = np.asarray(
                sum([x_vals ** (c) * base.fit_coeffs[abs(c - self.degree)]
                     for c in range(self.degree + 1).__reversed__()])) - sum(
                base.fit_errors)
            upper_y = np.asarray(
                sum([x_vals ** (c) * base.fit_coeffs[abs(c - self.degree)]
                     for c in range(self.degree + 1).__reversed__()])) + sum(
                base.fit_errors)

            self.ax.plot(x_vals, lower_y, linestyle=linestyle,
                         c=sigma_colour, label=sigma_label)
            self.ax.plot(x_vals, upper_y, linestyle=linestyle,
                         c=sigma_colour)
        return None


class Histogram(Base):

    def __innit__(self, x: Union[tuple, list, np.ndarray, str] = None,
                  **kwargs):
        self.x = x
        bins = int(np.round(len(self.x) / np.sqrt(2)))
        if "binning" in kwargs:
            bins = kwargs["binning"]
            try:
                bins = int(bins)
            except ValueError:
                raise ValueError("Bins must be an integer or convertible to an"
                                 " integer.")

            test_inp(bins, int, "binning", True)

        norm_label = "Label"
        if "norm_label" in kwargs:
            norm_label = kwargs["norm_label"]
            test_inp(norm_label, str, "norm label")

        if "sigma_lines" in kwargs:
            pass

    def __call__(self):
        pass

    def histogram(self):
        pass

class LivePlot(Base):
    """

    Func > update function for Matplotlob.Animate, standard function by
           static class method update
    pass_x_to_y > Pass x input to y function, standard False
    end_point > either a time in seconds or a function that will run parallel
                (in thread) and when the function task is finished stops the plot.

    """
    def __init__(self, *args, func: Callable = None, pass_x_to_y: bool = False,
                 end_point: Union[float, int, Callable] = 10, **kwargs):
        super().__init__(*args, **kwargs)

        self.args, self.kwargs = args, kwargs

        if self.x is not None and self.y is None:
            self.y = self.x

        self.func = self.update
        if func is not None:
            self.func = func

        x_data, y_data = [], []
        yargs = len(signature(self.y).parameters)

        self.line_dict = {0: {"x": self.x, "y": self.y,
                              "xd": x_data, "yd": y_data,
                              "iter": 0, "yargs": yargs,
                              "plot": self.ax.plot([], [], linestyle=self.linestyle,
                                           marker=self.marker, label=self.label,
                                                   ms=self.capsize, linewidth=self.linewidth),
                              "plotargs": [self.args, self.kwargs]}}

        self.fit_dict = {}
        self.fpa_dict = {}

        self.thread = None
        self.endpoint = None
        self.passframe = False

        self.tstart = time.time()

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def append(self, x: Callable = None, y: Callable= None, *args, **kwargs):
        # Deadcode?
        if x is not None and y is None:
            y = x

        linestyle = "-"
        if "linestyle" in kwargs:
            linestyle = kwargs["linestyle"]

        marker = "."
        if "marker" in kwargs:
            marker = kwargs["marker"]

        x_data, y_data = [], []
        yargs = len(signature(y).parameters)

        self.line_dict[len(self.line_dict.keys())] =\
            {"x": x, "y": y,
             "xd": x_data, "yd": y_data,
             "iter": 0, "yargs": yargs,
             "plot": self.ax.plot([], [], linestyle=linestyle,
                                  marker=marker, label=self.label),
             "plotargs": [args, kwargs]}

    def live_fit(self, target: int):
        self.fit_dict[target] = {
            "plot": self.ax.plot([], [], linestyle=self.linestyle,
                                 marker=self.marker, label="Fit",
                                 c=self.line_dict[target]["plot"].color),
            "x": [], "y": []}

    def live_floating_point_average(self, target: int, sample_size: int):
        """
        Initialise a fpa for a target datastream

        target > target data stream int of added data
        sample_size > sample size in frames (or datapoints) for the fpa
        """
        self.fpa_dict[target] = {"plot": self.ax.plot([], [], linestyle="--",
                                                      label="Floating point average",
                                                      c=self.line_dict[target]["plot"][0].get_c(),
                                                      linewidth=1),
                                 "sample_size": sample_size,
                                 "xf": [], "yf": []}

    def comp_live_floating_point_average(self, target):
        xd, yd = self.line_dict[target]["xd"], self.line_dict[target]["yd"]

        # print("True", y_data.__len__())
        new_len = int(len(yd) / self.fpa_dict[target]["sample_size"])
        # print(new_len)
        y = compress_ind(np.asarray(yd), new_len)[0]
        # print(compress_ind(np.arange(0, y_data.__len__()), new_len)[0].astype(int))
        x = np.asarray(xd)[(compress_ind(np.arange(0, yd.__len__()), new_len)[0]).astype(int)]
        return x, y

    def run(self, *args, **kwargs):
        """
        Interval > interval for FuncAnimation
        passframe > pass frame to x function if x is function
        """
        interval = 1
        if "interval" in kwargs:
            interval = kwargs["interval"]

        if "endpoint" in kwargs:
            self.endpoint = kwargs["endpoint"]

        if "passframe" in kwargs:
            self.passframe = kwargs["endpoint"]

        if self.endpoint.__class__.__name__ == "function":
            import threading

            endarg = False
            if "endarg" in kwargs:
                endarg = kwargs["endarg"]

            self.thread = threading.Thread(target=self.endpoint,
                                           args=(endarg, ) if endarg else ())
            self.thread.start()

        self.single_form(*self.args, **self.kwargs)
        plt.legend()

        self.animation = FuncAnimation(self.fig, self.update, interval=interval)

        plt.show()

        if "x" in self.kwargs:
            del self.kwargs["x"]

        if "y" in self.kwargs:
            del self.kwargs["y"]

        bplot = Default(self.line_dict[0]["xd"], self.line_dict[0]["yd"], *self.args, **self.kwargs)
        if len(self.line_dict.keys()) > 1:
            for eplot in range(1, len(self.line_dict.keys())):
                bplot += Default(self.line_dict[eplot]["xd"], self.line_dict[eplot]["yd"], add_mode=True,
                                 *self.line_dict[eplot]["plotargs"][0], **self.line_dict[eplot]["plotargs"][1])

        # Change colour and linedwidth back to normal
        # Colour : c=self.line_dict[eplot]["plot"][0].get_c(
        # Linedwidth: None
        if len(self.fpa_dict.keys()) >= 1:
            for eplot in list(self.fpa_dict.keys()):
                bplot += Default(self.fpa_dict[eplot]["xf"], self.fpa_dict[eplot]["yf"], line_only=True,
                                                      label="Floating point average",
                                 colour="C1",
                                 linestyle="--",
                                 marker="",
                                 add_mode=True, linewidth=7.0)

        return bplot

    def update(self, frame):
        # For all lines
        for i in list(self.line_dict.keys()):
            # Rename the dictionary
            ldict = self.line_dict

            # If the input function takes no arguments,
            #   Check if the x entry is a list or a function in case of function
            #   Call and ad the return value, in case of list append the frame nr
            #   Since the y function doesnt take a frame or an x append the values to lists.

            if ldict[i]["yargs"] == 0:
                if isinstance(ldict[i]["x"],(list, range, np.ndarray, tuple)):
                    ldict[i]["xd"].append(ldict[i]["x"]) 
                elif isinstance(ldict[i]["x"], types.FunctionType):
                    print(ldict[i]["x"]())
                    ldict[i]["xd"].append(ldict[i]["x"]())
                else:
                    ldict[i]["xd"].append(frame)
                print(isinstance(ldict[i]["x"], types.FunctionType))
                print(ldict[i]["x"])
                print(ldict[i]["xd"])
                ldict[i]["yd"].append(ldict[i]["y"]())

            # If the input takes one argument assume x
            #   Again check if x is a list to check for predetermined x values
            #   from list or function.
            #   Append both values to list for export

            elif ldict[i]["yargs"] == 1:
                if isinstance(ldict[i]["x"], (list, range, np.ndarray, tuple)):
                    ldict[i]["xd"].append(ldict[i]["x"])
                elif isinstance(ldict[i]["x"], types.FunctionType):
                    ldict[i]["xd"].append(ldict[i]["x"]())
                else:
                    ldict[i]["xd"].append(frame)

                if self.passframe:
                    ldict[i]["yd"].append(ldict[i]["y"](frame))
                else:
                    ldict[i]["yd"].append(ldict[i]["y"](ldict[i]["xd"][-1]))

            # If the input takes two arguments assume x, frame
            #   The animation will be a standing animation in the range of the x-list
            #   will wrap around iteration == len(x) to zero, the full x-array is passed
            #   to the y function, assume that x is an array and not a function.
            #   TODO: Add function possibility to 2 arg y input

            elif ldict[i]["yargs"] >= 2:
                ldict[i]["xd"] = ldict[i]["x"]
                ldict[i]["yd"] = ldict[i]["y"](ldict[i]["x"], frame)

            # Set the line data and update the frame/iteration according to
            # input values to assure no index error shows up.
            ldict[i]["plot"][0].set_data(ldict[i]["xd"], ldict[i]["yd"])

            if isinstance(ldict[i]["x"], (list, range, np.ndarray, tuple)):
                ldict[i]["iter"] = (ldict[i]["iter"] + 1) % len(ldict[i]["iter"])

        for i in list(self.fpa_dict.keys()):
            if frame % self.fpa_dict[i]["sample_size"] == 0 and frame >= self.fpa_dict[i]["sample_size"]:
                xf, yf = self.comp_live_floating_point_average(i)

                self.fpa_dict[i]["xf"] = xf
                self.fpa_dict[i]["yf"] = yf

                self.fpa_dict[i]["plot"][0].set_data(xf, yf)


        # Check if the thread is running, if not start a new one
        # New threads cant be started thus the thread is "reset"
        if self.thread is not None:
            if not self.thread.is_alive():
                plt.close() # Gives error, TODO: Fix this error

        if self.endpoint.__class__.__name__ in ("int", "float"): 
            if time.time() > self.endpoint + self.tstart:
                plt.close()  # Gives error, TODO: Fix this error

        if self.x_lim or self.y_lim:
            if self.x_lim:
                self.ax.set_xlim(self.x_lim)
            if self.y_lim:
                self.ax.set_ylim(self.y_lim)

        else:
            self.fig.gca().relim()
            self.fig.gca().autoscale_view()

        return None

    


def format_figure(x_label: str, y_label: str, grid: bool = True, **kwargs):
    return Default(**kwargs).single_form(x_label, y_label, grid, **kwargs)


def multi_plot(plots: list, fig_size: tuple = (10, 6), save_as: str = "",
               run_plot=True, plot_title=""):
    """
    Combine multiple AdrianPack plots into one figure with a grid layout.

    :param plots: List of plots to be plotted in a grid
    :type plots: List[List[Default, Histogram]]
    :param fig_size: Size of the figure when plotted
    :type fig_size: Tuple
    :param save_as: Name of the file to save the plot as
    :type save_as: str
    :param run_plot: Argument to run the plot after creation
    :type run_plot: bool
    :return: fig, axes
    :rtype: Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    test_inp(plots, list, "plot grid")

    plt.rcParams["figure.figsize"] = fig_size

    if type(plots[0]) is not list:
        plots = [plots, ]

    for i in plots:
        test_inp(i, list, "row %s" % i)
        try:
            assert len(i) == len(plots[0])
        except AssertionError:
            raise IndexError("Plots must be given in an mxn matrix or m long"
                             "row/column vector.")

    for i in plots:
        for j in i:
            test_inp(j, (Default, Histogram), "row %s" % i)

    rows = len(plots)
    columns = len(plots[0])

    colours = ["C%s" % i for i in range(1, 10)]

    plt.clf()
    plt.close()

    fig, axes = plt.subplots(rows, columns)

    if plot_title != "":
        fig.suptitle(plot_title)

    # Column vector
    if columns == 1:
        for ax in range(len(axes)):
            if len(plots[ax][0].plots) >= 1:
                for extra_plot in plots[ax][0].plots:
                    colour = extra_plot.colour

                    if not colour:
                        extra_plot.colour = colours[
                            plots[ax][0].plots.index(extra_plot)]

                    extra_plot.default_plot(ax=axes[ax], fig=fig,
                                            return_error=True,
                                            fig_format=False)
                    extra_plot.colour = colour

            plots[ax][0].default_plot(ax=axes[ax], fig=fig, return_error=True)

    # Row vector
    elif rows == 1:
        for ax in range(len(axes)):
            if len(plots[0][ax].plots) >= 1:
                for extra_plot in plots[0][ax].plots:
                    colour = extra_plot.colour

                    if not colour:
                        extra_plot.colour = colours[
                            plots[0][ax].plots.index(extra_plot)]

                    extra_plot.default_plot(ax=axes[ax], fig=fig,
                                            return_error=True,
                                            fig_format=False)
                    extra_plot.colour = colour

            plots[0][ax].default_plot(ax=axes[ax], fig=fig, return_error=True)


    # Matrix
    else:
        for row in range(rows):
            for ax in range(len(axes[row])):
                if len(plots[row][ax].plots) >= 1:
                    for extra_plot in plots[row][ax].plots:
                        colour = extra_plot.colour

                        if not colour:
                            extra_plot.colour = colours[
                                plots[row][ax].plots.index(extra_plot)]

                        extra_plot.default_plot(ax=axes[row][ax], fig=fig,
                                                return_error=True,
                                                fig_format=False)

                        extra_plot.colour = colour

                plots[row][ax].default_plot(ax=axes[row][ax], fig=fig,
                                            return_error=True)

    plt.tight_layout()

    if plot_title != "":
        # Adjust for title
        fig.subplots_adjust(top=0.88)

    if run_plot:
        (lambda save_as:
         plt.show() if save_as == '' else plt.savefig(save_as,
                                                      bbox_inches='tight')
         )(save_as)

    return fig, axes


if __name__ == "__main__":
    import time
    from datetime import datetime
    import random

    t_start = time.time()

    noise = 0.5
    def f(x) -> float:
        return 1/10 * x + 4 *np.sin(1 / 4 * x) + noise * random.random() * (
            -1) ** random.randint(0, 1)

def dtm():
    return time.time() - t_start
    
    def g(x: float) -> float:
        return np.cos(1/16 * x) + noise * random.random() * (-1)**random.randint(0, 1)

    def sleeptimer(t):
        for i in range(t):
            time.sleep(1)
            print(i)
        return None


    lp = LivePlot(x=dtm, y=f, x_label="time", y_label="Data",
                  linestyle="", capsize=4, linewidth=2)
    # lp.append(y=g, connecting_line=True)
    lp.live_floating_point_average(0, 10)
    plot = lp.run(interval=10, endpoint=20)
    plot2 = Default(np.linspace(0, 200, 100), y=f(np.linspace(0, 200, 100)), add_mode=True)
    multi_plot([plot, plot2])

