import types

import numpy.random
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from typing import Sized, Iterable, Union, Optional, Any, Type, Tuple, List, \
    Dict, Callable

try:
    from TN_code.plotten.TISTNplot import TNFormatter
except ImportError:
    try:
        from .TN_code.plotten.TISTNplot import TNFormatter
    except ImportError:
        TNFormatter = False

try:
    from Helper import test_inp
except ImportError:
    from .Helper import test_inp

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

        # Read x and y values
        self.file_format = kwargs["file_format"] if "file_format" in kwargs else ["x", "y", "x_err", "y_err"]
        test_inp(self.file_format, list, "File format")

        if len(args) >= 2 and args[1] is not None:
            self.x = args[0]
            self.y = args[1]
        elif len(args) == 1 or args[1] is None:
            self.x = args[0]
            self.y = None
        elif "file" in kwargs:
            self.x = kwargs["file"]

        if self.x.__class__.__name__ in ["ndarray", "list", "tuple"]:
            # either matrix or normal x and y
            self.x = np.asarray(self.x, dtype=np.float32)
            if self.y is not None:
                self.y = np.asarray(self.y, dtype=np.float32)
            # 1 Dimensional array
            if self.x.ndim == 1:
                if self.y is None:
                    self.y = self.x
                    self.x = range(len(self.x))

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
            data_obj = Fileread(path=self.x, **kwargs) if type(self.x) is str else self.x
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

        self.kwargs = kwargs

    def array_parse(self, data):
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

            self.fig = kwargs["fig_ax"][0]
            self.ax = kwargs["fig_ax"][1]

        if "x_lim" in self.kwargs:
            self.ax.set_xlim(self.x_lim)
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

            self.ax.set_xlim(x_lim)

        if "y_lim" in self.kwargs:
            self.ax.set_ylim(self.y_lim)
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

            self.ax.set_ylim(y_lim)

        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)

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

            self.ax.xaxis.set_major_formatter(TNFormatter(x_pr))
            self.ax.yaxis.set_major_formatter(TNFormatter(y_pr))

        if grid:
            plt.grid()

        if "legend_loc" in self.kwargs:
            test_inp(self.kwargs["legend_loc"], str, "legenc loc")
            plt.legend(loc=self.kwargs["legend_loc"])
        elif "legend_loc" in kwargs:
            test_inp(self.kwargs["legend_loc"], str, "legenc loc")
            plt.legend(loc=kwargs["legend_loc"])
        else:
            plt.legend(loc='lower right')

        plt.tight_layout()
        if "fig_ax" in kwargs:
            return self.fig, self.ax
        else:
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
            self.fit()
        elif 'fx' in kwargs:
            self.func = kwargs["fx"]
            test_inp(self.func, types.FunctionType, "f(x)")
            self.fit()
        else:
            pass

        self.connecting_line = False
        if 'connecting_line' in kwargs:
            test_inp(kwargs["connecting_line"], bool, "connecting_line")
            self.connecting_line = kwargs['connecting_line']

        self.connecting_line_label = "Connection"
        if 'connecting_line_label' in kwargs:
            test_inp(kwargs["connecting_line_label"], str, "connecting_line_label")
            self.connecting_line_label = kwargs['connecting_line_label']

        self.line_only = False
        self.scatter = True
        if 'line_only' in kwargs:
            test_inp(kwargs["line_only"], bool, "line_only")
            self.connecting_line = kwargs['line_only']
            self.scatter = not kwargs['line_only']

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
                raise IndexError("x Error and y list are not of the same size.")
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
                raise IndexError("y Error and y list are not of the same size.")
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

    def __add__(self, other):
        test_inp(other, Default, "Aplot.Default")

        other.fit()

        if "colour" in other.kwargs:
            test_inp(other.kwargs["colour"], str, "colour")
            color = other.kwargs["colour"]
        else:
            color = "C1"

        if other.scatter:
            if len(other.y_err) == 0 and len(other.x_err) == 0:
                self.ax.scatter(other.x, other.y, label=other.label, color=color)
            elif len(other.y_err) == 0 or len(other.x_err) == 0:
                if len(other.y_err) == 0:
                    self.ax.errorbar(other.x, other.y, xerr=other.x_err,
                                     label=other.label, fmt=color + 'o',
                                     linestyle='',
                                     capsize=4)
                else:
                    self.ax.errorbar(other.x, other.y, yerr=other.y_err,
                                     label=other.label, fmt=color + 'o',
                                     linestyle='',
                                     capsize=4)
            else:
                self.ax.errorbar(other.x, other.y, xerr=other.x_err, yerr=other.y_err,
                                 label=other.label, fmt=color + 'o',
                                 linestyle='',
                                 capsize=4)

        if other.connecting_line:
            self.ax.plot(other.x, other.y, label=other.connecting_line_label,
                         color=color)

        if other.sigma_uncertainty:
            self.sigma_uncertainty(base=other)

        fit_x = np.linspace(min(other.x), max(other.x), other.n_points)

        fit_pr = 3
        if "fit_precision" in other.kwargs:
            test_inp(other.kwargs["fit_precision"], int, "fit precision")
            fit_pr = other.kwargs["fit_precision"]

        if other.degree is not None or other.func is not None:
            if self.decimal_comma:
                str_fit_coeffs = [str(np.around(c, fit_pr)).replace(".", ",") for c
                                  in
                                  other.fit_coeffs]
            else:
                str_fit_coeffs = [str(np.around(c, fit_pr)) for c in
                                  other.fit_coeffs]

        if other.func is not None:
            self.ax.plot(fit_x, other.func(fit_x, *other.fit_coeffs),
                         linestyle="--", c=color,
                         label=(
                             lambda _: other.func_format.format(*str_fit_coeffs)
                             if other.func_format != "" else "Fit")(None))
        elif other.degree is not None:
            if other.func_format != '':
                self.ax.plot(fit_x,
                             sum([fit_x ** (c) * other.fit_coeffs[
                                 abs(c - other.degree)]
                                  for c in
                                  range(other.degree + 1).__reversed__()]),
                             linestyle="--", c=color,
                             label=(other.func_format.format(*str_fit_coeffs)))
            else:
                self.ax.plot(fit_x,
                             sum([fit_x ** (c) * other.fit_coeffs[
                                 abs(c - other.degree)]
                                  for c in
                                  range(other.degree + 1).__reversed__()]),
                             linestyle="--", c=color,
                             label=(
                                         "Fit with function %s = " % other.response_var +
                                         other.degree_dict[other.degree].format(
                                             *str_fit_coeffs)))


        return self

    def default_plot(self, show_error: bool = None,
                     return_error: bool = None):
        """
        Plot a 2D data set with errors in both x and y axes. The data
        will be fitted according to the input arguments in __innit__.

        Requires a Aplot object to plot with x, y or file input.

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

        if "colour" in self.kwargs:
            test_inp(self.kwargs["colour"], str, "colour")
            color = self.kwargs["colour"]
        else:
            color = "C0"

        if "marker_fmt" in self.kwargs:
            test_inp(self.kwargs["marker_fmt"], str, "marker format")
            mark = self.kwargs["marker_fmt"]
        else:
            mark = "C0"

        if self.scatter:
            if len(self.y_err) == 0 and len(self.x_err) == 0:
                self.ax.scatter(self.x, self.y, label=self.label, color=color)
            elif len(self.y_err) == 0 or len(self.x_err) == 0:
                if len(self.y_err) == 0:
                    self.ax.errorbar(self.x, self.y, xerr=self.x_err,
                                     label=self.label, fmt=color + mark,
                                     linestyle='',
                                     capsize=4)
                else:
                    self.ax.errorbar(self.x, self.y, yerr=self.y_err,
                                     label=self.label, fmt=color + mark,
                                     linestyle='',
                                     capsize=4)
            else:
                self.ax.errorbar(self.x, self.y, xerr=self.x_err, yerr=self.y_err,
                                 label=self.label, fmt=color + mark, linestyle='',
                                 capsize=4)

        if self.connecting_line:
            self.ax.plot(self.x, self.y, label=self.connecting_line_label,
                         color=color)

        # FIT PLOTTING
        if show_error:
            print(self.fit_errors)

        fit_x = np.linspace(min(self.x), max(self.x), self.n_points)

        fit_pr = 3
        if "fit_precision" in self.kwargs:
            test_inp(self.kwargs["fit_precision"], int, "fit precision")
            fit_pr = self.kwargs["fit_precision"]

        if TNFormatter:
            if self.degree is not None or self.func is not None:
                str_fit_coeffs = [str(np.around(c, fit_pr)).replace(".", ",") for c
                                  in
                                  self.fit_coeffs]
        else:
            if self.degree is not None or self.func is not None:
                str_fit_coeffs = [str(np.around(c, fit_pr)) for c in self.fit_coeffs]

        if self.func is not None:
            self.ax.plot(fit_x, self.func(fit_x, *self.fit_coeffs),
                         linestyle="--", c=color,
                         label=(
                             lambda _: self.func_format.format(*str_fit_coeffs)
                             if self.func_format != "" else "Fit")(None))
        elif self.degree is not None:
            if self.func_format != '':
                self.ax.plot(fit_x,
                             sum([fit_x ** (c) * self.fit_coeffs[
                                 abs(c - self.degree)]
                                  for c in
                                  range(self.degree + 1).__reversed__()]),
                             linestyle="--", c=color,
                             label=(self.func_format.format(*str_fit_coeffs)))
            else:
                self.ax.plot(fit_x,
                             sum([fit_x ** (c) * self.fit_coeffs[
                                 abs(c - self.degree)]
                                  for c in
                                  range(self.degree + 1).__reversed__()]),
                             linestyle="--", c=color,
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

        super().single_form(x_label, y_label, grid=grid, fig_ax=(self.fig, self.ax))

        if not self.return_object:
            (lambda save_as:
             plt.show() if save_as == '' else plt.savefig(save_as,
                                                          bbox_inches='tight')
             )(self.save_as)

        if return_error:
            return {"Coeffs": self.fit_coeffs, "Errors": self.fit_errors}

        return None

    def fit(self) -> Optional[Dict[str, Union[np.ndarray, Iterable, int, float]]]:
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
            return lambda x: sum([x ** (c) * self.fit_coeffs[abs(c - self.degree)]
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

        sigma_label: str = base.kwargs["sigma_label"] if "sigma_label" in\
                                                         base.kwargs else\
                                                        "%s sigma boundary" % base.sigma_uncertainty
        sigma_colour: str = base.kwargs["sigma_colour"] if "sigma_colour" in\
                                                         base.kwargs else "gray"

        x_vals = np.linspace(min(self.x), max(self.x), self.n_points)
        if self.func is not None:
            # Plot the lower bound
            self.ax.plot(x_vals,
                         y=base.func(x_vals, *base.fit_coeffs) - sum(base.fit_errors),
                         linestyle=linestyle,
                         c=sigma_colour, label=sigma_label)
            self.ax.plot(x_vals,
                         base.func(x_vals, *base.fit_coeffs) + sum(base.fit_errors),
                         linestyle=linestyle,
                         c=sigma_colour)
        else:
            lower_y = np.asarray(sum([x_vals ** (c) * base.fit_coeffs[abs(c - self.degree)]
                           for c in range(self.degree + 1).__reversed__()])) - sum(base.fit_errors)
            upper_y = np.asarray(sum([x_vals ** (c) * base.fit_coeffs[abs(c - self.degree)]
                           for c in range(self.degree + 1).__reversed__()])) + sum(base.fit_errors)

            self.ax.plot(x_vals, lower_y, linestyle=linestyle,
                         c=sigma_colour, label=sigma_label)
            self.ax.plot(x_vals, upper_y, linestyle=linestyle,
                         c=sigma_colour)
        return None


class Multi:

    def __innit__(self):
        pass

    def __call__(self):
        pass


class Hist:

    def __innit__(self):
        pass

    def __call__(self):
        pass

    def histogram(self):
        """

        :return:
        """
        self.fig, self.ax = plt.subplots()

        bins = int(np.round(len(self.x) / np.sqrt(2)))
        if "binning" in self.kwargs:
            bins = self.kwargs["binning"]
            try:
                bins = int(bins)
            except ValueError:
                raise ValueError("Bins must be an integer or convertible to an"
                                 " integer.")

            test_inp(bins, int, "binning", True)

        norm_label = "Label"
        if "norm_label" in self.kwargs:
            norm_label = self.kwargs["norm_label"]
            test_inp(norm_label, str, "norm label")

        if "sigma_lines" in self.kwargs:
            pass

        self.ax.hist(self.x, bins=bins, label=self.label)

        norm_dist = self.norm_dist_plot()
        self.ax.plot(norm_dist[0], norm_dist[1], label=norm_label)

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

        self.single_form(x_label, y_label, grid)

        if not self.return_object:
            (lambda save_as:
             plt.show() if save_as == '' else plt.savefig(save_as,
                                                          bbox_inches='tight')
             )(self.save_as)

        return None

    def norm_dist_plot(self, scaling: bool = True):
        """

        :return:
        """
        if "x_lim" in self.kwargs:
            x = np.linspace(self.x_lim[0], self.x_lim[1], self.n_points)
        else:
            x = np.linspace(min(self.x) + max(self.x) * 1 / 10,
                            max(self.x) + max(self.x) * 1 / 10, self.n_points)

        mu, std = np.average(x), np.std(x)
        dist = scipy.stats.norm.pdf(x, mu, std)

        if scaling:
            factor = np.average(self.x) / np.max(dist)
            if "scaling_factor" in self.kwargs:
                factor = self.kwargs["scaling_factor"]
                test_inp(factor, (float, int), "scaling factor")
            dist = dist * abs(factor)

        return x, dist, mu


if __name__ == "__main__":
    import time
    import Aplot

    t_start = time.time()


    def f(x: float, a: float, b: float) -> float:
        return a * x ** 2 + b * x


    points = 40
    x = np.linspace(-5, 5, points)
    noise = np.random.randint(-2, 2, points)
    plot = Default(x=x, y=np.array([i * -4.32 + 9.123 for i in x] + noise),
          y_err=10, x_err=0.1, y_lim=[-50, 50], x_label="bruh x", y_label="bruh y", degree=2,
                   connecting_line=True,
                   func_format="$y_{{result}} = {0} \cdot \cos{{ (x_{{func}} \cdot {1}) }} + {0}^{{x_{{func}} + {2}}}$")
    
    add = Default(x=x, y=np.array([i * 4.4 + 0.12 for i in x] + noise),
          y_err=10, x_err=0.1, add_mode=True, line_mode=True, degree=2)
    # hist = Aplot([3, 2, 3, 1, 3, 4, 2, 4, 5, 6, 5], mode="hist", x_lim=[0, 7],
    #              x_label="X-as", grid=False)()
    plot += add

    plot()

    print("t: ", time.time() - t_start)
