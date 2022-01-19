
from typing import Sized
from matplotlib.ticker import EngFormatter
import numpy as np
import matplotlib.pyplot as plt

try:
    from TISTNplot import TNFormatter
except ImportError:
    TNFormatter = EngFormatter(places=1, sep="\N{THIN SPACE}")

def plottr(x_val: Sized, y_val: Sized, x_label: str,
                 y_label: str, degree: int, round: int, force_zero=False,
                 label='', save_as='') -> None:
    """"
            Build plots with fit

            :param: x_val -> Sized/Iterable with x values
            :param: y_val -> Sized/Iterable with y values
            :param: degree -> int, degree of func.
            :param: force_zero -> bool, force the graph through zero *non functional*

            :param: x_label -> str, label on the x-axis
            :param: y_label -> str, label on the y-axis

            :param: round -> int, sign. fig. in the fit label
            :param: label -> str, fig label
            :param: save_as -> str, name.format

    """



    fig, ax = plt.subplots()
    fit = np.polyfit(x_val, y_val, degree)

    degree_dict = {
        0: '{0}',
        1: '${0}x + {1}$',
        2: r'${0}x^2 + {1}x + {2}$',
        3: r'${0}x^3 + {1}x^2 + {2}x + {3}$'
    }

    ax.scatter(
        x_val,
        y_val,
        label=(lambda label: None if label == '' else label)(label)
    )

    # Removable when TN_code is not available
    ax.xaxis.set_major_formatter(TNFormatter(3))
    ax.yaxis.set_major_formatter(TNFormatter(3))

    ax.plot(
        x_val,
        sum([(np.array(x_val) ** (i + 2)) * fit[i] for i in
             range(degree)]),
        label=(
            degree_dict[degree].format(*(np.around(fit, round))))
    )

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.grid()
    plt.legend(loc='lower right')
    plt.tight_layout()

    (lambda save_as:
     plt.show() if save_as == '' else plt.savefig(save_as,
                                                  bbox_inches='tight')
     )(save_as)
    return None

class Ez_plotter:
    def __init__(self, x_val: Sized, y_val: Sized, x_label: str,
                 y_label: str, degree: int, round: int, force_zero=False,
                 label='', save_as=''):
        """"
        Build plots with fit

        :param: x_val -> Sized/Iterable with x values
        :param: y_val -> Sized/Iterable with y values
        :param: degree -> int, degree of func.
        :param: force_zero -> bool, force the graph through zero *non functional*

        :param: x_label -> str, label on the x-axis
        :param: y_label -> str, label on the y-axis

        :param: round -> int, sign. fig. in the fit label
        :param: label -> str, fig label
        :param: save_as -> str, name.format

        """
        assert len(x_val) == len(y_val)
        assert isinstance(degree, int) and isinstance(round, int)
        assert isinstance(save_as, str) and isinstance(label, str)
        assert isinstance(force_zero, bool)

        self.x = x_val
        self.x_label = x_label
        self.y = y_val
        self.y_label = y_label

        self.label = label
        self.degree = degree
        self.force_zero = force_zero

        self.save_as = save_as
        self.round = round

    def __call__(self):
        fig, ax = plt.subplots()
        fit = np.polyfit(self.x, self.y, self.degree)

        degree_dict = {
            0: '{0}',
            1: '${0}x + {1}$',
            2: r'${0}x^2 + {1}x + {2}$',
            3: r'${0}x^3 + {1}x^2 + {2}x + {3}$'
        }

        ax.scatter(
            self.x,
            self.y,
            label=(lambda label: None if label == '' else label)(self.label)
        )

        # Removable when TN_code is not available
        ax.xaxis.set_major_formatter(TNFormatter(3))
        ax.yaxis.set_major_formatter(TNFormatter(3))

        ax.plot(
            self.x,
            sum([(np.array(self.x) ** (i + 2)) * fit[i] for i in
                 range(self.degree)]),
            label=(
                degree_dict[self.degree].format(*(np.around(fit, self.round))))
        )

        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)

        plt.grid()
        plt.legend(loc='lower right')
        plt.tight_layout()

        (lambda save_as:
         plt.show() if save_as == '' else plt.savefig(save_as,
                                                      bbox_inches='tight')
         )(self.save_as)