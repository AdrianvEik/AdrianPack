""""
Collection of helper functions

- DMM_ERR: Error in DMM (currently only DC  voltage)
- Peak_finder: Find peaks in weird Data sets.
-

DEPENDENCIES:
    numpy, pandas, matplotlib and *TISTNplot.
*optional
"""



def calc_err_DMM(unit: str, val: float, freq=1.0) -> Iterable:
    """
    unit in "(factor) (volt/amp)"
    val, value in SI.
    e_type "DC"/"AC"
    freq frequency in Hertz
    """
    unit = unit.lower()
    # Standard input is DC
    assert (lambda u: u.split(' ')[2] if len(u.split(' ')) == 3 else 'dc')(unit)\
           in ['ac', 'dc']
    # Assert the inputs are correct
    assert (lambda u: u.split(' ')[1] if 2 <= len(u.split(' ')) <= 3 else u.split(' ')[0])(unit)\
           in ['volt', 'ampere']
    assert (lambda u: u.split(' ')[0] if 2 <= len(u.split(' ')) <= 3 else 'None')(unit)\
           in ['nano', 'micro', 'milli', 'None', 'kilo', 'mega']

    factor_val = (lambda u: u.split(' ')[0] if len(u.split(' ')) == 2 else 'None')(unit)
    unit = (lambda u: u.split(' ')[1] if len(u.split(' ')) == 2 else u.split(' ')[0])(unit)
    e_type = (lambda u: u.split(' ')[2] if len(u.split(' ')) == 3 else 'dc')(unit)

    factor = {
        'nano': 10e-9,
        'micro': 10e-6,
        'milli': 10e-3,
        'None': 1,
        'kilo': 10e3,
        'mega': 10e6
    }
    val = val * factor[factor_val]

    if e_type is "dc":
        if unit in 'volt':
            unit = {0.4: 4 * 10**(-5),
                    4: 4 * 10**(-4),
                    40: 4 * 10**(-3),
                    400: 4 * 10**(-2),
                    1000: 4 * 10**(-1)}
            distance = [(lambda i: i - val if i - val > 0 else float('inf'))(i)
                        for i in list(unit.keys())]
            return unit[list(unit.keys())[
                distance.index(min(distance))]] + val * 0.08 * 10 ** (-2)
        elif unit in 'ampere':
            unit = {0.4: 4 * 10**(-5),
                    4: 4 * 10**(-4),
                    40: 4 * 10**(-3),
                    400: 4 * 10**(-2),
                    1000: 4 * 10**(-1)}
            distance = [(lambda i: i - val if i - val > 0 else float('inf'))(i)
                        for i in list(unit.keys())]
            return unit[list(unit.keys())[
                distance.index(min(distance))]] + val * 0.08 * 10 ** (-2)

    elif e_type is "ac":
        if unit in 'volt':
            unit_freq = {0.4: 4 * 10**(-5),
                    4: 4 * 10**(-4),
                    40: 4 * 10**(-3),
                    400: 4 * 10**(-2)
                    }
        elif unit in 'ampere':
            return None

