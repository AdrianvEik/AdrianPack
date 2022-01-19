
import pandas as pd
import numpy as np
import math
from typing import Sized, Iterable


def csvead(path=None, cols=None, rows=None, delimiter=';',
           head=True, start_row=None, output='dict', dtype='object'):
    """
            Parse csv data to dict.

            Support for xlsx and csv, selecting specific columns and/or rows,
            specifying header and delimiter. Dependencies: pandas and numpy.

            :rtype: object
            :param path:
                Path of the file. Always str
            :param cols:
                Input in a tuple for more than one column or in int/str for one col
                input in (int/str, int/str) OR int/str OR ((int, label), (int, label))
                OR a combination.

                Used to specify the index/label of columns to append to dict
                Note: keys will be ints or strings.
            :param rows:
                Input in a tuple. ((row, label), (row, label), ...)
                OR (row, row, ...) OR combination of ((row, label), row)

                Used to specify the index of the row (int) and give a label (str) to
                append to dict. Note, keys of the rows will be strings.
            :param head:
                Boolean, default True change if the csv file doesn't have an header.
            :param delimiter:
                Default: ';', change for custom delimiter must be str.
            :return:
                Dict with all or specified rows/cols.
            """

    Data_reader(path=path, cols=cols, rows=rows, delimiter=delimiter,
                head=head, start_row=start_row, output=output, dtype=dtype)()
    return None


class Data_reader:
    """
        Parse csv data to dict.

        Support for xlsx and csv, selecting specific columns and/or rows,
        specifying header and delimiter. Dependencies: pandas and numpy.

        :rtype: object
        :param path:
            Path of the file. Always str
        :param cols:
            Input in a tuple for more than one column or in int/str for one col
            input in (int/str, int/str) OR int/str OR ((int, label), (int, label))
            OR a combination.

            Used to specify the index/label of columns to append to dict
            Note: keys will be ints or strings.
        :param rows:
            Input in a tuple. ((row, label), (row, label), ...)
            OR (row, row, ...) OR combination of ((row, label), row)

            Used to specify the index of the row (int) and give a label (str) to
            append to dict. Note, keys of the rows will be strings.
        :param head:
            Boolean, default True change if the csv file doesn't have an header.
        :param delimiter:
            Default: ';', change for custom delimiter must be str.
        :return:
            Dict with all or specified rows/cols.
        """

    # TODO: documentation, docstrings and write support.
    def __init__(self, path=None, cols=None, rows=None, delimiter=';',
                 head=True, start_row=None, output='dict', dtype='object'):
        # TODO: csv_path optioneel maken
        self.cols = cols
        self.rows = rows
        self.path = path
        self.delimiter = delimiter
        self.dtype = dtype
        if path is not None:
            with open(path, mode='r') as f:
                if start_row is not None:
                    self.test_inp(start_row, int, 'start row')
                else:
                    start_row = 0
                if path.split('.')[1] in ('csv', 'txt'):
                    df = pd.read_csv(f, delimiter=self.delimiter,
                                     skiprows=range(start_row))
                elif path.split('.')[1] == 'xlsx':
                    df = pd.read_excel(path, skiprows=range(start_row))
                else:
                    raise Exception('Expected .csv or .xlsx, got .' +
                                    path.split('.')[1])

                if head:
                    heads = list(df.columns)
                else:
                    # Make a numerical list instead of headers
                    heads = [i for i in range(len(list(df.columns)))]

                    # Add the first row at the top
                    df.loc[-1] = list(df.columns)
                    df.index = df.index + 1
                    df = df.sort_index()
                f.close()
            self.head = head
            self.heads = heads
            self.arr = df.to_numpy()
            self.arr_t = self.arr.transpose()
            self.output = output

            self.data_dict = dict()

    def __call__(self, read_all=False):
        if self.cols is not None:
            self.read_cols()

        if self.rows is not None:
            self.read_rows()

        if (self.rows is None and self.cols is None) or read_all:
            for i in range(len(self.heads)):
                self.data_dict[self.heads[i]] = self.arr_t[i][
                    ~pd.isnull(self.arr_t[i])].astype(self.dtype)

        if self.output is not None:
            self.test_inp(self.output, str, 'Output', True)
            if self.output in ('dict', 'dictionary'):
                return self.data_dict
            elif self.output in ('df', 'dataframe'):
                df = pd.DataFrame(dict([(i, pd.Series(j))
                                        for i, j in self.data_dict.items()]))
                return df
            elif self.output in ('matrix', 'array', 'arr', 'numpy', 'np'):
                matrix = np.zeros(
                    (len(self.data_dict),
                     max([len(i) for i in self.data_dict.values()]))
                )
                matrix.fill(np.nan)

                # Fill the array with values.
                for i in range(len(self.data_dict)):
                    for j in range(len(list(self.data_dict.values())[i])):
                        matrix[i, j] = list(self.data_dict.values())[i][j]
                return matrix.transpose()
            else:
                raise Exception('Expected output in df or dict not in %s'
                                % self.output)

    def read_cols(self):
        """
        Read specific columns and append to the dict
        :return:
        Data_dict
        """
        # Correctly format the columns
        if not isinstance(self.cols, (tuple, list)):
            self.cols = (self.cols,)

        for i in self.cols:
            if isinstance(i, int):
                # Test if the input isn't out of range.
                try:
                    self.arr_t[i]
                except IndexError:
                    print('\x1b[31m' +
                          'IndexError: There are %s columns, %s is out of'
                          ' bounds. Continuing...  ' % (i, len(self.arr_t))
                          + '\x1b[0m')
                    continue

                if self.heads[i] not in self.data_dict:
                    self.data_dict[self.heads[i]] = self.arr_t[i][
                        ~pd.isnull(self.arr_t[i])
                    ].astype(self.dtype)
                else:
                    print('\x1b[33m' +
                          'Column with index %s already added to dict, '
                          'Continuing...  ' % i
                          + '\x1b[0m')
                    continue

            elif isinstance(i, (tuple, list)):
                if len(i) == 2:
                    if isinstance(i[0], int):
                        if self.heads[i[0]] not in self.data_dict:
                            self.data_dict[i[1]] = self.arr_t[i[0]][
                                ~pd.isnull(self.arr_t[i[0]])
                            ].astype(self.dtype)
                        else:
                            print('\x1b[33m' +
                                  'Column "%s" already added to dict,'
                                  ' Continuing...  ' % i[1]
                                  + '\x1b[0m')
                            continue
                    else:
                        self.test_inp(i[0], int, 'columns', True)
                else:
                    raise Exception('Expected tuple of length 2 got length: %s'
                                    % len(i))
            elif isinstance(i, str) and self.head is True:
                try:
                    assert i in self.heads
                except AssertionError:
                    print('\x1b[31m' +
                          'LookupError: "%s" not in csv file continuing...' % i
                          + '\x1b[0m')
                    continue

                if i not in self.data_dict:
                    self.data_dict[i] = self.arr_t[
                        self.heads.index(i)
                    ][
                        ~pd.isnull(self.arr_t[self.heads.index(i)])
                    ].astype(self.dtype)
                else:
                    print('\x1b[33m' +
                          'Column "%s" already added to dict,'
                          ' Continuing...  ' % i
                          + '\x1b[0m')
                    continue
            else:
                if self.head:
                    self.test_inp(i, (int, list, tuple, str), 'columns')
                else:
                    self.test_inp(i, (int, list, tuple), 'columns')

        return self.data_dict

    def read_rows(self):
        """
        Read specific rows.
        :return:
            Data_dict
        """
        # Format the rows
        if not isinstance(self.rows, (tuple, list)):
            self.rows = ((self.rows,),)
        row_list = list()
        for j in self.rows:
            if not isinstance(j, (tuple, list)):
                row_list.append((j,))
            else:
                row_list.append(j)

        for i in row_list:
            if len(i) == 2:
                if isinstance(i[0], int) \
                        and isinstance(i[1], (str, int, float, bytes)):
                    try:
                        self.arr[i[0]]
                    except IndexError:
                        print('\x1b[31m' +
                              'IndexError: There are %s rows,'
                              ' %s is out of bounds.' % (len(self.arr), i[0])
                              + '\x1b[0m')
                        continue

                    if not i[1] in self.data_dict:
                        self.data_dict[i[1]] = self.arr[i[0]].astype(self.dtype)
                    else:
                        print('\x1b[33m' +
                              'Row %s already added to dict continuing...'
                              % i[1]
                              + '\x1b[0m')
                        continue
                else:
                    self.test_inp(i[0], int, 'rows', True)
            elif len(i) == 1:
                if isinstance(i[0], int):
                    try:
                        self.arr[i[0]]
                    except IndexError:
                        print('\x1b[31m' +
                              'IndexError: There are '
                              + str(len(self.arr))
                              + ' rows, ' + str(i[0]) +
                              ' is out of bounds. Continuing...  '
                              + '\x1b[0m')
                        continue

                    if not 'r_' + str(row_list.index(i)) in self.data_dict:
                        self.data_dict['r_' + str(row_list.index(i))] \
                            = self.arr[i[0]].astype(self.dtype)
                    else:
                        print('\x1b[33m' +
                              'Row with index ' + str(i[0]) + ' '
                              + 'already added to dict, Continuing...  '
                              + '\x1b[0m')
                        continue
                else:
                    self.test_inp(i, int, 'row', True)
            else:
                raise Exception('Expected input of length 1 or 2 got %s'
                                % len(i))
        return self.data_dict

    def writer(self, cols=None, rows=None, single=None, new_file=None,
               table=None):
        """
        Write columns and/or rows in a csv or txt file.
        :param cols:
            input (col nr, iterable)
            OR
            iterable
        :param rows:
            input (row nr, iterable)
            OR
            iterable
        :param single:
            input ((row, col), int;str;float;byte)
        :param table:
            input (pandas df or numpy matrix, (optional positional args))
            e.g. df or (df, 'row=3') or (df, 'col=4') or
            (df, ('row=2', 'col=3'))
            NOTE the position is the position of the top left entry.
        :param new_file:
            input (file name, type) OR filename
            e.g. (DataSheet, csv) or (Datasheet.csv)
        """

        if new_file is not None and isinstance(new_file, tuple):
            self.path = str(new_file[0]) + '.' + str(new_file[1])
        elif new_file is not None and isinstance(new_file, str):
            self.path = new_file
        elif new_file is not None:
            self.test_inp(new_file, (tuple, list, str), 'path', True)

        if new_file is None:
            self.output = 'df'
            df = self.__call__(read_all=True)
            num_list, head = False, []
            if cols is not None:
                if isinstance(cols, (tuple, list)):
                    for i in cols:
                        if isinstance(i, dict):
                            head += list(i.keys())
                            values = list(i.values())
                            index = None
                            for j in range(len(values)):
                                if not isinstance(values[j], (list, tuple)):
                                    values[j] = [values[j]]
                            max_value = [len(j) for j in values] \
                                .index(max([len(j) for j in values]))
                            for j in range(len(list(i.values())[max_value])):
                                if math.isnan(list(i.values())[max_value][j]):
                                    index = j
                                    print(j, "J")
                                else:
                                    pass
                            if index is None:
                                index = max([len(j) for j in values])

                        elif isinstance(i, (list, tuple)):
                            head.append(cols.index(i))
                        else:
                            num_list = True
                            head = cols[0]
                    if num_list:
                        pass
                    else:
                        for i in range(len(head)):
                            if isinstance(head[i], tuple):
                                self.test_inp(head[i][0], str, 'header')
                                self.test_inp(head[i][1], int, 'location')

                                self.data_dict[self.heads[head[i][1]]] = \
                                    values[i]
                                df = pd.DataFrame(dict([(i, pd.Series(j))
                                                        for i, j in
                                                        self.data_dict.items()]))
                                df.rename(columns={head[i][1]: head[i][0]})
                                df = df.truncate(after=index - 1)
            df.to_csv(self.path, sep=self.delimiter, index=False)

    @staticmethod
    def test_inp(test_obj, test_if, name_inp, value=False):
        """
        Test a value if it returns false raise an exception
        :param: test_obj
        Object to be tested.
        :param:test_if
        The input that is tested to be equal as. (in int, str, double etc)
        :param: value
        Bool, if True the exception also shows test_obj not recommended for
        long lists.
        :param: name_inp
        String, the formal name of the object shown in exception.
        """
        assert isinstance(name_inp, str)
        try:
            assert isinstance(test_obj, test_if)
        except AssertionError:
            if not isinstance(test_if, tuple):
                if not value:
                    raise TypeError(
                        'Expected %s for %s but got %s' %
                        (test_if.__name__,
                         name_inp, test_obj.__class__.__name__)
                    )
                else:
                    raise TypeError(
                        'Expected %s for %s but got type %s with'
                        ' value: %s' %
                        (test_if.__name__, name_inp,
                         test_obj.__class__.__name__, test_obj)
                    )
            else:
                test_if = [i.__name__ for i in test_if]
                if not value:
                    raise TypeError(
                        'Expected %s for %s but got %s' %
                        (', '.join(test_if), name_inp,
                         test_obj.__class__.__name__)
                    )
                else:
                    raise TypeError(
                        'Expected %s for %s but got type %s with'
                        ' value: %s' %
                        (', '.join(test_if), name_inp,
                         test_obj.__class__.__name__, test_obj)
                    )
        return None

    @staticmethod
    def filter_nan(arr: Iterable) -> np.array:
        """
        Filter nan's out of array
        :param arr: Numpy array
        :return: Filtered array
        """
        print(arr)
        print('filtered: ', np.array[lambda v: v == v, arr])
        return np.array[lambda v: v == v, arr]
