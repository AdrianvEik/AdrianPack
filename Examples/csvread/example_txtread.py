
from AdrianPack.Fileread import Fileread, csvread
from AdrianPack.Aplot import Default
from AdrianPack.Extra import Compress_array

from TN_code.plotten.TISTNplot import TNFormatter

# Read the TextData.txt file and return columns 1 and 4 as a numpy array.
Data_txt = Fileread(path=r"Data\TextData.txt", output="numpy", cols=[1, 4])()
# Plot the numpy array returned by Fileread using AdrianPack.Aplot.Default
Default(Data_txt)()

Data_csv = Fileread(path=r"Data\CsvData.csv", output="numpy", cols=[6, 7], start_row=1)()
Default(Data_csv)()

Data_xlsx = Fileread(path=r"Data\ExcelData.xlsx", output="numpy", cols=["P", "U"])()
Default(Data_xlsx)()


