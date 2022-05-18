
from AdrianPack.Fileread import Fileread, csvread
from AdrianPack.Aplot import Default

Data_txt = csvread(path=r"Data\TextData.txt", output="numpy", cols=[1, 4])()
Default(Data_txt)()

Data_csv = Fileread(path=r"Data\CsvData.csv", output="numpy", cols=[6, 7], start_row=1)()
Default(Data_csv)()

Data_xlsx = Fileread(path=r"Data\ExcelData.xlsx", output="numpy", cols=["P", "U"])()
Default(Data_xlsx)()

Data_combination = Fileread(path=[r"Data\TextData.txt", r"Data\CsvData.csv", r"Data\ExcelData.xlsx"],
                        cols={0: [(1, "x"), (4, "y")], 1: [0, 1], 2:[0, 2]})()
Default(Data_combination["x"], Data_combination["y"])()