# Dependecies
* Numpy
* Pandas

# What to use for?
* Data reading
* Data writing
* Plotting (In Ez_plotter)

**The data_reader class can be used to convert csv, xlsx and txt to numpy, 
dictionary or pandas objects and back.**
****
# How to read?
**Import the Data_reader class**  

from Data_reader import data_reader as dr

**Create the object and call dr(params)(). The np, pd or dict is only 
created when called.**

my_dict = dr(path="My_path.txt", output="dict")()

**The my_dict object contains the content in a dict with column headers as 
keys and column content in a list as value.**
****
# How to write?
**!!Writer currently only accepts dictionaries and only writes to CSV!!**  

**Call the writer object with**

dr(path=path).writer(cols={(col_name, col_pos): col_content})  
dr(path="My_path.csv").writer(cols={("some_key", 0): [a, list, with, objects]})

**This will fill the first row with the content in the list, this list can 
be any iterable object.**
name function is bugged the row will not have the same name as the key but 
col index.

