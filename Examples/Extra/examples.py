
from AdrianPack.Extra import trap_int, Compress_array
from AdrianPack.Aplot import Default
from AdrianPack.Fileread import Fileread
import numpy as np

noise = np.random.random(500) * 100

# Create a list with 200 x values between -10 and 10
x = np.linspace(-10, 10, 500)
# Calculate values of the function y = 4x + 2
y = 4 * x**2 + 2 * x + 1 + noise

x_err = np.full(x.shape, 0.1)
y_err = np.full(x.shape, 0.05)

area, area_err = trap_int(x, y, x_err=x_err, y_err=y_err)

# COMPRESS EXAMPLE
x_c, y_c = Compress_array([x, y], width_ind=50)

plot = Default(x_c, y_c, colour="C1", data_label="Compressed", degree=2, legend_loc="upper center")
plot_add = Default(x, y, add_mode=True, colour="C0", data_label="Original", degree=2)
plot += plot_add
plot()

Data_txt = Fileread(path=r"TextData.txt", output="numpy", cols=[1, 4])()

Data_compress = Compress_array([Data_txt[:, 0], Data_txt[:, 1]],
                               width_ind=100)
plot = Default(Data_compress[0], Data_compress[1], colour="C1",
               connecting_line=True, data_label="Compressed",
               x_label="Time $t$ [s]", y_label="Acceleration $a$ [$\mathrm{ms^{-1}}$]",
               save_as="compress_plot.png")
plot_1 = Default(Data_txt[:, 0], Data_txt[:, 1], add_mode=True, colour="C0", data_label="Original")

plot += plot_1
plot()
