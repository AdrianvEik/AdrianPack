
from AdrianPack.Aplot import Default

# Import data from sample library
from Data.sample_data_lib import angle_data
from Data.sample_data_lib import LH_n115_TE_r, LH_n2_TE_r, LH_n3_TE_r, LH_n4_TE_r
from Data.sample_data_lib import LH_n115_TE_t, LH_n2_TE_t, LH_n3_TE_t, LH_n4_TE_t

# Create some plots
TE_115r = Default(angle_data, LH_n115_TE_r)
TE_115r()
