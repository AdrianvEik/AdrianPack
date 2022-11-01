
from AdrianPack.Fileread import Fileread

# SAMPLE DATA TO USE FOR EXAMPLES
# A | R | T | RT
# n2 = 1.15
LH_n115_TE = Fileread("SampleData/Para_1,15_Ez.csv", cols=[0, 1, 2, 3])()

angle_data = LH_n115_TE["A"]
LH_n115_TE_r = LH_n115_TE["R"]
LH_n115_TE_t = LH_n115_TE["T"]
LH_n115_TE_rt = LH_n115_TE["RT"]

LH_n115_TM = Fileread("SampleData/Para_1,15_Hz.csv", cols=[0, 1, 2, 3])()

LH_n115_TM_r = LH_n115_TM["R"]
LH_n115_TM_t = LH_n115_TM["T"]
LH_n115_TM_rt = LH_n115_TM["RT"]

# n2 = 1.3
LH_n130_TE = Fileread("SampleData/Para_1,3_Ez.csv", cols=[0, 1, 2, 3])()

LH_n130_TE_r = LH_n130_TE["R"]
LH_n130_TE_t = LH_n130_TE["T"]
LH_n130_TE_rt = LH_n130_TE["RT"]

LH_n130_TM = Fileread("SampleData/Para_1,3_Hz.csv", cols=[0, 1, 2, 3])()

LH_n130_TM_r = LH_n130_TM["R"]
LH_n130_TM_t = LH_n130_TM["T"]
LH_n130_TM_rt = LH_n130_TM["RT"]

# n2 = 1.45
LH_n145_TE = Fileread("SampleData/Para_1,45_Ez.csv", cols=[0, 1, 2, 3])()

LH_n145_TE_r = LH_n145_TE["R"]
LH_n145_TE_t = LH_n145_TE["T"]
LH_n145_TE_rt = LH_n145_TE["RT"]

LH_n145_TM = Fileread("SampleData/Para_1,45_Hz.csv", cols=[0, 1, 2, 3])()

LH_n145_TM_r = LH_n145_TM["R"]
LH_n145_TM_t = LH_n145_TM["T"]
LH_n145_TM_rt = LH_n145_TM["RT"]

# n2 = 1.60
LH_n160_TE = Fileread("SampleData/Para_1,6_Ez.csv", cols=[0, 1, 2, 3])()

LH_n160_TE_r = LH_n160_TE["R"]
LH_n160_TE_t = LH_n160_TE["T"]
LH_n160_TE_rt = LH_n160_TE["RT"]

LH_n160_TM = Fileread("SampleData/Para_1,6_Hz.csv", cols=[0, 1, 2, 3])()

LH_n160_TM_r = LH_n160_TM["R"]
LH_n160_TM_t = LH_n160_TM["T"]
LH_n160_TM_rt = LH_n160_TM["RT"]

# n2 = 1.75
LH_n175_TE = Fileread("SampleData/Para_1,75_Ez.csv", cols=[0, 1, 2, 3])()

LH_n175_TE_r = LH_n175_TE["R"]
LH_n175_TE_t = LH_n175_TE["T"]
LH_n175_TE_rt = LH_n175_TE["RT"]

LH_n175_TM = Fileread("SampleData/Para_1,75_Hz.csv", cols=[0, 1, 2, 3])()

LH_n175_TM_r = LH_n175_TM["R"]
LH_n175_TM_t = LH_n175_TM["T"]
LH_n175_TM_rt = LH_n175_TM["RT"]

# n2 = 1.9
LH_n190_TE = Fileread("SampleData/Para_1,9_Ez.csv", cols=[0, 1, 2, 3])()

LH_n190_TE_r = LH_n190_TE["R"]
LH_n190_TE_t = LH_n190_TE["T"]
LH_n190_TE_rt = LH_n190_TE["RT"]

LH_n190_TM = Fileread("SampleData/Para_1,9_Hz.csv", cols=[0, 1, 2, 3])()

LH_n190_TM_r = LH_n190_TM["R"]
LH_n190_TM_t = LH_n190_TM["T"]
LH_n190_TM_rt = LH_n190_TM["RT"]

# n2 = 2
LH_n2_TE = Fileread("SampleData/Para_2_Ez.csv", cols=[0, 1, 2, 3])()

LH_n2_TE_r = LH_n2_TE["R"]
LH_n2_TE_t = LH_n2_TE["T"]
LH_n2_TE_rt = LH_n2_TE["RT"]

LH_n2_TM = Fileread("SampleData/Para_2_Hz.csv", cols=[0, 1, 2, 3])()

LH_n2_TM_r = LH_n2_TM["R"]
LH_n2_TM_t = LH_n2_TM["T"]
LH_n2_TM_rt = LH_n2_TM["RT"]

# n2 = 3
LH_n3_TE = Fileread("SampleData/Para_3_Ez.csv", cols=[0, 1, 2, 3])()

LH_n3_TE_r = LH_n3_TE["R"]
LH_n3_TE_t = LH_n3_TE["T"]
LH_n3_TE_rt = LH_n3_TE["RT"]

LH_n3_TM = Fileread("SampleData/Para_3_Hz.csv", cols=[0, 1, 2, 3])()

LH_n3_TM_r = LH_n3_TM["R"]
LH_n3_TM_t = LH_n3_TM["T"]
LH_n3_TM_rt = LH_n3_TM["RT"]

# n2 = 4
LH_n4_TE = Fileread("SampleData/Para_4_Ez.csv", cols=[0, 1, 2, 3])()

LH_n4_TE_r = LH_n4_TE["R"]
LH_n4_TE_t = LH_n4_TE["T"]
LH_n4_TE_rt = LH_n4_TE["RT"]

LH_n4_TM = Fileread("SampleData/Para_4_Hz.csv", cols=[0, 1, 2, 3])()

LH_n4_TM_r = LH_n4_TM["R"]
LH_n4_TM_t = LH_n4_TM["T"]
LH_n4_TM_rt = LH_n4_TM["RT"]
