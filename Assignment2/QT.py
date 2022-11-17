import random
import numpy as np
from Auxiliary import LoadData
from GibbsSampling import Gibbs

k = np.array(["aa","bb","cc"])
v = ["a","b","c"]
ddict = dict(zip(k,v))


a = np.array([10,1,1000,100,100000000,100])
b = np.log10(a)
print(b)