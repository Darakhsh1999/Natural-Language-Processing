import random
import numpy as np
from Auxiliary import LoadData
from GibbsSampling import Gibbs

a = np.arange(3)
k = np.array(["aa","bb","cc"])
v = ["a","b","c"]
ddict = dict(zip(k,v))

data = LoadData(1550)
gibbs = Gibbs(data, 1550, 10, 10)



def IsCool(name, n):
    print(f"{name} is very cool"+"!"*n)


IsCool("Pambaulettox", 197)